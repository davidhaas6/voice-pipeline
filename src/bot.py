import asyncio
import io
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import discord
import numpy as np
from dotenv import load_dotenv
from mistralai import Mistral

from src.audio_utils import resample_audio_poly

from .audio_utils import calculate_rms, pcm_to_wav
from .logger import get_logger
from .tts import TTSManager

logger = get_logger(__name__)

load_dotenv()

api_key = os.environ["MISTRAL_API_KEY"]
T2S_MODEL = "voxtral-mini-latest"
T2T_MODEL = "mistral-medium-2508"  # https://docs.mistral.ai/getting-started/models#premier-models

client = Mistral(api_key=api_key)


# Configuration
VAD_RMS_THRESHOLD = 300
VAD_RMS_CONTINUE_THRESHOLD = 100
SILENCE_DURATION = 1.2  # Seconds of silence before considering a "turn" finished
DISCORD_FRAME_SIZE_MS = 20
SAMPLING_RATE = 48000
CHANNELS = 2
SAMPLE_WIDTH = 2  # 16-bit PCM

# 48kHz * 2ch * 16-bit = 192000 bytes/sec
MIN_TURN_SECONDS = 0.25  # timer for "is user actually talking, or is that a blip"
MIN_TURN_BYTES = int(SAMPLING_RATE * CHANNELS * SAMPLE_WIDTH * MIN_TURN_SECONDS)

PLAYBACK_QUEUE_MAX_SECONDS = 25.0
PLAYBACK_QUEUE_MAXSIZE = int(PLAYBACK_QUEUE_MAX_SECONDS * 1000 / DISCORD_FRAME_SIZE_MS)


@dataclass
class VoiceState:
    vc: discord.VoiceClient
    sink: discord.sinks.WaveSink
    stop_event: threading.Event


@dataclass
class TurnState:
    user_audio_buffer: io.BytesIO = None
    user_speaking: bool = False
    last_voice_time: float = 0
    interrupted_for_current_turn: bool = False


@dataclass
class PlaybackState:
    queue: queue.Queue
    lock: threading.Lock = field(default_factory=threading.Lock)
    thread: threading.Thread = None
    generation: int = 0


@dataclass
class ChatState:
    history: deque[dict[str, str]] = field(default_factory=lambda: deque(maxlen=50))


@dataclass
class ServerContext:
    guild_id: int
    voice: VoiceState
    turn: TurnState
    playback: PlaybackState
    chat: ChatState
    processing_task: asyncio.Task = None
    pipeline_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    tts: TTSManager = field(default_factory=TTSManager)

    async def cleanup(self):
        """Unified cleanup for this server context."""
        logger.info(f"Cleaning up context for guild {self.guild_id}")
        self.voice.stop_event.set()

        # 1. Stop recording ASAP so the library's internal recv_audio thread exits
        # before we disconnect and close the socket.
        try:
            if (
                self.voice.vc
                and self.voice.vc.is_connected()
                and getattr(self.voice.vc, "recording", False)
            ):
                self.voice.vc.stop_recording()
        except Exception as e:
            if "Not currently recording" not in str(e):
                logger.error(
                    f"Error signaling stop_recording for guild {self.guild_id}: {e}"
                )
            else:
                logger.debug(f"Recording already stopped for guild {self.guild_id}")

        # 2. Stop TTS
        try:
            if self.tts:
                self.tts.stop()
        except Exception as e:
            logger.error(f"Error stopping TTS for guild {self.guild_id}: {e}")

        # 3. Flush playback queue
        while not self.playback.queue.empty():
            try:
                self.playback.queue.get_nowait()
            except queue.Empty:
                break

        # 4. Stop playback thread
        # We wait for the playback thread to exit its loop (so it doesn't try to send
        # to a closed socket) before we disconnect. timeout=0.5 is safe since it's
        # a 20ms loop that checks stop_event.
        if self.playback.thread and self.playback.thread.is_alive():
            await asyncio.to_thread(self.playback.thread.join, timeout=0.5)

        # 5. Disconnect from voice
        try:
            if self.voice.vc and self.voice.vc.is_connected():
                await self.voice.vc.disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting from guild {self.guild_id}: {e}")

        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass


class DiscordVoiceBot(discord.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contexts: dict[int, ServerContext] = {}
        self.system_prompt = """You are **Sponge**, a chill uncensored chatbot speaking with your friends in a Discord voice channel. Keep replies short: a few words or a sentence. Use plain words and contractions. No emojis, no bullet points, no special characters. Avoid formatting. No moralizing, lecturing, or over-explaining. Don't be corny or too serious. If asked to speak in a language with a different alphabet, respond in phonetic English spelling of your response (e.g. "Ni hao" instead of "你好"). This will be read aloud, so output natural dialogue.\nThis is a group chat, so don't reply to everything. Infer if you are being directly addressed. If you are not addressed, just output the string 'Silence'. Quality over quantity.""".strip()

    async def start_in_voice(self, ctx: discord.ApplicationContext):
        """Joins a voice channel and starts the pipeline."""
        if not ctx.author.voice:
            await ctx.followup.send("You're not in a voice channel!", ephemeral=True)
            return

        guild_id = ctx.guild.id
        if guild_id in self.contexts:
            await self.stop_in_voice(guild_id)

        try:
            vc = await ctx.author.voice.channel.connect(timeout=20, reconnect=True)
        except Exception as e:
            logger.error(f"Failed to connect to voice: {e}")
            await ctx.followup.send(
                "Failed to connect to your voice channel.", ephemeral=True
            )
            return

        # Wait for the voice client to be fully connected before recording
        count = 0
        while not vc.is_connected() and count < 100:
            await asyncio.sleep(0.1)
            count += 1

        if not vc.is_connected():
            await ctx.followup.send(
                "Failed to connect to voice channel within timeout.", ephemeral=True
            )
            return

        sink = discord.sinks.WaveSink()

        async def finished_callback(sink, *args):
            pass

        vc.start_recording(sink, finished_callback)

        stop_event = threading.Event()
        playback_queue = queue.Queue(maxsize=PLAYBACK_QUEUE_MAXSIZE)

        context = ServerContext(
            guild_id=guild_id,
            voice=VoiceState(vc=vc, sink=sink, stop_event=stop_event),
            turn=TurnState(),
            playback=PlaybackState(queue=playback_queue),
            chat=ChatState(),
        )

        context.processing_task = asyncio.create_task(
            self.continuous_audio_processing(context)
        )
        context.playback.thread = threading.Thread(
            target=self.playback_worker, args=(context,), daemon=True
        )
        context.playback.thread.start()

        self.contexts[guild_id] = context
        await ctx.followup.send(
            f"Joined {ctx.author.voice.channel.name}!", ephemeral=True
        )

    async def stop_in_voice(self, guild_id: int):
        """Single cleanup path used by /leave, shutdown, disconnect handler."""
        context = self.contexts.get(guild_id)
        if not context:
            return

        await context.cleanup()
        if guild_id in self.contexts:
            del self.contexts[guild_id]

    async def close(self):
        """Cleanup all voice contexts before closing."""
        logger.info("Bot shutting down, cleaning up all voice contexts...")
        # Use list() to avoid dictionary size changed during iteration
        for guild_id in list(self.contexts.keys()):
            await self.stop_in_voice(guild_id)
        await super().close()

    # --- MODEL COMPONENTS ---

    async def transcribe_audio(self, audio_data: bytes) -> str:
        # https://docs.mistral.ai/capabilities/audio_transcription#transcription
        logger.info("Transcribing audio...")
        try:
            transcription_response = await client.audio.transcriptions.complete_async(
                model=T2S_MODEL,
                file={
                    "file_name": "audio.wav",
                    "content": audio_data,
                },
                context_bias=["sponge", "bot"],
            )
            transcript = transcription_response.text
            logger.info(f"Transcript: {transcript}")
            return transcript
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

    async def decide_to_respond(self, transcript: str) -> bool:
        """Deciding if the bot should speak"""
        return True

    async def generate_response_text(self, chat_history: deque[dict[str, str]]) -> str:
        """Text-to-Text (T2T)"""
        # The API expects a list, and we'll take the full sliding window from the deque
        chat_context = list(chat_history)
        chat_context.insert(
            0,
            {
                "role": "system",
                "content": self.system_prompt,
            },
        )
        try:
            chat_response = await client.chat.complete_async(
                model=T2T_MODEL, messages=chat_context, temperature=1, max_tokens=200
            )
        except Exception as e:
            logger.error(f"Error during T2T: {e}")
            return "Silence"
        content = chat_response.choices[0].message.content or ""
        return content.strip()

    async def generate_response_audio(self, context: ServerContext, text: str):
        """Generates TTS audio and streams it to the playback queue."""
        sanitized_text = text.strip().lower()
        if not sanitized_text or (
            "silence" in sanitized_text and len(sanitized_text) < 10
        ):
            logger.info("The bot remains silent.")
            return
        logger.debug(f'Requesting speech for: "{text[:10]}..."')

        my_gen = self._new_speak_epoch(context, flush_queue=True)

        def on_audio_chunk(chunk_np: np.ndarray):
            # Fast drop if invalidated (barge-in or newer bot utterance)
            if my_gen != context.playback.generation:
                return

            # pocket_tts provides float32 mono at tts.sample_rate
            resampled = resample_audio_poly(chunk_np, context.tts.sample_rate, 48000)
            resampled = np.clip(resampled, -1.0, 1.0)
            int16_data = (resampled * 32767).astype(np.int16)

            stereo_data = np.repeat(int16_data[:, np.newaxis], 2, axis=1).flatten()

            # Enqueue in 20ms frames
            # 48000 Hz * 20ms = 960 samples per channel
            # 2 channels * 2 bytes = 4 bytes per sample
            # Total 3840 bytes per 20ms frame
            frame_size_samples = int(48000 * (DISCORD_FRAME_SIZE_MS / 1000))
            frame_size_bytes = frame_size_samples * CHANNELS * SAMPLE_WIDTH
            raw_bytes = stereo_data.tobytes()

            with context.playback.lock:
                if my_gen != context.playback.generation:
                    return
                for i in range(0, len(raw_bytes), frame_size_bytes):
                    frame = raw_bytes[i : i + frame_size_bytes]
                    if len(frame) < frame_size_bytes:
                        frame += b"\x00" * (frame_size_bytes - len(frame))
                    while True:
                        try:
                            context.playback.queue.put_nowait(frame)
                            break
                        except queue.Full:
                            try:
                                context.playback.queue.get_nowait()  # drop oldest frame - note this could cause skipping
                            except queue.Empty:
                                break

        context.tts.speak(text, callback=on_audio_chunk)

    # --- PIPELINE ORCHESTRATION ---

    async def run_pipeline(self, context: ServerContext, audio_data: bytes):
        """Coordinates the STT -> T2T -> TTS flow"""
        async with context.pipeline_lock:
            pipeline_start = time.perf_counter()
            try:
                transcript = await self.transcribe_audio(audio_data)
                stt_done = time.perf_counter()
                logger.debug(f"STT took {(stt_done - pipeline_start) * 1000:.2f}ms")

                if not transcript:
                    return
                context.chat.history.append({"role": "user", "content": transcript})
                if not await self.decide_to_respond(transcript):
                    return

                response_text = await self.generate_response_text(context.chat.history)
                t2t_done = time.perf_counter()
                logger.debug(f"T2T took {(t2t_done - stt_done) * 1000:.2f}ms")

                context.chat.history.append(
                    {"role": "assistant", "content": response_text or "Silence"}
                )

                await self.generate_response_audio(context, response_text)
            except Exception as e:
                logger.error(f"Error in pipeline: {e}")

    # --- AUDIO INPUT ---

    async def continuous_audio_processing(self, context: ServerContext):
        """Continuously polls the sink for new audio and handles VAD"""
        logger.info(f"Started audio processing for guild {context.guild_id}")

        while not context.voice.stop_event.is_set():
            await asyncio.sleep(0.1)

            for user_id, audio in list(context.voice.sink.audio_data.items()):
                audio.file.seek(0)
                data = audio.file.read()
                if not data:
                    continue
                audio.file.seek(0)
                audio.file.truncate()

                rms = calculate_rms(data)
                logger.debug(f"RMS: {rms:.4f}")  # Uncomment this to tune VAD_THRESHOLD

                threshold = (
                    VAD_RMS_CONTINUE_THRESHOLD
                    if context.turn.user_speaking
                    else VAD_RMS_THRESHOLD
                )
                if rms > threshold:
                    if not context.turn.user_speaking:
                        logger.info("User started speaking...")
                        context.turn.user_speaking = True
                        context.turn.user_audio_buffer = io.BytesIO()
                        context.turn.interrupted_for_current_turn = (
                            False  # reset for this turn
                        )

                    context.turn.user_audio_buffer.write(data)
                    context.turn.last_voice_time = time.time()

                    if (
                        not context.turn.interrupted_for_current_turn
                        and context.turn.user_audio_buffer.tell() >= MIN_TURN_BYTES
                    ):
                        logger.info("Sustained speech detected. Interrupting bot.")
                        context.turn.interrupted_for_current_turn = True
                        self._invalidate_playback(context, flush_queue=True)

            if context.turn.user_speaking:
                # Check if silence duration has passed
                if time.time() - context.turn.last_voice_time > SILENCE_DURATION:
                    audio_data = context.turn.user_audio_buffer.getvalue()
                    context.turn.user_speaking = False
                    context.turn.interrupted_for_current_turn = False  # reset

                    if len(audio_data) < MIN_TURN_BYTES:
                        logger.info(
                            f"Dropping short utterance ({len(audio_data)} bytes)."
                        )
                        context.turn.user_audio_buffer = None
                        continue

                    # Consider using to_mono_16k_wav for better optimization.
                    # will need to upload some reshape and stereo assumptions
                    wav_data = pcm_to_wav(
                        audio_data, CHANNELS, SAMPLE_WIDTH, SAMPLING_RATE
                    )

                    logger.info("User finished speaking. Triggering pipeline...")
                    asyncio.create_task(self.run_pipeline(context, wav_data))
                    context.turn.user_audio_buffer = None

    # --- PLAYBACK ---
    def _invalidate_playback(self, context: ServerContext, *, flush_queue: bool = True):
        """
        Invalidates any in-flight TTS callbacks and optionally flushes queued audio frames.
        """
        with context.playback.lock:
            context.playback.generation += 1
            try:
                context.tts.stop()
            except Exception:
                pass

            if flush_queue:
                while not context.playback.queue.empty():
                    try:
                        context.playback.queue.get_nowait()
                    except queue.Empty:
                        break

    def _new_speak_epoch(
        self, context: ServerContext, *, flush_queue: bool = False
    ) -> int:
        """
        Starts a new epoch for a fresh bot utterance.
        By default we don't flush the queue here (optional).
        """
        with context.playback.lock:
            context.playback.generation += 1
            if flush_queue:
                while not context.playback.queue.empty():
                    try:
                        context.playback.queue.get_nowait()
                    except queue.Empty:
                        break
            return context.playback.generation

    def playback_worker(self, context: ServerContext):
        logger.info(f"Playback worker started for guild {context.guild_id}")

        TARGET_INTERVAL = DISCORD_FRAME_SIZE_MS / 1000.0
        next_frame_time = time.perf_counter()

        while not context.voice.stop_event.is_set():
            # If we got disconnected, bail fast
            if not context.voice.vc or not context.voice.vc.is_connected():
                break

            now = time.perf_counter()

            # If we're very late (e.g. stalled for 100ms+), resync instead of trying to catch up
            if now - next_frame_time > 0.1:
                next_frame_time = now

            sent = False
            try:
                frame = context.playback.queue.get_nowait()
            except queue.Empty:
                frame = None

            if frame is not None:
                try:
                    context.voice.vc.send_audio_packet(frame, encode=True)
                    sent = True
                except OSError:
                    logger.warning(
                        "Socket already closed (WinError 10038) or similar: exit thread quietly"
                    )
                    break
                except Exception:
                    logger.error("Error sending audio packet")
                    break

            if sent:
                # Only advance schedule when we actually sent audio.
                next_frame_time += TARGET_INTERVAL
            else:
                # No audio to send: don't march time forward forever.
                # Resync so the next real frame doesn't trigger catch-up jitter.
                next_frame_time = now + TARGET_INTERVAL

            # Hybrid sleep (your original idea), but safe
            while True:
                now = time.perf_counter()
                remaining = next_frame_time - now
                if remaining <= 0:
                    break
                if context.voice.stop_event.is_set():
                    return
                if remaining > 0.005:
                    time.sleep(remaining - 0.003)
                # else busy-wait for sub-ms precision
