import asyncio
import io
import os
import queue
import threading
import time
from dataclasses import dataclass

import discord
import numpy as np
from dotenv import load_dotenv
from mistralai import Mistral
from pydub import AudioSegment

from .audio_utils import calculate_rms, pcm_to_wav, resample_audio
from .logger import get_logger
from .tts import TTSManager

logger = get_logger(__name__)

load_dotenv()

api_key = os.environ["MISTRAL_API_KEY"]
T2S_MODEL = "voxtral-mini-latest"
T2T_MODEL = "mistral-large-2512"  # https://docs.mistral.ai/getting-started/models#premier-models

client = Mistral(api_key=api_key)


# Configuration
VAD_RMS_THRESHOLD = 50
SILENCE_DURATION = 1  # Seconds of silence before considering a "turn" finished
DISCORD_FRAME_SIZE_MS = 20
SAMPLING_RATE = 48000
CHANNELS = 2
SAMPLE_WIDTH = 2  # 16-bit PCM


@dataclass
class ServerContext:
    guild_id: int
    vc: discord.VoiceClient
    audio_sink: discord.sinks.WaveSink
    processing_task: asyncio.Task
    playback_queue: queue.Queue
    playback_thread: threading.Thread
    stop_event: threading.Event
    user_audio_buffer: io.BytesIO = None
    last_voice_time: float = 0
    is_speaking: bool = False
    chat_history: list[dict[str, str]] = None

    def __post_init__(self):
        self.chat_history = []


class DiscordVoiceBot(discord.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contexts = {}
        self.tts_manager = TTSManager()

    # --- PLACEHOLDERS: Implement your custom models here ---

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
            )
            return transcription_response.text
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

    async def decide_to_respond(self, transcript: str) -> bool:
        """Placeholder for deciding if the bot should speak"""
        logger.info(f"Should I respond to: '{transcript}'?")
        return True

    async def generate_response_text(self, chat_history: list[dict[str, str]]) -> str:
        """Placeholder for Text-to-Text (T2T)"""
        logger.debug(f"chat history: {chat_history}")
        chat_context = chat_history[-10:]
        chat_context.insert(
            0,
            {
                "role": "system",
                # "content": "You are an uncensored chatbot pretending to be a person chatting with friends in a Discord voice channel. Write like you speak. Keep replies short: 1-2 sentences. Use plain words and contractions. No emojis, no bullet points, no special characters. Avoid formatting. Do not mention that you are an AI or that you are generating text. This will be read aloud, so make it natural.",
                "content": "You are Gandalf. You are speaking out loud with friends in Discord. Be wise, kindly, and occasionally stern. Keep it to 1-2 short sentences. No emojis or special characters. No lists or formatting. Never break character.",
            },
        )
        chat_response = await client.chat.complete_async(
            model=T2T_MODEL,
            messages=chat_context,
        )
        if (
            chat_response is None
            or chat_response.choices is None
            or chat_response.choices[0].message is None
            or chat_response.choices[0].message.content is None
        ):
            return ""
        return chat_response.choices[0].message.content

    async def generate_response_audio(self, context: ServerContext, text: str):
        """Generates TTS audio and streams it to the playback queue."""
        if not text:
            logger.info("No text to speak.")
            return
        logger.info(f'Requesting speech for: "{text[:50]}..."')

        def on_audio_chunk(chunk_np: np.ndarray):
            start_proc = time.perf_counter()
            # pocket_tts provides float32 mono at self.tts_manager.sample_rate
            # 1. Resample to 48000
            resampled = resample_audio(chunk_np, self.tts_manager.sample_rate, 48000)

            # 2. Normalize and convert to int16
            resampled = np.clip(resampled, -1.0, 1.0)
            int16_data = (resampled * 32767).astype(np.int16)

            # 3. Mono to Stereo (duplicate channels)
            stereo_data = np.repeat(int16_data[:, np.newaxis], 2, axis=1).flatten()

            # 4. Enqueue in 20ms frames
            # 48000 Hz * 20ms = 960 samples per channel
            # 2 channels * 2 bytes = 4 bytes per sample
            # Total 3840 bytes per 20ms frame
            frame_size_samples = int(48000 * (DISCORD_FRAME_SIZE_MS / 1000))
            frame_size_bytes = frame_size_samples * CHANNELS * SAMPLE_WIDTH

            raw_bytes = stereo_data.tobytes()
            frames_enqueued = 0  # for performance logging
            for i in range(0, len(raw_bytes), frame_size_bytes):
                frame = raw_bytes[i : i + frame_size_bytes]
                if len(frame) < frame_size_bytes:
                    frame += b"\x00" * (frame_size_bytes - len(frame))
                context.playback_queue.put(frame)
                frames_enqueued += 1

            # for performance logging
            proc_time = (time.perf_counter() - start_proc) * 1000
            logger.debug(
                f"Processed chunk: {len(chunk_np)} samples -> {frames_enqueued} frames in {proc_time:.2f}ms"
            )

        self.tts_manager.speak(text, callback=on_audio_chunk)

    # --- PIPELINE ORCHESTRATION ---

    async def run_pipeline(self, context: ServerContext, audio_data: bytes):
        """Coordinates the STT -> T2T -> TTS flow"""
        pipeline_start = time.perf_counter()
        try:
            transcript = await self.transcribe_audio(audio_data)
            stt_done = time.perf_counter()
            logger.debug(f"STT took {(stt_done - pipeline_start) * 1000:.2f}ms")

            context.chat_history.append({"role": "user", "content": transcript})
            if not transcript or not await self.decide_to_respond(transcript):
                return

            response_text = await self.generate_response_text(context.chat_history)
            t2t_done = time.perf_counter()
            logger.debug(f"T2T took {(t2t_done - stt_done) * 1000:.2f}ms")

            logger.info(f"Assistant: {response_text}")
            context.chat_history.append({"role": "assistant", "content": response_text})

            await self.generate_response_audio(context, response_text)
        except Exception as e:
            logger.error(f"Error in pipeline: {e}")

    # --- AUDIO HANDLING & VAD ---

    async def continuous_audio_processing(self, context: ServerContext):
        """Continuously polls the sink for new audio and handles VAD"""
        logger.info(f"Started audio processing for guild {context.guild_id}")

        while not context.stop_event.is_set():
            await asyncio.sleep(0.1)

            # Get audio from all users in the sink
            # For simplicity, we'll just look at the first active user's audio
            # A more robust version would mix all users.
            for user_id, audio in list(context.audio_sink.audio_data.items()):
                if audio.file.tell() == 0:
                    continue

                audio.file.seek(0)
                data = audio.file.read()
                audio.file.seek(0)
                audio.file.truncate()

                rms = calculate_rms(data)
                logger.debug(f"RMS: {rms:.4f}")  # Uncomment this to tune VAD_THRESHOLD

                if rms > VAD_RMS_THRESHOLD:
                    if not context.is_speaking:
                        logger.info("User started speaking...")
                        self.tts_manager.stop()
                        # Clear old playback frames
                        while not context.playback_queue.empty():
                            try:
                                context.playback_queue.get_nowait()
                            except queue.Empty:
                                break

                        context.is_speaking = True
                        context.user_audio_buffer = io.BytesIO()

                    context.user_audio_buffer.write(data)
                    context.last_voice_time = time.time()

            if context.is_speaking:
                # Check if silence duration has passed
                if time.time() - context.last_voice_time > SILENCE_DURATION:
                    logger.info("User finished speaking. Triggering pipeline...")
                    context.is_speaking = False

                    # Prepare WAV for STT
                    audio_data = context.user_audio_buffer.getvalue()
                    wav_data = pcm_to_wav(
                        audio_data, CHANNELS, SAMPLE_WIDTH, SAMPLING_RATE
                    )

                    # Run pipeline in background
                    asyncio.create_task(self.run_pipeline(context, wav_data))
                    context.user_audio_buffer = None

    # --- SMOOTH PLAYBACK (JITTER BUFFER) ---

    def enqueue_audio_for_playback(self, context: ServerContext, audio_data: bytes):
        """Splits audio into 20ms frames and adds them to the playback queue."""
        # Convert to Discord format: 48kHz, 16-bit, stereo
        # (Assuming the TTS output might need conversion)
        try:
            seg = AudioSegment.from_file(io.BytesIO(audio_data))
            seg = seg.set_frame_rate(48000).set_channels(2).set_sample_width(2)
            raw_data = seg.raw_data

            frame_size = (
                int(48000 * (DISCORD_FRAME_SIZE_MS / 1000)) * CHANNELS * SAMPLE_WIDTH
            )

            for i in range(0, len(raw_data), frame_size):
                frame = raw_data[i : i + frame_size]
                if len(frame) < frame_size:
                    frame += b"\x00" * (frame_size - len(frame))
                context.playback_queue.put(frame)
        except Exception as e:
            logger.error(f"Error enqueuing audio: {e}")

    def playback_worker(self, context: ServerContext):
        """Dedicated thread to pull frames from the queue and send to Discord every 20ms."""
        logger.info(f"Playback worker started for guild {context.guild_id}")

        # Target interval in seconds
        TARGET_INTERVAL = DISCORD_FRAME_SIZE_MS / 1000.0
        next_frame_time = time.perf_counter()

        while not context.stop_event.is_set():
            # 1. Send frame if available
            try:
                frame = context.playback_queue.get_nowait()
                context.vc.send_audio_packet(frame, encode=True)
            except queue.Empty:
                pass

            # 2. Precise timing for the next frame
            next_frame_time += TARGET_INTERVAL

            # 3. Hybrid Sleep (Sleep then Busy-Wait to fix stuttering)
            while True:
                now = time.perf_counter()
                remaining = next_frame_time - now
                if remaining <= 0:
                    break
                if remaining > 0.005:  # If more than 5ms left, give up CPU
                    time.sleep(remaining - 0.003)  # Sleep slightly less than required
                # Otherwise busy-wait for sub-millisecond precision


# --- BOT COMMANDS ---

bot = DiscordVoiceBot()


@bot.command()
async def join(ctx: discord.ApplicationContext):
    if not ctx.author.voice:
        return await ctx.respond("You're not in a voice channel!")

    vc = await ctx.author.voice.channel.connect()
    guild_id = ctx.guild.id

    sink = discord.sinks.WaveSink()

    # Wait for the voice client to be fully connected before recording
    count = 0
    while not vc.is_connected() and count < 100:
        await asyncio.sleep(0.1)
        count += 1

    if vc.is_connected():
        # Library expects a coroutine for the callback
        async def finished_callback(sink, *args):
            pass

        vc.start_recording(sink, finished_callback)
    else:
        return await ctx.respond("Failed to connect to voice channel within timeout.")

    stop_event = threading.Event()
    playback_queue = queue.Queue()

    context = ServerContext(
        guild_id=guild_id,
        vc=vc,
        audio_sink=sink,
        processing_task=None,
        playback_queue=playback_queue,
        playback_thread=None,
        stop_event=stop_event,
    )

    # Start background threads/tasks
    context.processing_task = asyncio.create_task(
        bot.continuous_audio_processing(context)
    )
    context.playback_thread = threading.Thread(
        target=bot.playback_worker, args=(context,), daemon=True
    )
    context.playback_thread.start()

    bot.contexts[guild_id] = context
    await ctx.respond(f"Joined {ctx.author.voice.channel.name}!")


@bot.command()
async def leave(ctx: discord.ApplicationContext):
    guild_id = ctx.guild.id
    if guild_id in bot.contexts:
        context = bot.contexts[guild_id]
        context.stop_event.set()
        await context.vc.disconnect()
        context.processing_task.cancel()
        del bot.contexts[guild_id]
        await ctx.respond("Left the voice channel.")
    else:
        await ctx.respond("I'm not in a voice channel here.")


def run():
    token = os.getenv("BOT_TOKEN")
    if not token:
        logger.error("BOT_TOKEN not found in .env file")
    else:
        bot.run(token)
