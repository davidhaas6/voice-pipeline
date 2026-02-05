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
from .tts import TTSManager

load_dotenv()

api_key = os.environ["MISTRAL_API_KEY"]
model = "voxtral-mini-latest"

client = Mistral(api_key=api_key)


# Configuration
VAD_THRESHOLD = (
    10  # RMS threshold for voice detection (tuned lower for better sensitivity)
)
SILENCE_DURATION = 0.8  # Seconds of silence before considering a "turn" finished
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


class ModularDiscordBot(discord.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contexts = {}
        self.tts_manager = TTSManager()

    # --- PLACEHOLDERS: Implement your custom models here ---

    async def transcribe_audio(self, audio_data: bytes) -> str:
        """Transcribe audio using Mistral API"""
        print("[STT] Transcribing audio...")
        try:
            transcription_response = await client.audio.transcriptions.complete_async(
                model=model,
                file={
                    "file_name": "audio.wav",
                    "content": audio_data,
                },
            )
            return transcription_response.text
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""

    async def decide_to_respond(self, transcript: str) -> bool:
        """Placeholder for deciding if the bot should speak"""
        print(f"[Decide] Should I respond to: '{transcript}'?")
        return True

    async def generate_response_text(self, transcript: str) -> str:
        """Placeholder for Text-to-Text (T2T)"""
        print("[T2T] Generating response text...")
        return "Hello! I am a modular Discord bot. How can I help you today?"

    async def generate_response_audio(self, context: ServerContext, text: str):
        """Generates TTS audio and streams it to the playback queue."""
        print(f"[TTS] Speaking: {text}")

        def on_audio_chunk(chunk_np: np.ndarray):
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
            for i in range(0, len(raw_bytes), frame_size_bytes):
                frame = raw_bytes[i : i + frame_size_bytes]
                if len(frame) < frame_size_bytes:
                    frame += b"\x00" * (frame_size_bytes - len(frame))
                context.playback_queue.put(frame)

        self.tts_manager.speak(text, callback=on_audio_chunk)

    # --- PIPELINE ORCHESTRATION ---

    async def run_pipeline(self, context: ServerContext, audio_data: bytes):
        """Coordinates the STT -> T2T -> TTS flow"""
        try:
            transcript = await self.transcribe_audio(audio_data)
            if not transcript or not await self.decide_to_respond(transcript):
                return

            response_text = await self.generate_response_text(transcript)
            print(f"Assistant: {response_text}")

            await self.generate_response_audio(context, transcript)
        except Exception as e:
            print(f"Error in pipeline: {e}")

    # --- AUDIO HANDLING & VAD ---

    async def continuous_audio_processing(self, context: ServerContext):
        """Continuously polls the sink for new audio and handles VAD"""
        print(f"Started audio processing for guild {context.guild_id}")

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
                print(f"DEBUG RMS: {rms:.4f}")  # Uncomment this to tune VAD_THRESHOLD

                if rms > VAD_THRESHOLD:
                    if not context.is_speaking:
                        print("User started speaking...")
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
                    print("still speaking")

            if context.is_speaking:
                # Check if silence duration has passed
                if time.time() - context.last_voice_time > SILENCE_DURATION:
                    print("User finished speaking. Triggering pipeline...")
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
            print(f"Error enqueuing audio: {e}")

    def playback_worker(self, context: ServerContext):
        """Dedicated thread to pull frames from the queue and send to Discord every 20ms."""
        print(f"Playback worker started for guild {context.guild_id}")

        while not context.stop_event.is_set():
            start_time = time.perf_counter()

            try:
                # Non-blocking get from queue
                frame = context.playback_queue.get_nowait()
                context.vc.send_audio_packet(frame, encode=True)
            except queue.Empty:
                pass

            # Precise 20ms timing
            elapsed = time.perf_counter() - start_time
            sleep_time = max(0, (DISCORD_FRAME_SIZE_MS / 1000) - elapsed)
            time.sleep(sleep_time)


# --- BOT COMMANDS ---

bot = ModularDiscordBot()


@bot.command()
async def join(ctx: discord.ApplicationContext):
    if not ctx.author.voice:
        return await ctx.respond("You're not in a voice channel!")

    vc = await ctx.author.voice.channel.connect()
    guild_id = ctx.guild.id

    sink = discord.sinks.WaveSink()

    # Wait for the voice client to be fully connected before recording
    # (Fixes RecordingException: Not connected to voice channel)
    count = 0
    while not vc.is_connected() and count < 100:
        await asyncio.sleep(0.1)
        count += 1

    if vc.is_connected():
        vc.start_recording(sink, lambda *args: None)
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
        print("Error: BOT_TOKEN not found in .env file")
    else:
        bot.run(token)
