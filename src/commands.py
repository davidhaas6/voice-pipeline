import asyncio
import os
import queue
import threading
from collections import deque

import discord

from .bot import VAD_RMS_THRESHOLD, DiscordVoiceBot, ServerContext
from .logger import get_logger
from .tts import TTSManager

logger = get_logger(__name__)

bot = DiscordVoiceBot()


@bot.command()
async def join(ctx: discord.ApplicationContext):
    """Joins a voice channel and starts the pipeline (STT -> T2T -> TTS)."""
    # TODO: refactor most of this into a function in DiscordVoiceBot
    await ctx.defer()  # Give us time to connect and setup

    if not ctx.author.voice:
        return await ctx.followup.send("You're not in a voice channel!")

    try:
        vc = await ctx.author.voice.channel.connect(timeout=20, reconnect=True)
    except Exception as e:
        logger.error(f"Failed to connect to voice: {e}")
        return await ctx.followup.send("Failed to connect to your voice channel.")

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
        chat_history=deque(maxlen=50),
        tts=TTSManager(),
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
    await ctx.followup.send(f"Joined {ctx.author.voice.channel.name}!")


@bot.command()
async def leave(ctx: discord.ApplicationContext):
    # TODO: refactor most of this into a function in DiscordVoiceBot
    guild_id = ctx.guild.id
    context = bot.contexts.get(guild_id)
    if not context:
        return await ctx.respond("I'm not in a voice channel here.")

    context.stop_event.set()

    try:
        context.tts.stop()
    except Exception as e:
        logger.error(f"Error stopping TTS: {e}")

    while not context.playback_queue.empty():
        try:
            context.playback_queue.get_nowait()
        except queue.Empty:
            break

    # stops playback thread sending before disconnect closes the socket
    if context.playback_thread and context.playback_thread.is_alive():
        context.playback_thread.join(timeout=1.0)

    # stop recording
    try:
        if context.vc and context.vc.is_connected():
            context.vc.stop_recording()
    except Exception as e:
        logger.error(f"Error stopping recording: {e}")

    try:
        await context.vc.disconnect()
    except Exception as e:
        logger.error(f"Error disconnecting from voice channel: {e}")

    if context.processing_task:
        context.processing_task.cancel()

    del bot.contexts[guild_id]
    await ctx.respond("Left the voice channel.")


@bot.command()
async def system(ctx: discord.ApplicationContext, new_prompt: str = None):
    """Updates or views the bot's system prompt."""
    if new_prompt is None:
        await ctx.respond(f"Current system prompt: ```{bot.system_prompt}```")
    else:
        bot.system_prompt = new_prompt
        await ctx.respond(f"System prompt updated to: ```{new_prompt}```")


@bot.command()
async def clear(ctx: discord.ApplicationContext):
    """Clears the bot's chat history."""
    guild_id = ctx.guild.id
    context = bot.contexts.get(guild_id)
    if not context:
        return await ctx.respond("I'm not in a voice channel here.")
    context.chat_history.clear()
    await ctx.respond("Chat history cleared.")


@bot.event
async def on_voice_state_update(member, before, after):
    """Handle unexpected disconnections"""
    # TODO: plug in cleanup function
    if member.id == bot.user.id and before.channel and not after.channel:
        guild_id = before.channel.guild.id
        if guild_id in bot.contexts:
            logger.warning(
                f"Bot was disconnected from {guild_id}, cleaning up (MOCK) ..."
            )
            # await cleanup_context(bot.contexts[guild_id])
            del bot.contexts[guild_id]


@bot.command()
async def status(ctx: discord.ApplicationContext):
    """Show bot status and diagnostics."""
    guild_id = ctx.guild.id
    context = bot.contexts.get(guild_id)

    if not context:
        return await ctx.respond("Not active in this server.")

    playback_alive = context.playback_thread and context.playback_thread.is_alive()
    queue_size = context.playback_queue.qsize()
    vc_connected = context.vc and context.vc.is_connected()
    speaking = "🎤 User speaking" if context.user_speaking else "🔇 Silent"

    status_msg = f"""**Bot Status**
Voice: {"✅ Connected" if vc_connected else "❌ Disconnected"}
Playback: {"✅ Running" if playback_alive else "❌ Dead"}
Queue: {queue_size} frames
VAD Threshold: {VAD_RMS_THRESHOLD}
{speaking}
Chat History: {len(context.chat_history)}/50 messages
"""

    await ctx.respond(status_msg)


def run():
    token = os.getenv("BOT_TOKEN")
    if not token:
        logger.error("BOT_TOKEN not found in .env file")
    else:
        try:
            bot.run(token)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt: shutting down...")
        finally:
            try:
                for context in bot.contexts.values():
                    try:
                        context.tts.stop()
                    except Exception as e:
                        logger.error(f"Error stopping TTS: {e}")
                    context.stop_event.set()
                    if context.playback_thread and context.playback_thread.is_alive():
                        context.playback_thread.join(timeout=1.0)
                    if context.processing_task:
                        context.processing_task.cancel()
            except Exception as e:
                logger.error(f"Error cleaning up context: {e}")
