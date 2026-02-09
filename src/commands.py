import os

import discord

from .bot import (
    VAD_RMS_THRESHOLD,
    DiscordVoiceBot,
)
from .logger import get_logger

logger = get_logger(__name__)

bot = DiscordVoiceBot()


@bot.command()
async def join(ctx: discord.ApplicationContext):
    """Joins a voice channel and starts the pipeline (STT -> T2T -> TTS)."""
    await ctx.defer(ephemeral=True)
    await bot.start_in_voice(ctx)


@bot.command()
async def leave(ctx: discord.ApplicationContext):
    """Leaves the voice channel and stops the pipeline."""
    guild_id = ctx.guild.id
    if guild_id not in bot.contexts:
        return await ctx.respond("I'm not in a voice channel here.", ephemeral=True)

    await bot.stop_in_voice(guild_id)
    await ctx.respond("Left the voice channel.", ephemeral=True)


@bot.command()
async def system(ctx: discord.ApplicationContext, new_prompt: str = None):
    """Updates or views the bot's system prompt."""
    if new_prompt is None:
        await ctx.respond(
            f"Current system prompt: ```{bot.system_prompt}```", ephemeral=True
        )
    else:
        bot.system_prompt = new_prompt
        await ctx.respond(
            f"System prompt updated to: ```{new_prompt}```", ephemeral=True
        )


@bot.command()
async def clear(ctx: discord.ApplicationContext):
    """Clears the bot's chat history."""
    guild_id = ctx.guild.id
    context = bot.contexts.get(guild_id)
    if not context:
        return await ctx.respond("I'm not in a voice channel here.", ephemeral=True)
    context.chat_history.clear()
    await ctx.respond("Chat history cleared.", ephemeral=True)


@bot.event
async def on_voice_state_update(member, before, after):
    """Handle unexpected disconnections"""
    if member.id == bot.user.id and before.channel and not after.channel:
        guild_id = before.channel.guild.id
        if guild_id in bot.contexts:
            logger.warning(f"Bot was disconnected from {guild_id}, cleaning up...")
            await bot.stop_in_voice(guild_id)


@bot.command()
async def status(ctx: discord.ApplicationContext):
    """Show bot status and diagnostics."""
    guild_id = ctx.guild.id
    context = bot.contexts.get(guild_id)

    if not context:
        return await ctx.respond("Not active in this server.", ephemeral=True)

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

    await ctx.respond(status_msg, ephemeral=True)


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
            # Note: bot.close() handles context cleanup now
            logger.info("Bot run finished.")
