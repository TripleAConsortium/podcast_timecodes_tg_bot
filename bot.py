#!/usr/bin/env python3
import asyncio
import os
import json
import tempfile
import subprocess
import logging
from pathlib import Path

import httpx
from dotenv import load_dotenv
from telethon import TelegramClient, events
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
TRIGGER_COMMAND = os.getenv("TRIGGER_COMMAND", "/timecodes")
CHAT_ID = int(os.getenv("CHAT_ID", "0"))
BOT_TOKEN = os.getenv("BOT_TOKEN", "")

client = TelegramClient("podcast_bot", API_ID, API_HASH)

processed_ids: set[int] = set()

deepseek = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)


async def bot_api_reply(chat_id: int, reply_to_message_id: int, text: str) -> int | None:
    """Send a reply via Bot API. Returns message_id on success, None on failure."""
    if not BOT_TOKEN:
        log.warning("BOT_TOKEN not set, cannot send via Bot API")
        return None
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "reply_to_message_id": reply_to_message_id,
        "text": text,
    }
    async with httpx.AsyncClient() as http:
        resp = await http.post(url, json=payload)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("result", {}).get("message_id")
        log.error(f"Bot API error: {resp.status_code} {resp.text}")
        return None


async def bot_api_edit(chat_id: int, message_id: int, text: str) -> bool:
    """Edit a message via Bot API."""
    if not BOT_TOKEN:
        return False
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/editMessageText"
    payload = {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": text,
    }
    async with httpx.AsyncClient() as http:
        resp = await http.post(url, json=payload)
        if resp.status_code == 200:
            return True
        log.error(f"Bot API edit error: {resp.status_code} {resp.text}")
        return False


WHISPER_CPP_MODEL = os.getenv("WHISPER_CPP_MODEL", os.path.expanduser("~/.local/share/whisper-cpp/ggml-base.bin"))


def transcribe(audio_path: str) -> list[dict]:
    """Run whisper-cpp and return segments with timestamps."""
    with tempfile.TemporaryDirectory() as tmp:
        # whisper-cpp requires WAV input.
        wav_path = os.path.join(tmp, "audio.wav")
        conv = subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path],
            capture_output=True, timeout=120,
        )
        if conv.returncode != 0:
            log.error("FFmpeg conversion failed: %s", conv.stderr.decode(errors="replace")[:300])
            return []

        json_out = os.path.join(tmp, "audio")
        cmd = [
            "whisper-cli",
            "-m", WHISPER_CPP_MODEL,
            "-l", "ru",
            "-f", wav_path,
            "-oj",
            "-of", json_out,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=3600)
        json_file = Path(json_out + ".json")
        if not json_file.exists():
            log.error("whisper-cpp produced no output. stderr: %s", result.stderr.decode(errors="replace")[:300])
            return []
        with open(json_file) as f:
            data = json.load(f)
    # whisper-cpp JSON uses "transcription" key with "offsets" for timestamps.
    raw = data.get("transcription", [])
    segments = []
    for item in raw:
        offsets = item.get("offsets", {})
        segments.append({
            "start": offsets.get("from", 0) / 1000.0,
            "end": offsets.get("to", 0) / 1000.0,
            "text": item.get("text", ""),
        })
    return segments


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def build_transcript_with_times(segments: list[dict]) -> str:
    """Build transcript text with timestamps for LLM."""
    lines = []
    for seg in segments:
        ts = format_timestamp(seg["start"])
        lines.append(f"[{ts}] {seg['text'].strip()}")
    return "\n".join(lines)


def generate_timecodes(transcript: str) -> str:
    """Ask DeepSeek to split transcript into topics with timecodes."""
    prompt = (
        "Ниже — транскрипт подкаста с таймкодами.\n"
        "Раздели его на смысловые темы. Для каждой темы укажи таймкод начала и краткое название.\n"
        "Формат ответа — только список таймкодов, без лишнего текста:\n"
        "00:00 — Название темы\n"
        "05:30 — Другая тема\n"
        "...\n\n"
        f"Транскрипт:\n{transcript}"
    )
    response = deepseek.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def has_audio(message) -> bool:
    """Check if message has voice, audio, or document with audio mime."""
    if message.voice or message.audio:
        return True
    if message.document:
        mime = (message.file.mime_type if message.file else "") or ""
        name = ((message.file.name if message.file else "") or "").lower()
        if mime.startswith("audio/") or name.endswith((".m4a", ".mp3", ".ogg", ".opus", ".wav", ".flac", ".aac")):
            return True
    return False


async def process_audio(message, status_message=None, use_bot=False):
    """Download, transcribe, and generate timecodes for an audio message."""
    suffix = ".ogg"
    if message.audio:
        suffix = ".mp3"
    elif message.document:
        name = ((message.file.name if message.file else "") or "").lower()
        if name.endswith(".m4a"):
            suffix = ".m4a"
        elif name.endswith(".mp3"):
            suffix = ".mp3"

    chat_id = message.chat_id
    msg_id = message.id

    # For bot mode, send initial status via Bot API.
    bot_status_id = None
    if use_bot and BOT_TOKEN and not status_message:
        bot_status_id = await bot_api_reply(chat_id, msg_id, "Скачиваю аудио...")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
    try:
        await message.download_media(file=tmp_path)
        log.info(f"Downloaded to {tmp_path}")

        if status_message:
            await status_message.edit("Транскрибирую...")
        elif bot_status_id:
            await bot_api_edit(chat_id, bot_status_id, "Транскрибирую...")

        log.info("Transcribing...")
        loop = asyncio.get_event_loop()
        segments = await loop.run_in_executor(None, transcribe, tmp_path)
        if not segments:
            text = "Не удалось распознать речь."
            if status_message:
                await status_message.edit(text)
            elif bot_status_id:
                if not await bot_api_edit(chat_id, bot_status_id, text):
                    await message.reply(text)
            elif use_bot and BOT_TOKEN:
                if not await bot_api_reply(chat_id, msg_id, text):
                    await message.reply(text)
            else:
                await message.reply(text)
            return

        transcript = build_transcript_with_times(segments)
        log.info(f"Transcribed {len(segments)} segments")

        if status_message:
            await status_message.edit("Генерирую таймкоды...")
        elif bot_status_id:
            await bot_api_edit(chat_id, bot_status_id, "Генерирую таймкоды...")

        log.info("Generating timecodes...")
        timecodes = generate_timecodes(transcript)
        log.info("Timecodes generated")

        reply_text = f"{timecodes}\n\n#подкаст"
        if len(reply_text) > 4096:
            reply_text = reply_text[:4090] + "\n..."

        if status_message:
            await status_message.edit(reply_text)
        elif bot_status_id:
            if not await bot_api_edit(chat_id, bot_status_id, reply_text):
                await message.reply(reply_text)
        elif use_bot and BOT_TOKEN:
            if not await bot_api_reply(chat_id, msg_id, reply_text):
                await message.reply(reply_text)
        else:
            await message.reply(reply_text)

    except Exception as e:
        log.error(f"Error: {e}")
        text = f"Ошибка: {e}"
        if status_message:
            await status_message.edit(text)
        elif bot_status_id:
            if not await bot_api_edit(chat_id, bot_status_id, text):
                await message.reply(text)
        elif use_bot and BOT_TOKEN:
            if not await bot_api_reply(chat_id, msg_id, text):
                await message.reply(text)
        else:
            await message.reply(text)
    finally:
        os.unlink(tmp_path)


async def process_audio_as_caption(message):
    """Download, transcribe, generate timecodes and set as caption on the original message."""
    suffix = ".ogg"
    if message.audio:
        suffix = ".mp3"
    elif message.document:
        name = ((message.file.name if message.file else "") or "").lower()
        if name.endswith(".m4a"):
            suffix = ".m4a"
        elif name.endswith(".mp3"):
            suffix = ".mp3"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
    try:
        await message.download_media(file=tmp_path)
        log.info(f"Downloaded to {tmp_path}")

        log.info("Transcribing...")
        loop = asyncio.get_event_loop()
        segments = await loop.run_in_executor(None, transcribe, tmp_path)
        if not segments:
            await message.reply("Не удалось распознать речь.")
            return

        transcript = build_transcript_with_times(segments)
        log.info(f"Transcribed {len(segments)} segments")

        log.info("Generating timecodes...")
        timecodes = generate_timecodes(transcript)
        log.info("Timecodes generated")

        caption = f"{timecodes}\n\n#подкаст"
        if len(caption) > 1024:
            caption = caption[:1020] + "\n..."

        try:
            await client.edit_message(message, caption)
            log.info("Caption added to own message")
        except Exception as edit_err:
            log.warning(f"Caption edit failed ({type(edit_err).__name__}: {edit_err}); replying with timecodes")
            await message.reply(caption)

    except Exception as e:
        log.error(f"Error: {e}")
        await message.reply(f"Ошибка: {e}")
    finally:
        os.unlink(tmp_path)


@client.on(events.NewMessage(
    chats=CHAT_ID if CHAT_ID else None,
    func=lambda e: has_audio(e.message),
))
async def handle_new_audio(event):
    """Auto-process new audio messages."""
    message = event.message
    if message.id in processed_ids:
        log.info(f"Skipping already processed message {message.id}")
        return
    processed_ids.add(message.id)

    sender = await message.get_sender()
    sender_name = getattr(sender, "first_name", None) or "Unknown"
    log.info(f"Received audio from {sender_name} in {message.chat_id}")

    if message.out:
        await process_audio_as_caption(message)
    else:
        await process_audio(message, use_bot=bool(BOT_TOKEN))


@client.on(events.NewMessage(pattern=f"^{TRIGGER_COMMAND}$"))
async def handle_reply_trigger(event):
    """Process old audio when someone replies with /timecodes."""
    message = event.message
    if not message.is_reply:
        return
    target = await message.get_reply_message()
    if not target or not has_audio(target):
        await message.reply("Ответьте этой командой на голосовое или аудио сообщение.")
        return

    sender = await message.get_sender()
    sender_name = getattr(sender, "first_name", None) or "Unknown"
    log.info(f"Manual trigger from {sender_name} for message {target.id}")

    # Edit the trigger message into a status, then replace with result.
    status = await message.edit("Скачиваю аудио...")
    await process_audio(target, status_message=status)


CAPTION_COMMAND = os.getenv("CAPTION_COMMAND", "/tc")


@client.on(events.NewMessage(pattern=f"^{CAPTION_COMMAND}$"))
async def handle_caption_trigger(event):
    """Process audio and attach timecodes as caption to the original message."""
    message = event.message
    if not message.is_reply:
        return
    target = await message.get_reply_message()
    if not target or not has_audio(target):
        await message.reply("Ответьте этой командой на голосовое или аудио сообщение.")
        return

    sender = await message.get_sender()
    sender_name = getattr(sender, "first_name", None) or "Unknown"
    log.info(f"Caption trigger from {sender_name} for message {target.id}")

    suffix = ".ogg"
    if target.audio:
        suffix = ".mp3"
    elif target.document:
        name = ((target.file.name if target.file else "") or "").lower()
        if name.endswith(".m4a"):
            suffix = ".m4a"
        elif name.endswith(".mp3"):
            suffix = ".mp3"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
    try:
        await target.download_media(file=tmp_path)
        log.info(f"Downloaded to {tmp_path}")

        # Delete command message.
        await message.delete()

        log.info("Transcribing...")
        loop = asyncio.get_event_loop()
        segments = await loop.run_in_executor(None, transcribe, tmp_path)
        if not segments:
            await target.reply("Не удалось распознать речь.")
            return

        transcript = build_transcript_with_times(segments)
        log.info(f"Transcribed {len(segments)} segments")

        log.info("Generating timecodes...")
        timecodes = generate_timecodes(transcript)
        log.info("Timecodes generated")

        caption = f"{timecodes}\n\n#подкаст"
        if len(caption) > 1024:
            caption = caption[:1020] + "\n..."

        try:
            await client.edit_message(target, caption)
            log.info("Caption added to original message")
        except Exception as edit_err:
            log.warning(f"Caption edit failed ({type(edit_err).__name__}: {edit_err}); replying with timecodes")
            await target.reply(caption)

    except Exception as e:
        log.error(f"Error: {e}")
        await target.reply(f"Ошибка: {e}")
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    log.info("Starting podcast bot...")
    client.start()
    client.run_until_disconnected()
