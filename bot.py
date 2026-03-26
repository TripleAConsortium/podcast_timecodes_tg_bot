#!/usr/bin/env python3
import asyncio
import os
import json
import tempfile
import subprocess
import logging
from pathlib import Path

from dotenv import load_dotenv
from pyrogram import Client, filters
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

app = Client("podcast_bot", api_id=API_ID, api_hash=API_HASH)

chat_filter = filters.chat(CHAT_ID) if CHAT_ID else filters.all
processed_ids: set[int] = set()

deepseek = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)


def transcribe(audio_path: str) -> list[dict]:
    """Run whisper and return segments with timestamps."""
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            "whisper",
            audio_path,
            "--model", WHISPER_MODEL,
            "--language", "ru",
            "--output_format", "json",
            "--output_dir", tmp,
        ]
        subprocess.run(cmd, capture_output=True, timeout=3600)
        json_file = next(Path(tmp).glob("*.json"))
        with open(json_file) as f:
            data = json.load(f)
    return data.get("segments", [])


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
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def has_audio(message) -> bool:
    """Check if message has voice, audio, or document with audio mime."""
    if message.voice or message.audio:
        return True
    if message.document:
        mime = message.document.mime_type or ""
        name = (message.document.file_name or "").lower()
        if mime.startswith("audio/") or name.endswith((".m4a", ".mp3", ".ogg", ".opus", ".wav", ".flac", ".aac")):
            return True
    return False


async def process_audio(client, message, status_message=None):
    """Download, transcribe, and generate timecodes for an audio message."""
    suffix = ".ogg"
    if message.document:
        name = (message.document.file_name or "").lower()
        if name.endswith(".m4a"):
            suffix = ".m4a"
        elif name.endswith(".mp3"):
            suffix = ".mp3"
    elif message.audio:
        suffix = ".mp3"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
    try:
        await message.download(file_name=tmp_path)
        log.info(f"Downloaded to {tmp_path}")

        if status_message:
            await status_message.edit_text("Транскрибирую...")

        log.info("Transcribing...")
        loop = asyncio.get_event_loop()
        segments = await loop.run_in_executor(None, transcribe, tmp_path)
        if not segments:
            text = "Не удалось распознать речь."
            if status_message:
                await status_message.edit_text(text)
            else:
                await message.reply_text(text)
            return

        transcript = build_transcript_with_times(segments)
        log.info(f"Transcribed {len(segments)} segments")

        if status_message:
            await status_message.edit_text("Генерирую таймкоды...")

        log.info("Generating timecodes...")
        timecodes = generate_timecodes(transcript)
        log.info("Timecodes generated")

        reply_text = f"{timecodes}\n\n#подкаст"
        if len(reply_text) > 4096:
            reply_text = reply_text[:4090] + "\n..."

        if status_message:
            await status_message.edit_text(reply_text)
        else:
            await message.reply_text(reply_text)

    except Exception as e:
        log.error(f"Error: {e}")
        text = f"Ошибка: {e}"
        if status_message:
            await status_message.edit_text(text)
        else:
            await message.reply_text(text)
    finally:
        os.unlink(tmp_path)


@app.on_message(chat_filter & (filters.voice | filters.audio | (filters.document & filters.create(lambda _, __, m: has_audio(m)))))
async def handle_new_audio(client, message):
    """Auto-process new audio messages."""
    if message.id in processed_ids:
        log.info(f"Skipping already processed message {message.id}")
        return
    processed_ids.add(message.id)
    log.info(f"Received audio from {message.from_user.first_name} in {message.chat.id}")
    await process_audio(client, message)


@app.on_message(chat_filter & filters.reply & filters.regex(f"^{TRIGGER_COMMAND}$"))
async def handle_reply_trigger(client, message):
    """Process old audio when someone replies with /timecodes."""
    target = message.reply_to_message
    if not target or not has_audio(target):
        await message.reply_text("Ответьте этой командой на голосовое или аудио сообщение.")
        return

    log.info(f"Manual trigger from {message.from_user.first_name} for message {target.id}")

    # Edit the trigger message into a status, then replace with result.
    status = await message.edit_text("Скачиваю аудио...")
    await process_audio(client, target, status_message=status)


if __name__ == "__main__":
    log.info("Starting podcast bot...")
    app.run()
