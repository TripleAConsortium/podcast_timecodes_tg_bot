#!/usr/bin/env python3
import asyncio
import os
import re
import json
import shlex
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
USE_LOCAL_CLAUDE = os.getenv("USE_LOCAL_CLAUDE", "0") == "1"
LOCAL_CLAUDE_CMD = os.getenv("LOCAL_CLAUDE_CMD", "claude -p --model sonnet")
ASK_COMMAND = os.getenv("ASK_COMMAND", "/ask")
# Long audio: split the transcript into time windows and summarize each, then
# merge — a single LLM call over a 1-2h transcript tends to cover only the start.
CHUNK_MINUTES = int(os.getenv("CHUNK_MINUTES", "30"))
# Only chunk when the audio is longer than this (minutes); shorter → single call.
CHUNK_THRESHOLD_MIN = int(os.getenv("CHUNK_THRESHOLD_MIN", "45"))
# Telegram hard limits.
TG_MSG_LIMIT = 4096
TG_CAPTION_LIMIT = 1024

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


def generate_timecodes(transcript: str, part_context: tuple | None = None, backend: str | None = None) -> str:
    """Split a transcript into topic timecodes via DeepSeek API or local claude CLI.

    part_context = (index, total, start_ts, end_ts) marks this transcript as one
    time-window of a longer episode, so the model covers only this part and keeps
    the absolute timestamps it was given instead of resetting to zero.
    backend = "claude" | "deepseek" overrides the USE_LOCAL_CLAUDE default for this call.
    """
    if part_context:
        idx, total, start_ts, end_ts = part_context
        prompt = (
            f"Ниже — ЧАСТЬ {idx} из {total} транскрипта одного подкаста "
            f"(интервал {start_ts}–{end_ts}) с таймкодами.\n"
            "Раздели ЭТУ часть на смысловые темы. Таймкоды бери как есть — они "
            "абсолютные, от начала всего подкаста, не сбрасывай на ноль.\n"
            "Ответ — ТОЛЬКО список таймкодов этой части, без вступлений и итогов:\n"
            "00:00 — Название темы\n"
            "05:30 — Другая тема\n"
            "...\n\n"
            f"Транскрипт части:\n{transcript}"
        )
    else:
        prompt = (
            "Ниже — транскрипт подкаста с таймкодами.\n"
            "Раздели его на смысловые темы. Для каждой темы укажи таймкод начала и краткое название.\n"
            "Формат ответа — только список таймкодов, без лишнего текста:\n"
            "00:00 — Название темы\n"
            "05:30 — Другая тема\n"
            "...\n\n"
            f"Транскрипт:\n{transcript}"
        )
    use_claude = USE_LOCAL_CLAUDE if backend is None else backend == "claude"
    if use_claude:
        return run_local_claude(prompt)
    response = deepseek.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def run_local_claude(prompt: str) -> str:
    """Pipe the prompt into a local CLI (default: `claude -p --model sonnet`) and return stdout."""
    cmd = shlex.split(LOCAL_CLAUDE_CMD)
    log.info("Running local LLM: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=1500,
    )
    if result.returncode != 0:
        raise RuntimeError(f"local LLM failed (rc={result.returncode}): {result.stderr[:300]}")
    return result.stdout.strip()


def _partition_segments(segments: list[dict], window_seconds: float) -> list[list[dict]]:
    """Split segments into consecutive time windows of ~window_seconds each."""
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    window_start = segments[0]["start"] if segments else 0.0
    for seg in segments:
        if cur and seg["start"] - window_start >= window_seconds:
            chunks.append(cur)
            cur = []
            window_start = seg["start"]
        cur.append(seg)
    if cur:
        chunks.append(cur)
    return chunks


def _last_timecode_seconds(text: str) -> float:
    """Parse the latest mm:ss / h:mm:ss timestamp present in a timecode list."""
    last = 0.0
    for m in re.finditer(r"(?<!\d)(\d{1,2}):(\d{2})(?::(\d{2}))?(?!\d)", text):
        h_or_m, mm, ss = m.group(1), m.group(2), m.group(3)
        secs = int(h_or_m) * 3600 + int(mm) * 60 + int(ss) if ss is not None else int(h_or_m) * 60 + int(mm)
        last = max(last, secs)
    return last


def generate_timecodes_for_segments(segments: list[dict], backend: str | None = None) -> str:
    """Generate timecodes for a whole episode, chunking long audio by time windows.

    whisper transcribes the full file, but a single LLM call over a 1-2h transcript
    tends to cover only the beginning, so long episodes are split into CHUNK_MINUTES
    windows, each summarized independently, and the results merged.
    backend = "claude" | "deepseek" overrides the USE_LOCAL_CLAUDE default.
    """
    if not segments:
        return ""
    duration = segments[-1]["end"]
    active = backend or ("claude" if USE_LOCAL_CLAUDE else "deepseek")
    log.info("Transcript covers %s (%.1f min), backend=%s", format_timestamp(duration), duration / 60, active)
    if duration <= CHUNK_THRESHOLD_MIN * 60:
        result = generate_timecodes(build_transcript_with_times(segments), backend=backend)
    else:
        windows = _partition_segments(segments, CHUNK_MINUTES * 60)
        log.info("Long audio → %d windows of ~%d min", len(windows), CHUNK_MINUTES)
        parts = []
        for i, win in enumerate(windows, 1):
            start_ts = format_timestamp(win[0]["start"])
            end_ts = format_timestamp(win[-1]["end"])
            log.info("  window %d/%d %s–%s (%d segments)", i, len(windows), start_ts, end_ts, len(win))
            out = generate_timecodes(
                build_transcript_with_times(win),
                part_context=(i, len(windows), start_ts, end_ts),
                backend=backend,
            ).strip()
            if out:
                parts.append(out)
        result = "\n".join(parts)
    covered = _last_timecode_seconds(result)
    if duration - covered > 600:
        log.warning(
            "Timecodes may be incomplete: last %s vs audio end %s",
            format_timestamp(covered), format_timestamp(duration),
        )
    return result


def split_for_telegram(text: str, limit: int = TG_MSG_LIMIT) -> list[str]:
    """Split text into <=limit-char pieces at line boundaries (hard-split long lines)."""
    parts: list[str] = []
    cur = ""
    for line in text.split("\n"):
        while len(line) > limit:
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(line[:limit])
            line = line[limit:]
        if not cur:
            cur = line
        elif len(cur) + 1 + len(line) <= limit:
            cur += "\n" + line
        else:
            parts.append(cur)
            cur = line
    if cur:
        parts.append(cur)
    return parts or [""]


async def send_timecodes_as_replies(target, body: str, use_bot: bool = False):
    """Reply to `target` with `body`, split across multiple messages when too long."""
    parts = split_for_telegram(body, TG_MSG_LIMIT)
    for part in parts:
        sent = False
        if use_bot and BOT_TOKEN:
            sent = bool(await bot_api_reply(target.chat_id, target.id, part))
        if not sent:
            await target.reply(part)
    log.info("Posted timecodes as %d message(s)", len(parts))


def parse_backend(arg: str | None):
    """Map a /tc|/timecodes argument to a backend override.

    Returns (backend, error): backend is "claude"|"deepseek"|None; error is a
    user-facing message when the arg is present but unrecognized.
    """
    if not arg:
        return None, None
    a = arg.strip().lower()
    if a in ("claude", "cl"):
        return "claude", None
    if a in ("deepseek", "ds"):
        return "deepseek", None
    return None, f"Неизвестная модель «{arg}». Доступно: claude, deepseek."


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


async def process_audio(message, status_message=None, use_bot=False, backend=None):
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

        log.info(f"Transcribed {len(segments)} segments")

        if status_message:
            await status_message.edit("Генерирую таймкоды...")
        elif bot_status_id:
            await bot_api_edit(chat_id, bot_status_id, "Генерирую таймкоды...")

        log.info("Generating timecodes...")
        timecodes = generate_timecodes_for_segments(segments, backend=backend)
        log.info("Timecodes generated")

        body = f"{timecodes}\n\n#подкаст"
        parts = split_for_telegram(body, TG_MSG_LIMIT)
        first, rest = parts[0], parts[1:]

        # First part reuses the status message / expected reply slot.
        if status_message:
            await status_message.edit(first)
        elif bot_status_id:
            if not await bot_api_edit(chat_id, bot_status_id, first):
                await message.reply(first)
        elif use_bot and BOT_TOKEN:
            if not await bot_api_reply(chat_id, msg_id, first):
                await message.reply(first)
        else:
            await message.reply(first)

        # Any remaining parts go as follow-up replies to the audio.
        for part in rest:
            if use_bot and BOT_TOKEN and await bot_api_reply(chat_id, msg_id, part):
                continue
            await message.reply(part)

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

        log.info(f"Transcribed {len(segments)} segments")

        log.info("Generating timecodes...")
        timecodes = generate_timecodes_for_segments(segments)
        log.info("Timecodes generated")

        body = f"{timecodes}\n\n#подкаст"
        if len(body) <= TG_CAPTION_LIMIT:
            try:
                await client.edit_message(message, body)
                log.info("Caption added to own message")
            except Exception as edit_err:
                log.warning(f"Caption edit failed ({type(edit_err).__name__}: {edit_err}); replying with timecodes")
                await send_timecodes_as_replies(message, body)
        else:
            # Too long for a caption → post as a reply chain instead.
            await send_timecodes_as_replies(message, body)

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


@client.on(events.NewMessage(pattern=rf"(?i)^{re.escape(TRIGGER_COMMAND)}(?:\s+(\S+))?\s*$"))
async def handle_reply_trigger(event):
    """Process old audio on reply with /timecodes [claude|deepseek]."""
    message = event.message
    if not message.is_reply:
        return
    backend, err = parse_backend(event.pattern_match.group(1))
    if err:
        await message.reply(err)
        return
    target = await message.get_reply_message()
    if not target or not has_audio(target):
        await message.reply("Ответьте этой командой на голосовое или аудио сообщение.")
        return

    sender = await message.get_sender()
    sender_name = getattr(sender, "first_name", None) or "Unknown"
    log.info(f"Manual trigger from {sender_name} for message {target.id} (backend={backend or 'default'})")

    # Edit the trigger message into a status, then replace with result.
    status = await message.edit("Скачиваю аудио...")
    await process_audio(target, status_message=status, backend=backend)


CAPTION_COMMAND = os.getenv("CAPTION_COMMAND", "/tc")


@client.on(events.NewMessage(pattern=rf"(?i)^{re.escape(CAPTION_COMMAND)}(?:\s+(\S+))?\s*$"))
async def handle_caption_trigger(event):
    """Attach timecodes as caption; /tc [claude|deepseek] picks the model."""
    message = event.message
    if not message.is_reply:
        return
    backend, err = parse_backend(event.pattern_match.group(1))
    if err:
        await message.reply(err)
        return
    target = await message.get_reply_message()
    if not target or not has_audio(target):
        await message.reply("Ответьте этой командой на голосовое или аудио сообщение.")
        return

    sender = await message.get_sender()
    sender_name = getattr(sender, "first_name", None) or "Unknown"
    log.info(f"Caption trigger from {sender_name} for message {target.id} (backend={backend or 'default'})")

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

        log.info(f"Transcribed {len(segments)} segments")

        log.info("Generating timecodes...")
        timecodes = generate_timecodes_for_segments(segments, backend=backend)
        log.info("Timecodes generated")

        body = f"{timecodes}\n\n#подкаст"
        use_bot = (not target.out) and target.chat_id == CHAT_ID

        if target.out and len(body) <= TG_CAPTION_LIMIT:
            try:
                await client.edit_message(target, body)
                log.info("Caption added to original message")
            except Exception as edit_err:
                log.warning(f"Caption edit failed ({type(edit_err).__name__}: {edit_err}); replying with timecodes")
                await send_timecodes_as_replies(target, body, use_bot=use_bot)
        else:
            # Foreign message, or too long for a caption → reply (chain if needed).
            await send_timecodes_as_replies(target, body, use_bot=use_bot)

    except Exception as e:
        log.error(f"Error: {e}")
        await target.reply(f"Ошибка: {e}")
    finally:
        os.unlink(tmp_path)


def is_ask_command(event) -> bool:
    text = event.message.message or ""
    if not text.startswith(ASK_COMMAND):
        return False
    tail = text[len(ASK_COMMAND):]
    return tail == "" or tail[0].isspace()


@client.on(events.NewMessage(func=is_ask_command))
async def handle_ask_command(event):
    """Run the command text as a prompt through the local CLI, with the replied
    message attached as context. Result is sent via Bot API as a reply to the
    command message."""
    message = event.message
    if not message.is_reply:
        await message.reply("Ответьте этой командой на сообщение.")
        return

    target = await message.get_reply_message()
    if not target:
        await message.reply("Не могу прочитать сообщение, на которое отвечено.")
        return

    sender = await message.get_sender()
    sender_name = getattr(sender, "first_name", None) or "Unknown"

    raw_text = message.message or ""
    user_prompt = raw_text[len(ASK_COMMAND):].strip()
    target_text = (target.message or "").strip()

    log.info(f"Ask from {sender_name} reply→{target.id}, prompt={user_prompt[:80]!r}")

    if not user_prompt and not target_text:
        await message.reply("Пустой prompt и пустое сообщение.")
        return

    parts = [p for p in (user_prompt, target_text) if p]
    full_prompt = "\n\n".join(parts)

    chat_id = message.chat_id
    bot_status_id = None
    if BOT_TOKEN:
        bot_status_id = await bot_api_reply(chat_id, message.id, "Думаю...")

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_local_claude, full_prompt)
    except Exception as e:
        log.error(f"Ask error: {e}")
        text = f"Ошибка: {e}"
        if bot_status_id:
            if not await bot_api_edit(chat_id, bot_status_id, text):
                await message.reply(text)
        else:
            await message.reply(text)
        return

    if not result:
        result = "(пустой ответ)"
    if len(result) > 4096:
        result = result[:4090] + "\n..."

    if bot_status_id:
        if not await bot_api_edit(chat_id, bot_status_id, result):
            await message.reply(result)
    else:
        await message.reply(result)


if __name__ == "__main__":
    log.info("Starting podcast bot...")
    client.start()
    client.run_until_disconnected()
