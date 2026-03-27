# Podcast Timecodes Telegram Bot

A Telegram userbot that automatically generates chapter timecodes for podcast audio files. It transcribes audio using [OpenAI Whisper](https://github.com/openai/whisper) and then uses the [DeepSeek](https://platform.deepseek.com) LLM to split the transcript into semantic topics with timestamps.

## How it works

1. An audio message (voice note, audio file, or audio document) is received.
2. The audio is downloaded and transcribed locally using Whisper (`--language ru`).
3. The timestamped transcript is sent to DeepSeek (`deepseek-reasoner`) with a prompt asking it to identify topic boundaries.
4. The resulting timecode list is posted back to the chat as a reply or caption.

## Prerequisites

- Python 3.9+
- [OpenAI Whisper](https://github.com/openai/whisper) CLI installed and available in `PATH`:
  ```bash
  pip install openai-whisper
  ```
- A Telegram API ID and API Hash — obtain them at <https://my.telegram.org/apps>
- A DeepSeek API key — obtain one at <https://platform.deepseek.com>

## Installation

```bash
git clone https://github.com/TripleAConsortium/podcast_timecodes_tg_bot.git
cd podcast_timecodes_tg_bot
pip install -r requirements.txt
```

## Configuration

Copy the example env file and fill in your credentials:

```bash
cp .env.example .env
```

| Variable | Required | Default | Description |
|---|---|---|---|
| `API_ID` | ✅ | — | Telegram API ID (integer) |
| `API_HASH` | ✅ | — | Telegram API Hash |
| `DEEPSEEK_API_KEY` | ✅ | — | DeepSeek API key |
| `WHISPER_MODEL` | ❌ | `base` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) |
| `TRIGGER_COMMAND` | ❌ | `/timecodes` | Command used to manually trigger timecode generation as a reply |
| `CAPTION_COMMAND` | ❌ | `/tc` | Command used to trigger timecode generation and attach result as a caption |
| `CHAT_ID` | ❌ | `0` (all chats) | Restrict the bot to a single chat by its numeric ID |

## Running

```bash
python bot.py
```

On first run Pyrogram will ask for your phone number and a confirmation code to create a session. The session is saved to `podcast_bot.session` and reused on subsequent runs.

## Usage

The bot operates as a **userbot** (runs under your own Telegram account).

| Scenario | How to trigger |
|---|---|
| **Auto** — you or someone else posts an audio message | Timecodes are generated automatically and posted as a caption or reply |
| **Manual reply** — reply to any existing audio message with `/timecodes` | Timecodes are posted as a reply to the original audio message |
| **Caption mode** — reply to any existing audio message with `/tc` | Timecodes are edited into the caption of the original audio message |

Supported audio formats: voice messages, `.ogg`, `.opus`, `.m4a`, `.mp3`, `.wav`, `.flac`, `.aac`.

> **Note:** Transcription is currently hardcoded to Russian (`--language ru`). Support for other languages is not yet configurable without modifying the source.

## Example output

```
00:00 — Вступление и знакомство с гостем
07:15 — Обсуждение карьерного пути
23:40 — Советы начинающим разработчикам
41:10 — Планы на будущее
```

## License

This project is provided as-is without a specific license. See repository for details.
