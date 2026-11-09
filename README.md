# Samvaad

Bilingual AI meeting assistant. Give it a YouTube link or a recording in English or Hindi, and it produces an English transcript, a summary, action items, decisions, open questions, and a chat that answers only from that meeting.

## How it works

```text
YouTube link / MP3, MP4, M4A, WAV
        |  yt-dlp + FFmpeg (16 kHz mono)
        v
English -> local Whisper      Hindi -> Sarvam STT (25 s chunks, translated to English)
        |
        v
English transcript
   |-- Gemini map-reduce summary
   |-- Chunk-wise action items, decisions, open questions -> merge -> consolidate
   `-- MiniLM embeddings -> ChromaDB (scoped by meeting_id) -> RAG chat
```

## What this version fixes

The project started from the [Sheryians AI School tutorial](https://www.youtube.com/watch?v=xlYJhtL0qbQ). Two weaknesses were fixed:

1. **Meetings no longer mix.** Each chunk carries a stable `meeting_id`, derived from the YouTube video ID or the file's SHA-256 hash plus language. Every retrieval, count and re-index is filtered by it. Reprocessing a meeting replaces only that meeting's chunks.
2. **Long meetings no longer break extraction.** Action items, decisions and open questions are extracted per 3,000-character chunk, merged, de-duplicated and consolidated. If consolidation fails, the merged notes are kept. Chunks that fail are reported in the app.

Also included: per-chunk transcript caching so a failed run can resume, retry with backoff for rate limits, saved meetings, and Markdown export.

## Setup on Windows

Use Python 3.10 to 3.13 and install [FFmpeg](https://ffmpeg.org/download.html):

```powershell
winget install Gyan.FFmpeg
```

Open a new terminal, then run:

```powershell
cd D:\Samvaad
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
copy .env.example .env
```

If you use Anaconda and `venv` fails, run these in Command Prompt instead:

```bat
D:\anaconda3\Scripts\conda.exe create -p D:\Samvaad\.venv python=3.12 pip -y -c conda-forge --override-channels
D:\anaconda3\Scripts\activate.bat D:\Samvaad\.venv
python -m pip install -r requirements.txt
copy .env.example .env
```

Add your keys to `.env`. `GEMINI_API_KEY` is always required. `SARVAM_API_KEY` is needed only for Hindi audio. The first run downloads the Whisper and embedding models.

## Run

```powershell
streamlit run app.py
```

Or from the command line:

```powershell
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --language en --chat
python main.py "D:\recordings\standup.mp3" --language hi
```

## Logs

While a meeting is processing, you can follow every step in three places:

- **App:** the live panel above the results, and the **Activity log** section in the sidebar
- **Terminal:** the same steps with full error tracebacks, plus Whisper's progress bar
- **File:** `data/logs/samvaad.log`

Restart Streamlit after changing `.env` or the code.

## Tests

```powershell
python -m pytest
```

The tests use real FFmpeg and ChromaDB, and fake only paid or downloaded services (speech models, Gemini, embeddings).
