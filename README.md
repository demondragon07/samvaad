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

## Features

- **Flexible input:** process a YouTube link or an MP3, MP4, M4A or WAV recording.
- **Bilingual transcription:** English audio runs on local Whisper. Hindi audio is split into 25-second chunks and translated to English with Sarvam STT.
- **Meeting summary:** Gemini and LangChain LCEL produce a map-reduce summary, so long recordings are handled in sections.
- **Structured notes:** action items with owner and due date, decisions, and open questions are extracted from each 3,000-character section, then merged, de-duplicated and consolidated.
- **Grounded chat:** a ChromaDB RAG chatbot answers from the selected meeting's transcript and says so when something was not discussed.
- **Meeting isolation:** every chunk carries a stable `meeting_id`, based on the YouTube video ID or the file's SHA-256 hash plus language. Search, counts and re-indexing are all filtered by it, so one meeting never leaks into another.
- **Reliable long runs:** transcripts are cached per audio chunk so a failed run can resume. Rate limits and network errors are retried with backoff, and any skipped section is reported in the app.
- **Saved work:** meetings are stored locally, can be reopened from the sidebar, and can be exported as Markdown.
- **Observability:** each step and its duration appear in the app, the terminal and a rotating log file.

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

