"""Central settings for Samvaad. Values can be overridden from .env."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# Storage
DATA_DIR = ROOT / "data"
WORK_DIR = DATA_DIR / "work"
MEETINGS_DIR = DATA_DIR / "meetings"
CHROMA_DIR = DATA_DIR / "chroma"

# Languages
ENGLISH = "en"
HINDI = "hi"
LANGUAGES = {ENGLISH: "English", HINDI: "Hindi"}

# Audio. Whisper handles long files; Sarvam's REST API stops at 30 seconds.
SAMPLE_RATE = 16_000
WHISPER_CHUNK_SECONDS = 600
SARVAM_CHUNK_SECONDS = 25

# Speech to text
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
SARVAM_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_MODEL = os.getenv("SARVAM_MODEL", "saaras:v3")
SARVAM_TIMEOUT_SECONDS = 120

# LLM
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
LLM_TEMPERATURE = 0.2
LLM_ATTEMPTS = 4
LLM_BACKOFF_SECONDS = 5.0
LLM_TIMEOUT_SECONDS = 120
LLM_CALL_GAP_SECONDS = float(os.getenv("LLM_CALL_GAP_SECONDS", "4.0"))

# Text splitting
LLM_CHUNK_SIZE = 3000
LLM_CHUNK_OVERLAP = 200
RAG_CHUNK_SIZE = 1000
RAG_CHUNK_OVERLAP = 150

# Retrieval
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION = "samvaad_meetings"
RETRIEVER_K = 4


def require_key(name: str) -> str:
    """Return an API key from the environment or explain how to set it."""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is missing. Add it to the .env file in the project folder.")
    return value
