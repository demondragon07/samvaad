"""Speech to English text: local Whisper for English, Sarvam for Hindi."""
from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import requests

import config
from utils.logger import get_logger

logger = get_logger("transcriber")

ProgressFn = Optional[Callable[[int, int], None]]
_RETRY_STATUS = {429, 500, 502, 503, 504}


class TranscriptionError(RuntimeError):
    """A speech engine could not return text for an audio chunk."""


@lru_cache(maxsize=1)
def _whisper_model():
    import whisper

    logger.info("Loading Whisper model '%s' (the first run downloads it)", config.WHISPER_MODEL)
    model = whisper.load_model(config.WHISPER_MODEL)
    logger.info("Whisper model ready")
    return model


def transcribe_with_whisper(chunk: Path) -> str:
    import torch

    device = "GPU" if torch.cuda.is_available() else "CPU"
    logger.info("Whisper is running on %s. The terminal shows a progress bar.", device)
    # verbose=False makes Whisper draw a progress bar for long chunks.
    result = _whisper_model().transcribe(
        str(chunk), language="en", task="transcribe", fp16=torch.cuda.is_available(), verbose=False
    )
    return result["text"].strip()


def transcribe_with_sarvam(chunk: Path) -> str:
    """Translate one Hindi chunk (under 30 seconds) directly into English text."""
    headers = {"api-subscription-key": config.require_key("SARVAM_API_KEY")}
    data = {"model": config.SARVAM_MODEL, "mode": "translate", "language_code": "hi-IN"}
    last_error = "no response"

    for attempt in range(config.LLM_ATTEMPTS):
        if attempt:
            time.sleep(config.LLM_BACKOFF_SECONDS * 2 ** (attempt - 1))
        try:
            with open(chunk, "rb") as audio:
                response = requests.post(
                    config.SARVAM_URL,
                    headers=headers,
                    data=data,
                    files={"file": (chunk.name, audio, "audio/wav")},
                    timeout=config.SARVAM_TIMEOUT_SECONDS,
                )
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = str(exc)
            logger.warning("Sarvam network error on %s (attempt %d): %s", chunk.name, attempt + 1, exc)
            continue
        if response.status_code in _RETRY_STATUS:
            last_error = f"HTTP {response.status_code}"
            logger.warning("Sarvam returned %s on %s (attempt %d)", last_error, chunk.name, attempt + 1)
            continue
        if response.status_code != 200:
            raise TranscriptionError(f"Sarvam rejected {chunk.name}: HTTP {response.status_code} {response.text[:300]}")
        return (response.json().get("transcript") or "").strip()

    raise TranscriptionError(f"Sarvam failed on {chunk.name} after {config.LLM_ATTEMPTS} attempts: {last_error}")


def transcribe_all(chunks: list[Path], language: str, on_progress: ProgressFn = None) -> str:
    """Transcribe chunks in order. Finished chunks are cached, so a failed run can resume."""
    engine = transcribe_with_sarvam if language == config.HINDI else transcribe_with_whisper
    engine_name = "Sarvam" if language == config.HINDI else "Whisper"
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        cached = chunk.with_suffix(".txt")
        if cached.exists():
            text = cached.read_text(encoding="utf-8")
            logger.info("Chunk %d/%d: reused saved transcript", index, len(chunks))
        else:
            logger.info("Chunk %d/%d: transcribing with %s", index, len(chunks), engine_name)
            started = time.monotonic()
            text = engine(chunk)
            cached.write_text(text, encoding="utf-8")
            logger.info("Chunk %d/%d: done in %.1fs (%d characters)",
                        index, len(chunks), time.monotonic() - started, len(text))
        if text:
            parts.append(text)
        if on_progress:
            on_progress(index, len(chunks))
    return " ".join(parts).strip()
