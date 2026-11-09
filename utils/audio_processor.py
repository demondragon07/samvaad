"""Turn a YouTube link or media file into 16 kHz mono WAV chunks."""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import config
from utils.logger import get_logger

logger = get_logger("audio")

ALLOWED_EXTENSIONS = {".mp3", ".mp4", ".wav", ".m4a"}
YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"}
_VIDEO_ID = re.compile(r"^[\w-]{6,20}$")


class AudioError(RuntimeError):
    """The input could not be downloaded, decoded or split."""


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise AudioError("FFmpeg was not found. Install it and make sure it is on your PATH.")


def _ffmpeg(*args: str) -> None:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise AudioError(f"FFmpeg failed: {result.stderr.strip()[-500:]}")


def is_youtube_url(source: str) -> bool:
    parsed = urlparse(source.strip())
    return parsed.scheme in {"http", "https"} and (parsed.hostname or "").lower() in YOUTUBE_HOSTS


def youtube_video_id(url: str) -> str:
    """Extract the video id so the same video always maps to one meeting."""
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    parts = [part for part in parsed.path.split("/") if part]
    video_id = None
    if host == "youtu.be" and parts:
        video_id = parts[0]
    elif parts[:1] == ["watch"]:
        video_id = parse_qs(parsed.query).get("v", [None])[0]
    elif len(parts) >= 2 and parts[0] in {"shorts", "live", "embed"}:
        video_id = parts[1]
    if not video_id or not _VIDEO_ID.match(video_id):
        raise AudioError("That does not look like a YouTube video link.")
    return video_id


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_youtube(url: str, work_dir: Path) -> tuple[Path, str]:
    import yt_dlp

    options = {
        "format": "bestaudio/best",
        "outtmpl": str(work_dir / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    logger.info("Downloading audio from YouTube")
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as exc:
        raise AudioError(f"Could not download the video: {exc}") from exc
    path = Path(info["requested_downloads"][0]["filepath"])
    logger.info("Downloaded '%s' (%.1f MB)", info.get("title"), path.stat().st_size / 1_048_576)
    return path, info.get("title") or "YouTube video"


def normalize_audio(source: Path, target: Path) -> Path:
    """Convert to the mono 16 kHz PCM format both speech engines expect."""
    logger.info("Converting audio to 16 kHz mono WAV")
    _ffmpeg("-i", str(source), "-vn", "-ac", "1", "-ar", str(config.SAMPLE_RATE), "-c:a", "pcm_s16le", str(target))
    return target


def split_audio(wav: Path, chunk_dir: Path, chunk_seconds: int) -> list[Path]:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    _ffmpeg(
        "-i", str(wav), "-f", "segment", "-segment_time", str(chunk_seconds),
        "-reset_timestamps", "1", "-c", "copy", str(chunk_dir / "chunk_%05d.wav"),
    )
    chunks = sorted(chunk_dir.glob("chunk_*.wav"))
    if not chunks:
        raise AudioError("No audio was found in this input.")
    logger.info("Split audio into %d chunk(s) of up to %d seconds", len(chunks), chunk_seconds)
    return chunks


def prepare_chunks(wav_source: Path, work_dir: Path, language: str) -> list[Path]:
    seconds = config.SARVAM_CHUNK_SECONDS if language == config.HINDI else config.WHISPER_CHUNK_SECONDS
    normalized = normalize_audio(wav_source, work_dir / "audio_16k_mono.wav")
    return split_audio(normalized, work_dir / "chunks", seconds)
