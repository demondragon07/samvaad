"""Connect audio, transcription, LLM analysis, indexing and saved results."""
from __future__ import annotations

import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

import config
from core.extractor import ActionItem, extract
from core.llm import get_llm
from core.summarizer import summarize
from core.transcriber import transcribe_all
from utils import audio_processor as audio
from utils.logger import get_logger

logger = get_logger("pipeline")

Progress = Optional[Callable[[str, float], None]]


class Meeting(BaseModel):
    meeting_id: str
    title: str
    source: str
    language: str
    created_at: str
    transcript: str
    summary: str
    action_items: list[ActionItem] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def meeting_path(meeting_id: str) -> Path:
    return config.MEETINGS_DIR / f"{meeting_id}.json"


def load_meeting(meeting_id: str) -> Optional[Meeting]:
    path = meeting_path(meeting_id)
    return Meeting.model_validate_json(path.read_text(encoding="utf-8")) if path.exists() else None


def list_meetings() -> list[Meeting]:
    if not config.MEETINGS_DIR.exists():
        return []
    meetings = [
        Meeting.model_validate_json(path.read_text(encoding="utf-8"))
        for path in config.MEETINGS_DIR.glob("*.json")
    ]
    return sorted(meetings, key=lambda meeting: meeting.created_at, reverse=True)


def save_meeting(meeting: Meeting) -> None:
    config.MEETINGS_DIR.mkdir(parents=True, exist_ok=True)
    meeting_path(meeting.meeting_id).write_text(meeting.model_dump_json(indent=2), encoding="utf-8")


def resolve_meeting_id(source: str, language: str) -> str:
    """Give the same recording and language a stable, unique identity."""
    if audio.is_youtube_url(source):
        return f"yt-{audio.youtube_video_id(source)}-{language}"
    path = Path(source)
    if not path.is_file():
        raise audio.AudioError(f"File not found: {source}")
    if path.suffix.lower() not in audio.ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(audio.ALLOWED_EXTENSIONS))
        raise audio.AudioError(f"Unsupported file type. Use one of: {allowed}")
    return f"file-{audio.file_sha256(path)[:20]}-{language}"


def to_markdown(meeting: Meeting) -> str:
    def bullets(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items) or "_None found._"

    actions = [f"{item.task} (Owner: {item.owner}; Due: {item.due})" for item in meeting.action_items]
    return (
        f"# {meeting.title}\n\n## Summary\n\n{meeting.summary}\n\n"
        f"## Action Items\n\n{bullets(actions)}\n\n"
        f"## Decisions\n\n{bullets(meeting.decisions)}\n\n"
        f"## Open Questions\n\n{bullets(meeting.open_questions)}\n\n"
        f"## Transcript\n\n{meeting.transcript}\n"
    )


def process_meeting(
    source: str,
    language: str = config.ENGLISH,
    title: Optional[str] = None,
    force: bool = False,
    on_progress: Progress = None,
    llm: Optional[Any] = None,
    store: Optional[Any] = None,
) -> Meeting:
    def report(message: str, fraction: float) -> None:
        if on_progress:
            on_progress(message, fraction)

    def stage(message: str, start: float, end: float):
        return lambda done, total: report(f"{message} ({done}/{total})", start + (end - start) * done / total)

    if language not in config.LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")

    current_step, step_started, run_started = "setup", time.monotonic(), time.monotonic()

    def begin(step: str, fraction: float) -> None:
        nonlocal current_step, step_started
        logger.info("Finished %s in %.1fs", current_step, time.monotonic() - step_started)
        current_step, step_started = step, time.monotonic()
        logger.info("STEP: %s", step)
        report(step, fraction)

    logger.info("New request: language=%s, source=%s", language, Path(source).name)
    try:
        meeting_id = resolve_meeting_id(source, language)
    except Exception as exc:
        logger.error("Invalid input: %s", exc)
        raise
    if not force and (existing := load_meeting(meeting_id)):
        logger.info("Meeting %s is already saved, so it was loaded without reprocessing", meeting_id)
        report("Loaded saved meeting", 1.0)
        return existing

    try:
        meeting = _process(source, language, title, meeting_id, llm, store, begin, stage)
    except Exception:
        logger.exception("FAILED during step '%s' for meeting %s", current_step, meeting_id)
        raise
    logger.info("Finished %s in %.1fs", current_step, time.monotonic() - step_started)
    logger.info("DONE: meeting %s processed in %.1fs", meeting_id, time.monotonic() - run_started)
    report("Done", 1.0)
    return meeting


def _process(source, language, title, meeting_id, llm, store, begin, stage) -> Meeting:
    # Check setup first so a missing key does not waste a long download.
    audio.require_ffmpeg()
    if language == config.HINDI:
        config.require_key("SARVAM_API_KEY")
    llm = llm or get_llm()

    work_dir = config.WORK_DIR / meeting_id
    work_dir.mkdir(parents=True, exist_ok=True)

    begin("Preparing audio", 0.02)
    if audio.is_youtube_url(source):
        media, found_title = audio.download_youtube(source, work_dir)
    else:
        media, found_title = Path(source), Path(source).stem
    chunks = audio.prepare_chunks(media, work_dir, language)

    begin("Transcribing audio", 0.10)
    transcript = transcribe_all(chunks, language, stage("Transcribing", 0.10, 0.60))
    if not transcript:
        raise audio.AudioError("No speech could be transcribed from this recording.")

    begin("Summarizing", 0.60)
    summary, failed_summary = summarize(transcript, llm, stage("Summarizing", 0.60, 0.78))
    begin("Extracting notes", 0.78)
    notes, failed_notes = extract(transcript, llm, stage("Extracting notes", 0.78, 0.95))

    begin("Indexing for Q&A", 0.96)
    if store is None:
        from core.rag import MeetingStore

        store = MeetingStore()
    store.index(meeting_id, transcript)

    warnings = []
    if failed_summary:
        warnings.append(f"{failed_summary} transcript section(s) could not be summarized.")
    if failed_notes:
        warnings.append(f"{failed_notes} transcript section(s) could not be scanned for notes.")

    meeting = Meeting(
        meeting_id=meeting_id,
        title=(title or "").strip() or found_title,
        source=source if audio.is_youtube_url(source) else Path(source).name,
        language=language,
        created_at=datetime.now(timezone.utc).isoformat(),
        transcript=transcript,
        summary=summary,
        action_items=notes.action_items,
        decisions=notes.decisions,
        open_questions=notes.open_questions,
        warnings=warnings,
    )
    save_meeting(meeting)
    shutil.rmtree(work_dir, ignore_errors=True)
    return meeting
