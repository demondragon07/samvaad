"""End-to-end with real FFmpeg and Chroma; only speech and LLM calls are faked."""
import json
import shutil
import subprocess

import pytest
from langchain_core.runnables import RunnableLambda

import config
from core import pipeline
from core.rag import MeetingStore, build_qa_chain

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def fake_llm():
    def respond(prompt):
        text = prompt.to_string()
        if "extract structured notes" in text or "clean up notes" in text:
            return json.dumps({"action_items": [{"task": "Send minutes", "owner": "Asha", "due": "Friday"}],
                               "decisions": ["Launch approved"], "open_questions": []})
        return "A summary."

    return RunnableLambda(respond)


def tone(path, frequency):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", f"sine=frequency={frequency}:duration=3", str(path)], check=True)
    return path


def test_two_recordings_stay_separate_and_cache(tmp_path, monkeypatch, embeddings):
    transcripts = iter(["Alpha project launch discussion. " * 300, "Beta vendor contract discussion. " * 300])
    calls = []

    def fake_transcribe(chunks, language, on_progress=None):
        calls.append(language)
        return next(transcripts)

    monkeypatch.setattr(pipeline, "transcribe_all", fake_transcribe)
    store = MeetingStore(embeddings=embeddings, persist_dir=str(tmp_path / "db"))
    llm = fake_llm()

    first = pipeline.process_meeting(str(tone(tmp_path / "a.mp3", 300)), llm=llm, store=store)
    second = pipeline.process_meeting(str(tone(tmp_path / "b.mp3", 900)), llm=llm, store=store)

    assert first.meeting_id != second.meeting_id
    assert first.action_items[0].owner == "Asha"
    assert len(pipeline.list_meetings()) == 2
    assert not (config.WORK_DIR / first.meeting_id).exists()

    result = build_qa_chain(first.meeting_id, llm, store).invoke({"question": "Beta vendor contract?"})
    assert result["docs"] and all(doc.metadata["meeting_id"] == first.meeting_id for doc in result["docs"])

    again = pipeline.process_meeting(str(tmp_path / "a.mp3"), llm=llm, store=store)
    assert again.created_at == first.created_at and len(calls) == 2


def test_same_file_in_other_language_is_a_new_meeting(tmp_path):
    media = tone(tmp_path / "a.mp3", 300)
    assert pipeline.resolve_meeting_id(str(media), "en") != pipeline.resolve_meeting_id(str(media), "hi")


def test_missing_sarvam_key_fails_before_audio_work(tmp_path, monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SARVAM_API_KEY"):
        pipeline.process_meeting(str(tone(tmp_path / "a.mp3", 300)), language="hi", llm=fake_llm())


def test_unsupported_file_rejected(tmp_path):
    path = tmp_path / "notes.exe"
    path.write_text("x")
    with pytest.raises(pipeline.audio.AudioError, match="Unsupported"):
        pipeline.resolve_meeting_id(str(path), "en")
