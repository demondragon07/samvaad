from datetime import datetime, timezone
from pathlib import Path

from streamlit.testing.v1 import AppTest

from core import pipeline
from core.extractor import ActionItem

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def meeting(meeting_id, title):
    return pipeline.Meeting(
        meeting_id=meeting_id, title=title, source="x.mp3", language="en",
        created_at=datetime.now(timezone.utc).isoformat(), transcript="Hello.", summary=f"Summary of {title}",
        action_items=[ActionItem(task="Send minutes")], decisions=["Go"], open_questions=[],
    )


def test_empty_state_renders():
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    assert any("Turn a meeting" in header.value for header in app.header)


def test_saved_meetings_render_and_switch():
    pipeline.save_meeting(meeting("m1", "Budget"))
    pipeline.save_meeting(meeting("m2", "Hiring"))
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    selected = app.session_state["selected_id"]
    other = "m1" if selected == "m2" else "m2"
    app.selectbox(key="selected_id").select(other).run()
    assert not app.exception
    title = {"m1": "Budget", "m2": "Hiring"}[other]
    assert any(header.value == title for header in app.header)


def fake_store(monkeypatch):
    import streamlit as st

    import core.rag

    st.cache_resource.clear()
    monkeypatch.setattr(core.rag, "MeetingStore", lambda: object())


def click_process(app):
    app.text_input[0].set_value("https://www.youtube.com/watch?v=xlYJhtL0qbQ")
    next(button for button in app.button if button.label == "Process meeting").click().run()


def test_processing_success_opens_new_meeting(monkeypatch):
    fake_store(monkeypatch)
    pipeline.save_meeting(meeting("old", "Older meeting"))

    def fake_process(source, language, title, force, on_progress, store):
        on_progress("Working", 0.5)
        result = meeting("new", "Brand new meeting")
        result.created_at = "2000-01-01T00:00:00+00:00"
        pipeline.save_meeting(result)
        return result

    monkeypatch.setattr(pipeline, "process_meeting", fake_process)
    app = AppTest.from_file(APP, default_timeout=30).run()
    click_process(app)
    assert not app.exception
    assert app.session_state["selected_id"] == "new"
    assert any(header.value == "Brand new meeting" for header in app.header)


def test_processing_error_points_to_log(monkeypatch):
    fake_store(monkeypatch)

    def fail(*_args, **_kwargs):
        raise RuntimeError("Gemini key rejected")

    monkeypatch.setattr(pipeline, "process_meeting", fail)
    app = AppTest.from_file(APP, default_timeout=30).run()
    click_process(app)
    assert not app.exception
    assert "Gemini key rejected" in app.error[0].value
    assert "samvaad.log" in app.error[0].value
