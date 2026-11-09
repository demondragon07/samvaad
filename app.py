"""Streamlit interface for Samvaad."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

import config
from core.pipeline import Meeting, list_meetings, process_meeting, to_markdown
from utils.audio_processor import ALLOWED_EXTENSIONS
from utils.logger import capture_logs, get_logger, log_file, setup_logging, tail_log

st.set_page_config(page_title="Samvaad", page_icon="💬", layout="wide")
setup_logging()
logger = get_logger("app")


@st.cache_resource(show_spinner="Loading the search model...")
def get_store():
    from core.rag import MeetingStore

    return MeetingStore()


@st.cache_resource
def get_chat_llm():
    from core.llm import get_llm

    return get_llm()


def save_upload(upload) -> str:
    target_dir = config.WORK_DIR / "uploads"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{Path(upload.name).stem[:60]}{Path(upload.name).suffix.lower()}"
    target.write_bytes(upload.getbuffer())
    return str(target)


def run_pipeline(source: str, language: str, title: str, force: bool) -> None:
    st.caption("Please keep this tab open and avoid other controls until processing finishes.")
    bar = st.progress(0.0, text="Starting")
    log_box = st.empty()
    lines: list[str] = []

    def show_log(line: str) -> None:
        lines.append(line)
        log_box.code("\n".join(lines[-14:]), language=None, wrap_lines=True)

    try:
        with capture_logs(show_log):
            try:
                store = get_store()
            except Exception:
                logger.exception("Could not load the search model")
                raise
            meeting = process_meeting(
                source, language, title, force,
                on_progress=lambda message, fraction: bar.progress(min(fraction, 1.0), text=message),
                store=store,
            )
    except Exception as exc:
        bar.empty()
        st.error(f"{exc}\n\nFull details are in {log_file()}")
        return
    finally:
        if Path(source).parent == config.WORK_DIR / "uploads":
            Path(source).unlink(missing_ok=True)
    bar.empty()
    st.session_state.pending_id = meeting.meeting_id
    st.rerun()


def sidebar() -> tuple[list[Meeting], tuple | None]:
    request = None
    with st.sidebar:
        st.title("Samvaad")
        st.caption("Bilingual AI meeting assistant")

        kind = st.radio("Source", ["YouTube link", "Upload file"], horizontal=True)
        url = upload = None
        if kind == "YouTube link":
            url = st.text_input("Video link", placeholder="https://www.youtube.com/watch?v=...")
        else:
            upload = st.file_uploader("Recording", type=[ext.lstrip(".") for ext in sorted(ALLOWED_EXTENSIONS)])

        language = st.radio(
            "Spoken language", list(config.LANGUAGES), format_func=config.LANGUAGES.get, horizontal=True
        )
        title = st.text_input("Meeting title (optional)")
        force = st.checkbox("Process again if already saved")

        if st.button("Process meeting", type="primary", width="stretch"):
            source = url.strip() if url else save_upload(upload) if upload else ""
            if source:
                request = (source, language, title, force)
            else:
                st.warning("Add a link or a file first.")

        meetings = list_meetings()
        if meetings:
            st.divider()
            ids = [meeting.meeting_id for meeting in meetings]
            if "pending_id" in st.session_state:
                st.session_state.selected_id = st.session_state.pop("pending_id")
            titles = {meeting.meeting_id: meeting.title for meeting in meetings}
            if st.session_state.get("selected_id") not in ids:
                st.session_state.selected_id = ids[0]
            st.selectbox("Saved meetings", ids, format_func=titles.get, key="selected_id")

        st.divider()
        with st.expander("Activity log"):
            st.code(tail_log(30) or "No activity yet.", language=None, wrap_lines=True)
            st.caption(str(log_file()))
    return meetings, request


def bullet_list(items: list[str], empty: str) -> None:
    if items:
        st.markdown("\n".join(f"- {item}" for item in items))
    else:
        st.caption(empty)


def show_report(meeting: Meeting) -> None:
    summary, actions, decisions, questions, transcript = st.tabs(
        ["Summary", f"Actions ({len(meeting.action_items)})", f"Decisions ({len(meeting.decisions)})",
         f"Questions ({len(meeting.open_questions)})", "Transcript"]
    )
    with summary:
        st.markdown(meeting.summary)
    with actions:
        if meeting.action_items:
            rows = [{"Task": item.task, "Owner": item.owner, "Due": item.due} for item in meeting.action_items]
            st.dataframe(rows, width="stretch", hide_index=True)
        else:
            st.caption("No action items found.")
    with decisions:
        bullet_list(meeting.decisions, "No decisions found.")
    with questions:
        bullet_list(meeting.open_questions, "No open questions found.")
    with transcript:
        st.text_area("Transcript", meeting.transcript, height=420, disabled=True, label_visibility="collapsed")


def show_chat(meeting: Meeting) -> None:
    st.subheader("Ask this meeting")
    # History is keyed by meeting so conversations never mix either.
    history = st.session_state.setdefault("chats", {}).setdefault(meeting.meeting_id, [])
    box = st.container(height=440, border=True)
    with box:
        for message in history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    question = st.chat_input("What was decided about...?", key=f"chat_{meeting.meeting_id}")
    if not question:
        return

    history.append({"role": "user", "content": question})
    with box:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"), st.spinner("Searching this meeting..."):
            try:
                logger.info("Chat question for meeting %s", meeting.meeting_id)
                from core.rag import build_qa_chain

                answer = build_qa_chain(meeting.meeting_id, get_chat_llm(), get_store()).invoke(
                    {"question": question}
                )["answer"]
            except Exception as exc:
                logger.exception("Chat failed for meeting %s", meeting.meeting_id)
                answer = f"Sorry, I could not answer that: {exc}"
            st.markdown(answer)
    history.append({"role": "assistant", "content": answer})


def main() -> None:
    meetings, request = sidebar()
    if request:
        run_pipeline(*request)
    if not meetings:
        st.header("Turn a meeting into notes you can question")
        st.write("Add a YouTube link or recording in the sidebar to get a summary, action items, "
                 "decisions, open questions and a chat grounded in the transcript.")
        return

    meeting = next(item for item in meetings if item.meeting_id == st.session_state.selected_id)
    heading, download = st.columns([5, 1], vertical_alignment="bottom")
    heading.header(meeting.title)
    download.download_button(
        "Download notes", to_markdown(meeting), file_name=f"{meeting.meeting_id}.md",
        mime="text/markdown", width="stretch",
    )
    st.caption(f"{config.LANGUAGES[meeting.language]} audio · {meeting.source}")
    for warning in meeting.warnings:
        st.warning(warning)

    report, chat = st.columns([3, 2], gap="large")
    with report:
        show_report(meeting)
    with chat:
        show_chat(meeting)


main()
