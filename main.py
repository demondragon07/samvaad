"""Command-line entry point: python main.py <youtube-url-or-file> --language en"""
from __future__ import annotations

import argparse

import config
from core.pipeline import process_meeting, to_markdown
from utils.logger import log_file, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Samvaad: bilingual AI meeting assistant")
    parser.add_argument("source", help="YouTube link or path to an MP3, MP4, M4A or WAV file")
    parser.add_argument("--language", choices=sorted(config.LANGUAGES), default=config.ENGLISH)
    parser.add_argument("--title", help="Optional meeting title")
    parser.add_argument("--force", action="store_true", help="Process again even if already saved")
    parser.add_argument("--chat", action="store_true", help="Ask questions after processing")
    args = parser.parse_args()

    setup_logging()
    print(f"Log file: {log_file()}")
    meeting = process_meeting(args.source, args.language, args.title, args.force)
    print("\n" + to_markdown(meeting).split("## Transcript")[0])
    for warning in meeting.warnings:
        print(f"Warning: {warning}")

    if args.chat:
        from core.llm import get_llm
        from core.rag import MeetingStore, build_qa_chain

        chain = build_qa_chain(meeting.meeting_id, get_llm(), MeetingStore())
        print("Ask about this meeting. Press Enter on an empty line to quit.")
        while question := input("\nYou: ").strip():
            print("Samvaad:", chain.invoke({"question": question})["answer"])


if __name__ == "__main__":
    main()
