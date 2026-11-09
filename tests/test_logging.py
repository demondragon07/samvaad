import logging
import shutil
import subprocess
from logging.handlers import RotatingFileHandler

import pytest
from langchain_core.runnables import RunnableLambda

import config
from core import pipeline
from utils import logger as log


@pytest.fixture(autouse=True)
def fresh_logger():
    root = logging.getLogger(log.ROOT_NAME)
    saved = root.handlers[:]
    root.handlers.clear()
    log.setup_logging()
    yield
    for handler in root.handlers:
        handler.close()
    root.handlers[:] = saved


def test_setup_is_idempotent_and_writes_file():
    log.setup_logging()
    handlers = logging.getLogger(log.ROOT_NAME).handlers
    assert sum(isinstance(handler, RotatingFileHandler) for handler in handlers) == 1
    assert sum(type(handler) is logging.StreamHandler for handler in handlers) == 1
    log.get_logger("test").info("hello नमस्ते")
    assert "hello नमस्ते" in log.tail_log()
    assert log.log_file().parent == config.DATA_DIR / "logs"


def test_capture_sends_lines_and_detaches():
    seen = []
    with log.capture_logs(seen.append):
        log.get_logger("test").info("inside")
        log.get_logger("test").warning("careful")
    log.get_logger("test").info("outside")
    assert "inside" in seen[0] and "WARNING: careful" in seen[1] and len(seen) == 2


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_failure_records_step_and_keeps_transcript_out(tmp_path, monkeypatch, embeddings):
    media = tmp_path / "a.mp3"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", "sine=frequency=300:duration=2", str(media)], check=True)
    secret = "SECRETWORD spoken in the meeting."
    monkeypatch.setattr(pipeline, "transcribe_all", lambda *_args, **_kwargs: secret)

    def broken(_prompt):
        raise ValueError("model exploded")

    with pytest.raises(RuntimeError):
        pipeline.process_meeting(str(media), llm=RunnableLambda(broken), store=object())

    text = log.tail_log(200)
    assert "STEP: Preparing audio" in text and "STEP: Summarizing" in text
    assert "FAILED during step 'Summarizing'" in text
    assert "SECRETWORD" not in text
