"""Logging for Samvaad: terminal, data/logs/samvaad.log and the live app panel."""
from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, Iterator

import config

ROOT_NAME = "samvaad"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{ROOT_NAME}.{name}")


def log_file() -> Path:
    return config.DATA_DIR / "logs" / "samvaad.log"


def setup_logging() -> logging.Logger:
    """Attach handlers once, even when Streamlit re-runs the script."""
    logger = logging.getLogger(ROOT_NAME)
    if any(isinstance(handler, RotatingFileHandler) for handler in logger.handlers):
        return logger

    # Third-party loggers stay quiet, which also keeps request details out of the file.
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    log_file().parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
    handlers = [
        logging.StreamHandler(sys.stderr),
        RotatingFileHandler(log_file(), maxBytes=1_000_000, backupCount=3, encoding="utf-8", delay=True),
    ]
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def tail_log(lines: int = 40) -> str:
    """Return the newest log lines, or an empty string before the first run."""
    path = log_file()
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


class _ForwardHandler(logging.Handler):
    def __init__(self, callback: Callable[[str], None]):
        super().__init__(level=logging.INFO)
        self.callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        clock = time.strftime("%H:%M:%S", time.localtime(record.created))
        level = "" if record.levelno < logging.WARNING else f"{record.levelname}: "
        self.callback(f"{clock}  {level}{record.getMessage()}")


@contextmanager
def capture_logs(callback: Callable[[str], None]) -> Iterator[None]:
    """Send readable log lines to a live display while a block runs."""
    logger = logging.getLogger(ROOT_NAME)
    handler = _ForwardHandler(callback)
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
