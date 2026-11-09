"""LLM factory plus shared retry and text-splitting helpers."""
from __future__ import annotations

import time
from typing import Any, Callable, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from utils.logger import get_logger

logger = get_logger("llm")

ProgressFn = Optional[Callable[[int, int], None]]
_FATAL_STATUS = {400, 401, 403, 404, 422}


def get_llm(temperature: float = config.LLM_TEMPERATURE):
    from langchain_google_genai import ChatGoogleGenerativeAI

    logger.info("Using Gemini model %s", config.GEMINI_MODEL)
    return ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        google_api_key=config.require_key("GEMINI_API_KEY"),
        temperature=temperature,
        timeout=config.LLM_TIMEOUT_SECONDS,
        max_retries=1,  # invoke_with_retry owns retries, so they do not multiply.
    )


def split_text(text: str, size: int, overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", "। ", "? ", "! ", " ", ""]
    )
    return splitter.split_text(text)


def is_fatal(exc: Exception) -> bool:
    """Bad keys and invalid requests will fail again, so retrying only wastes time."""
    current: Optional[BaseException] = exc
    while current is not None:
        status = getattr(getattr(current, "response", None), "status_code", None) or getattr(current, "code", None)
        if status in _FATAL_STATUS:
            return True
        current = current.__cause__
    return False


def invoke_with_retry(chain: Any, inputs: dict) -> Any:
    """Invoke a chain, retrying rate limits, network errors and malformed output."""
    for attempt in range(config.LLM_ATTEMPTS):
        if config.LLM_CALL_GAP_SECONDS:
            time.sleep(config.LLM_CALL_GAP_SECONDS)
        try:
            return chain.invoke(inputs)
        except Exception as exc:
            if is_fatal(exc) or attempt == config.LLM_ATTEMPTS - 1:
                logger.error("LLM call failed: %s: %s", type(exc).__name__, str(exc)[:500])
                raise
            wait = config.LLM_BACKOFF_SECONDS * 2**attempt
            logger.warning("LLM attempt %d/%d failed (%s: %s). Retrying in %.0fs.",
                           attempt + 1, config.LLM_ATTEMPTS, type(exc).__name__, str(exc)[:300], wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


def map_chunks(chain: Any, chunks: list[str], on_progress: ProgressFn = None) -> tuple[list[Any], int]:
    """Run a chain over every chunk and return (results, number of failed chunks).

    One bad chunk should not throw away a long meeting, so it is counted and
    reported. If every chunk fails, or the API key is bad, the error is raised.
    """
    results, failed, last_error = [], 0, None
    for index, chunk in enumerate(chunks, start=1):
        logger.info("Text section %d/%d: sending to Gemini", index, len(chunks))
        try:
            results.append(invoke_with_retry(chain, {"text": chunk}))
        except Exception as exc:
            if is_fatal(exc):
                raise
            failed, last_error = failed + 1, exc
            logger.warning("Text section %d/%d was skipped after all retries", index, len(chunks))
        if on_progress:
            on_progress(index, len(chunks))
    if chunks and not results:
        raise RuntimeError(f"All {len(chunks)} transcript chunks failed. Last error: {last_error}")
    return results, failed
