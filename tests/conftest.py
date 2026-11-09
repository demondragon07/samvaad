import hashlib
import json
import re

import pytest
from langchain_core.embeddings import Embeddings

import config


class WordHashEmbeddings(Embeddings):
    """Offline embeddings where shared words produce nearby vectors."""

    size = 256

    def _embed(self, text):
        vector = [0.0] * self.size
        for word in re.findall(r"\w+", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.size] += 1.0
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(config, "MEETINGS_DIR", tmp_path / "meetings")
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(config, "LLM_CALL_GAP_SECONDS", 0)
    monkeypatch.setattr(config, "LLM_BACKOFF_SECONDS", 0)


@pytest.fixture
def embeddings():
    return WordHashEmbeddings()


def empty_notes():
    return json.dumps({"action_items": [], "decisions": [], "open_questions": []})
