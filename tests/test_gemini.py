"""Run the real LangChain Gemini class against a fake Google client."""
import json

import pytest
from google.genai import errors, types

import config
from core.extractor import extract
from core.llm import get_llm
from core.summarizer import summarize


class FakeModels:
    def __init__(self, behavior):
        self.behavior, self.calls = behavior, 0

    def generate_content(self, **_request):
        self.calls += 1
        return self.behavior(self.calls)


def response(text):
    part = types.Part(text=text, thought_signature=b"signature")
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[part]), finish_reason="STOP")]
    )


def llm_with(monkeypatch, behavior):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    llm = get_llm()
    fake = FakeModels(behavior)
    monkeypatch.setattr(type(llm.client), "models", fake)
    return llm, fake


def error(code, status):
    return errors.ClientError(code, {"error": {"code": code, "status": status, "message": status}})


NOTES = json.dumps({"action_items": [{"task": "Send minutes", "owner": "Asha"}], "decisions": [], "open_questions": []})


def test_gemini_text_reaches_summary_and_extraction(monkeypatch):
    llm, _ = llm_with(monkeypatch, lambda _call: response("Short summary."))
    assert summarize("Hello team.", llm)[0] == "Short summary."

    llm, _ = llm_with(monkeypatch, lambda _call: response(f"```json\n{NOTES}\n```"))
    notes, failed = extract("Asha will send minutes.", llm)
    assert failed == 0 and notes.action_items[0].owner == "Asha"


@pytest.mark.parametrize("code,status", [(400, "INVALID_ARGUMENT"), (403, "PERMISSION_DENIED"), (404, "NOT_FOUND")])
def test_bad_key_or_model_stops_immediately(monkeypatch, code, status):
    def fail(_call):
        raise error(code, status)

    llm, fake = llm_with(monkeypatch, fail)
    with pytest.raises(Exception, match=status):
        summarize("Hello team.", llm)
    assert fake.calls == 1


def test_rate_limit_is_retried(monkeypatch):
    def behavior(call):
        if call == 1:
            raise error(429, "RESOURCE_EXHAUSTED")
        return response("Recovered.")

    llm, fake = llm_with(monkeypatch, behavior)
    assert summarize("Hello team.", llm)[0] == "Recovered."
    assert fake.calls == 3


def test_missing_key_is_clear(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        get_llm()


def test_default_model_setting():
    assert config.GEMINI_MODEL.startswith("gemini-")
