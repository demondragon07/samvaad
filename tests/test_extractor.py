"""Fix 2: long meetings are extracted chunk by chunk with nothing lost."""
import json
import re

import httpx
import pytest
from langchain_core.runnables import RunnableLambda

import config
from core.extractor import ActionItem, Extraction, extract, merge_extractions, parse_extraction


def long_transcript(markers=30, filler_words=120):
    filler = " ".join(["the team talked through routine status updates"] * (filler_words // 7))
    return " ".join(f"{filler}. Task TASKID{n:03d} must be finished." for n in range(markers))


class FakeLLM:
    """Answers map calls from the chunk text and records every prompt."""

    def __init__(self, consolidate=None, fail_on=None):
        self.prompts, self.consolidate, self.fail_on = [], consolidate, fail_on

    def __call__(self, prompt):
        text = prompt.to_string()
        self.prompts.append(text)
        if "clean up notes" in text:
            if self.consolidate is not None:
                return self.consolidate
            return text[text.index('{\n "action_items"') :]
        if self.fail_on and self.fail_on in text:
            return "not json at all"
        items = [{"task": f"Finish {m}", "owner": None} for m in re.findall(r"TASKID\d{3}", text)]
        return json.dumps({"action_items": items, "decisions": [], "open_questions": []})


def test_long_transcript_is_sent_in_bounded_chunks_and_keeps_late_items():
    transcript = long_transcript()
    assert len(transcript) > config.LLM_CHUNK_SIZE * 3
    llm = FakeLLM()

    notes, failed = extract(transcript, RunnableLambda(llm))

    map_prompts = [p for p in llm.prompts if "clean up notes" not in p]
    overhead = 1000
    assert failed == 0 and len(map_prompts) > 3
    assert all(len(p) <= config.LLM_CHUNK_SIZE + overhead for p in map_prompts)
    assert not any(transcript in p for p in map_prompts)
    assert {item.task for item in notes.action_items} == {f"Finish TASKID{n:03d}" for n in range(30)}


def test_overlap_duplicates_are_merged_and_details_kept():
    merged = merge_extractions([
        Extraction(action_items=[ActionItem(task="Send the deck.")], decisions=["Ship in June"]),
        Extraction(action_items=[ActionItem(task="send the deck", owner="Asha", due="Friday")],
                   decisions=["ship in june."], open_questions=["Who owns QA?"]),
    ])
    assert len(merged.action_items) == 1
    assert (merged.action_items[0].owner, merged.action_items[0].due) == ("Asha", "Friday")
    assert merged.decisions == ["Ship in June"]
    assert merged.open_questions == ["Who owns QA?"]


@pytest.mark.parametrize("bad_output", ["sorry, no JSON", '{"action_items": [], "decisions": [], "open_questions": []}'])
def test_bad_consolidation_keeps_merged_notes(bad_output):
    notes, _ = extract(long_transcript(), RunnableLambda(FakeLLM(consolidate=bad_output)))
    assert len(notes.action_items) == 30


def test_one_bad_chunk_is_reported_not_hidden():
    notes, failed = extract(long_transcript(), RunnableLambda(FakeLLM(fail_on="TASKID015")))
    assert failed >= 1
    tasks = {item.task for item in notes.action_items}
    assert "Finish TASKID000" in tasks and "Finish TASKID029" in tasks


def test_invalid_api_key_stops_immediately():
    calls = []

    def unauthorized(_prompt):
        calls.append(1)
        request = httpx.Request("POST", "https://example.test")
        raise httpx.HTTPStatusError("401", request=request, response=httpx.Response(401, request=request))

    with pytest.raises(httpx.HTTPStatusError):
        extract(long_transcript(), RunnableLambda(unauthorized))
    assert len(calls) == 1


def test_parser_accepts_fences_and_null_owner():
    raw = '```json\n{"action_items": [{"task": "Book room", "owner": null}], "decisions": []}\n```'
    parsed = parse_extraction(raw)
    assert parsed.action_items[0].owner == "Not specified"
