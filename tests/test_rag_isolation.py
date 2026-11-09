"""Fix 1: one meeting's content must never reach another meeting's answers."""
from langchain_core.runnables import RunnableLambda

from core.rag import MeetingStore, build_qa_chain

BUDGET = "Budget review. The marketing budget for Diwali campaign was approved at forty lakh rupees. " * 12
HIRING = "Hiring sync. We will interview three backend engineers for the payments team next Tuesday. " * 12


def store_with_two(embeddings, tmp_path):
    store = MeetingStore(embeddings=embeddings, persist_dir=str(tmp_path / "db"))
    store.index("budget", BUDGET)
    store.index("hiring", HIRING)
    return store


def test_retrieval_never_crosses_meetings(embeddings, tmp_path):
    store = store_with_two(embeddings, tmp_path)
    # Asked in the budget meeting using words that occur only in the hiring meeting.
    docs = store.retriever("budget", k=50).invoke("interview backend engineers payments team")
    assert docs
    assert {doc.metadata["meeting_id"] for doc in docs} == {"budget"}
    assert not any("engineers" in doc.page_content for doc in docs)


def test_same_filter_finds_the_right_meeting(embeddings, tmp_path):
    store = store_with_two(embeddings, tmp_path)
    docs = store.retriever("hiring").invoke("interview backend engineers")
    assert docs and all("engineers" in doc.page_content for doc in docs)


def test_prompt_only_receives_selected_meeting(embeddings, tmp_path):
    store = store_with_two(embeddings, tmp_path)
    seen = []

    def fake_llm(prompt):
        seen.append(prompt.to_string())
        return "ok"

    result = build_qa_chain("budget", RunnableLambda(fake_llm), store).invoke(
        {"question": "Which engineers will we interview?"}
    )
    assert result["answer"] == "ok"
    assert "Diwali" in seen[0] and "backend engineers" not in seen[0]


def test_reindex_replaces_only_that_meeting(embeddings, tmp_path):
    store = store_with_two(embeddings, tmp_path)
    budget_count, hiring_count = store.count("budget"), store.count("hiring")
    assert store.index("budget", BUDGET) == budget_count
    assert store.count("budget") == budget_count
    assert store.count("hiring") == hiring_count

    store.index("budget", "A short corrected transcript.")
    assert store.count("budget") == 1
    assert store.count("hiring") == hiring_count


def test_index_survives_reopen(embeddings, tmp_path):
    store_with_two(embeddings, tmp_path)
    reopened = MeetingStore(embeddings=embeddings, persist_dir=str(tmp_path / "db"))
    assert reopened.count("budget") > 0
