"""
Retrieval quality benchmark on the fixed fictional corpus.

Core tests prove the ranking LOGIC is right. They cannot tell you whether the
real models still put the right passage in front of Claude, or whether the
free-refusal threshold still separates answerable questions from unanswerable
ones. A model or dependency upgrade can break either silently — this is the
test that notices.

Run `pytest -m model -s tests/integration/test_retrieval_quality.py` to see
the per-question report.
"""

import pytest

from corpus_data import ANSWERABLE, UNANSWERABLE, install_corpus
from rag_buddy import config, generate, interview, retrieve, store

# How many of the ANSWERABLE questions must get their passage into the context
# sent to Claude. Measured: all 12 with reranking.
MIN_RECALL = len(ANSWERABLE)

# The conservative value documented in config for anyone enabling free refusal.
SUGGESTED_THRESHOLD = -10.5


@pytest.fixture(scope="module")
def indexed(workspace):
    install_corpus(config.DOCUMENTS_DIR)
    return store.build_index()


def _found(hits, needle):
    return any(needle in hit.text for hit in hits)


@pytest.fixture(scope="module")
def measurements(indexed):
    answerable = []
    for question, needle in ANSWERABLE:
        reranked = retrieve.retrieve(question)
        vector = retrieve.retrieve(question, rerank=False)
        answerable.append({
            "question": question,
            "reranked_hit": _found(reranked, needle),
            "vector_hit": _found(vector, needle),
            "best": max(h.rerank_score for h in reranked),
        })
    unanswerable = {q: max(h.rerank_score for h in retrieve.retrieve(q)) for q in UNANSWERABLE}

    print(f"\n{'question':<74}{'rerank':>7}{'vector':>7}{'best':>8}")
    for row in answerable:
        print(f"{row['question'][:72]:<74}{'hit' if row['reranked_hit'] else 'MISS':>7}"
              f"{'hit' if row['vector_hit'] else 'MISS':>7}{row['best']:>+8.2f}")
    print("unanswerable:")
    for question, score in unanswerable.items():
        print(f"  {question[:70]:<72}{score:>+8.2f}")
    print(f"refusal threshold: {config.RERANK_MIN_SCORE}")
    return answerable, unanswerable


def test_correct_passage_reaches_the_model(measurements):
    answerable, _ = measurements
    misses = [r["question"] for r in answerable if not r["reranked_hit"]]
    assert len(answerable) - len(misses) >= MIN_RECALL, f"missed: {misses}"


def test_reranking_is_never_worse_than_vector_search_alone(measurements):
    answerable, _ = measurements
    assert sum(r["reranked_hit"] for r in answerable) >= sum(r["vector_hit"] for r in answerable)


def test_shipped_threshold_never_refuses_an_answerable_question(measurements):
    """
    This test caught a real bug. At the original default of -7.0, two questions
    this corpus DOES answer scored -7.6 and -9.25 and would have been refused
    without ever calling the API. Holds for any RERANK_MIN_SCORE, including
    None (off), which is the default because of this result.
    """
    answerable, _ = measurements
    threshold = config.RERANK_MIN_SCORE
    refused = {} if threshold is None else {
        r["question"]: round(r["best"], 2) for r in answerable if r["best"] < threshold}
    assert not refused, f"answerable but would be refused without calling the API: {refused}"


def test_suggested_conservative_threshold_is_safe_on_this_corpus(measurements):
    answerable, unanswerable = measurements
    assert min(r["best"] for r in answerable) > SUGGESTED_THRESHOLD
    assert max(unanswerable.values()) < SUGGESTED_THRESHOLD


def test_scores_separate_within_this_corpus(measurements):
    # Within a single corpus the two groups do separate — which is exactly why a
    # threshold looked safe when calibrated on one. Across corpora they overlap.
    answerable, unanswerable = measurements
    assert max(unanswerable.values()) < min(r["best"] for r in answerable)


def test_by_default_unanswerable_questions_reach_the_grounded_model(indexed, fakes):
    client = fakes.Client(fakes.Response("Your documents don't cover this."))
    generate.answer_question("What is your experience with Kubernetes?", client=client)
    assert len(client.messages.calls) == 1
    assert client.messages.calls[0]["system"] == generate.SYSTEM_PROMPT


def test_free_refusal_works_end_to_end_when_enabled(indexed, monkeypatch):
    monkeypatch.setattr(config, "RERANK_MIN_SCORE", SUGGESTED_THRESHOLD)
    # No client passed: constructing a real one would trip the API blocker.
    answer = generate.answer_question("What is your experience with Kubernetes?")
    assert answer.cost == 0.0
    assert "don't appear to cover" in answer.text


def test_answerable_question_sends_the_right_passage(indexed, fakes):
    client = fakes.Client(fakes.Response("ok"))
    generate.answer_question("How did you speed up the nightly ETL pipeline?", client=client)
    assert "6 hours to 40 minutes" in client.messages.calls[0]["messages"][0]["content"]


def test_interview_excerpts_come_from_one_document_in_order(indexed):
    _, excerpts = interview.pick_source_excerpts("resume.md", max_chunks=4)
    full = [row["text"] for row in store.get_chunks_by_source("resume.md")]
    positions = [full.index(e.text) for e in excerpts]
    assert {e.source for e in excerpts} == {"resume.md"}
    assert positions == sorted(positions)
