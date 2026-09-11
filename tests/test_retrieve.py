"""Retrieval ranking: candidate fetch, cross-encoder reranking, the diversity cap."""

import pytest

from rag_buddy import config, retrieve


class FakeCollection:
    def __init__(self, rows):
        self.rows = rows              # (text, source, distance), in vector order
        self.queries = []

    def count(self):
        return len(self.rows)

    def query(self, **kwargs):
        self.queries.append(kwargs)
        rows = self.rows[: kwargs["n_results"]]
        return {
            "documents": [[r[0] for r in rows]],
            "metadatas": [[{"source": r[1], "section": "S", "citation": f"{r[1]} § S"}
                           for r in rows]],
            "distances": [[r[2] for r in rows]],
        }


class FakeReranker:
    def __init__(self, scores):
        self.scores = scores          # passage text -> score
        self.pairs = []

    def predict(self, pairs):
        self.pairs.extend(pairs)
        return [self.scores[text] for _, text in pairs]


@pytest.fixture
def wire(monkeypatch):
    def install(rows, scores=None):
        collection = FakeCollection(rows)
        reranker = FakeReranker(scores or {})
        monkeypatch.setattr(retrieve, "get_collection", lambda: collection)
        monkeypatch.setattr(retrieve, "embed_texts", lambda texts, show_progress=True: [[0.0]])
        monkeypatch.setattr(retrieve, "get_reranker", lambda: reranker)
        return collection, reranker
    return install


def rows(n):
    return [(f"text {i}", f"doc{i}.md", 0.10 + i * 0.01) for i in range(n)]


def flat_scores(n):
    return {f"text {i}": 0.0 for i in range(n)}


def test_similarity_is_one_minus_cosine_distance(make_hit):
    assert make_hit(distance=0.25).similarity == pytest.approx(0.75)


def test_estimate_context_tokens(make_hit):
    hits = [make_hit(text="one two three four five"), make_hit(text="six seven eight nine ten")]
    assert retrieve.estimate_context_tokens(hits) == int(10 * config.WORDS_TO_TOKENS_RATIO)


class TestDiversityCap:
    def test_limits_chunks_per_source_and_renumbers(self, make_hit):
        hits = [make_hit(source=s, text=f"t{i}") for i, s in enumerate("aaabc")]
        chosen = retrieve._apply_diversity_cap(hits, max_per_source=2, limit=4)
        assert [h.source for h in chosen] == ["a", "a", "b", "c"]
        assert [h.rank for h in chosen] == [1, 2, 3, 4]

    def test_backfills_rather_than_returning_too_few(self, make_hit):
        hits = [make_hit(source="a", text=f"t{i}") for i in range(5)]
        chosen = retrieve._apply_diversity_cap(hits, max_per_source=2, limit=4)
        assert [h.text for h in chosen] == ["t0", "t1", "t2", "t3"]

    def test_fewer_hits_than_limit_returns_them_all(self, make_hit):
        hits = [make_hit(source="a"), make_hit(source="b")]
        assert len(retrieve._apply_diversity_cap(hits, max_per_source=2, limit=4)) == 2


class TestRetrieve:
    def test_rerank_fetches_candidates_and_reorders_by_reranker(self, wire):
        # The reranker prefers passages vector search ranked LOWER.
        collection, reranker = wire(rows(30), {f"text {i}": float(i) for i in range(30)})
        hits = retrieve.retrieve("q", top_k=4, rerank=True, max_per_source=99)
        assert collection.queries[0]["n_results"] == config.RERANK_CANDIDATES
        assert [h.text for h in hits] == ["text 19", "text 18", "text 17", "text 16"]
        assert hits[0].rerank_score == 19.0
        assert len(reranker.pairs) == config.RERANK_CANDIDATES

    def test_reranker_reads_question_with_each_passage(self, wire):
        _, reranker = wire(rows(3), flat_scores(3))
        retrieve.retrieve("why postgis", rerank=True)
        assert all(question == "why postgis" for question, _ in reranker.pairs)

    def test_without_rerank_fetches_only_top_k_in_vector_order(self, wire):
        collection, reranker = wire(rows(30))
        hits = retrieve.retrieve("q", top_k=4, rerank=False)
        assert collection.queries[0]["n_results"] == 4
        assert [h.text for h in hits] == ["text 0", "text 1", "text 2", "text 3"]
        assert all(h.rerank_score is None for h in hits)
        assert reranker.pairs == []

    def test_never_requests_more_rows_than_exist(self, wire):
        collection, _ = wire(rows(3), flat_scores(3))
        retrieve.retrieve("q", top_k=4, rerank=True)
        assert collection.queries[0]["n_results"] == 3

    def test_source_filter_passed_to_store(self, wire):
        collection, _ = wire(rows(5), flat_scores(5))
        retrieve.retrieve("q", source="resume.md")
        assert collection.queries[0]["where"] == {"source": "resume.md"}

    def test_no_filter_without_source(self, wire):
        collection, _ = wire(rows(5), flat_scores(5))
        retrieve.retrieve("q")
        assert "where" not in collection.queries[0]

    def test_empty_store_explains_how_to_fill_it(self, wire):
        wire([])
        with pytest.raises(RuntimeError, match="rag_buddy.store"):
            retrieve.retrieve("q")

    def test_min_similarity_drops_weak_matches(self, wire):
        wire([("close", "a.md", 0.2), ("far", "b.md", 0.9)])
        hits = retrieve.retrieve("q", rerank=False, min_similarity=0.5)
        assert [h.text for h in hits] == ["close"]

    def test_diversity_cap_applied_after_reranking(self, wire):
        data = [("a1", "same.md", 0.1), ("a2", "same.md", 0.2),
                ("a3", "same.md", 0.3), ("b1", "other.md", 0.4)]
        wire(data, {"a1": 9.0, "a2": 8.0, "a3": 7.0, "b1": 1.0})
        hits = retrieve.retrieve("q", top_k=3, rerank=True, max_per_source=2)
        assert [h.text for h in hits] == ["a1", "a2", "b1"]
