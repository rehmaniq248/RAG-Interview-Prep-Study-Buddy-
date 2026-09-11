"""Interview mode: question parsing, grading requests, sampling, cost tracking."""

import random

import anthropic
import pytest

from rag_buddy import config, interview


def rows(n, source="doc.md"):
    return [{"id": f"{source}::{i}", "text": f"chunk {i}",
             "metadata": {"source": source, "section": f"S{i}", "citation": f"{source} § S{i}"}}
            for i in range(n)]


@pytest.fixture
def store_with(monkeypatch):
    def install(sources):
        monkeypatch.setattr(interview, "collection_stats", lambda: {
            "exists": True, "count": sum(len(r) for r in sources.values()),
            "sources": sorted(sources)})
        monkeypatch.setattr(interview, "get_chunks_by_source", lambda s: sources[s])
    return install


class TestGenerateQuestions:
    def test_parses_q_lines_and_ignores_everything_else(self, fakes, make_hit):
        client = fakes.Client(fakes.Response("Here you go:\nQ: First?\nnot a question\nQ: Second?\n"))
        assert interview.generate_questions([make_hit()], n=5, client=client) == ["First?", "Second?"]

    def test_indented_q_lines_have_their_prefix_removed(self, fakes, make_hit):
        client = fakes.Client(fakes.Response("  Q: Indented question?\nQ: Normal?"))
        assert interview.generate_questions([make_hit()], n=5, client=client) == [
            "Indented question?", "Normal?"]

    def test_caps_at_n(self, fakes, make_hit):
        client = fakes.Client(fakes.Response("\n".join(f"Q: q{i}?" for i in range(9))))
        assert len(interview.generate_questions([make_hit()], n=3, client=client)) == 3

    def test_request_uses_generation_model_and_exact_count(self, fakes, make_hit):
        client = fakes.Client(fakes.Response("Q: a?"))
        interview.generate_questions([make_hit(text="the excerpt")], n=7, client=client)
        call = client.messages.calls[0]
        assert call["model"] == config.GENERATION_MODEL
        assert call["system"] == interview.QUESTION_SYSTEM_PROMPT
        assert "exactly 7" in call["messages"][0]["content"]
        assert "the excerpt" in call["messages"][0]["content"]

    def test_records_usage(self, fakes, make_hit):
        usage = interview.Usage()
        client = fakes.Client(fakes.Response("Q: a?", input_tokens=1000, output_tokens=200))
        interview.generate_questions([make_hit()], n=1, client=client, usage=usage)
        assert (usage.calls, usage.input_tokens, usage.output_tokens) == (1, 1000, 200)
        assert usage.cost == pytest.approx(config.price_of(config.GENERATION_MODEL, 1000, 200))


class TestGradeAnswer:
    def test_request_carries_notes_then_question_then_answer(self, fakes, make_hit):
        client = fakes.Client(fakes.Response("VERDICT: STRONG — fine."))
        out = interview.grade_answer("Why X?", "Because Y.", [make_hit(text="Notes say Y.")],
                                     client=client)
        call = client.messages.calls[0]
        assert call["model"] == config.EVALUATION_MODEL
        assert call["system"] == interview.GRADER_SYSTEM_PROMPT
        body = call["messages"][0]["content"]
        assert body.index("Notes say Y.") < body.index("Why X?") < body.index("Because Y.")
        assert out.startswith("VERDICT")

    def test_grading_is_priced_at_the_evaluation_model(self, fakes, make_hit):
        usage = interview.Usage()
        client = fakes.Client(fakes.Response("VERDICT: WEAK", input_tokens=1400, output_tokens=300))
        interview.grade_answer("q", "a", [make_hit()], client=client, usage=usage)
        assert usage.cost == pytest.approx(config.price_of(config.EVALUATION_MODEL, 1400, 300))

    def test_grader_prompt_keeps_the_calibration_rules(self):
        prompt = interview.GRADER_SYSTEM_PROMPT
        for term in ("CONTRADICTS", "MISSING", "UNVERIFIABLE", "STRONG", "ADEQUATE", "WEAK"):
            assert term in prompt
        assert "must never lower the verdict" in prompt
        assert "must quote actual text from the excerpts" in prompt


class TestPickSourceExcerpts:
    def test_empty_store_raises(self, monkeypatch):
        monkeypatch.setattr(interview, "collection_stats",
                            lambda: {"exists": True, "count": 0, "sources": []})
        with pytest.raises(RuntimeError, match="empty"):
            interview.pick_source_excerpts()

    def test_unknown_source_lists_the_available_ones(self, store_with):
        store_with({"resume.md": rows(3, "resume.md")})
        with pytest.raises(RuntimeError, match="resume.md"):
            interview.pick_source_excerpts("nope.md")

    def test_samples_at_most_max_chunks_kept_in_document_order(self, store_with):
        store_with({"doc.md": rows(20)})
        random.seed(0)
        source, excerpts = interview.pick_source_excerpts("doc.md", max_chunks=5)
        order = [int(e.section[1:]) for e in excerpts]
        assert source == "doc.md"
        assert len(excerpts) == 5
        assert order == sorted(order)

    def test_small_document_is_used_whole(self, store_with):
        store_with({"doc.md": rows(3)})
        _, excerpts = interview.pick_source_excerpts("doc.md", max_chunks=6)
        assert [e.text for e in excerpts] == ["chunk 0", "chunk 1", "chunk 2"]

    def test_excerpts_numbered_from_one(self, store_with):
        store_with({"doc.md": rows(2)})
        _, excerpts = interview.pick_source_excerpts("doc.md")
        assert [e.rank for e in excerpts] == [1, 2]


def test_usage_accumulates_and_summarises(fakes):
    usage = interview.Usage()
    usage.add(fakes.Response(input_tokens=10, output_tokens=5), config.GENERATION_MODEL)
    usage.add(fakes.Response(input_tokens=20, output_tokens=5), config.GENERATION_MODEL)
    assert (usage.calls, usage.input_tokens, usage.output_tokens) == (2, 30, 10)
    assert "2 API call(s)" in usage.summary()


def test_call_maps_auth_errors(fakes, api_error):
    client = fakes.Client(api_error(anthropic.AuthenticationError, 401))
    with pytest.raises(RuntimeError, match="API key"):
        interview._call(client, "model", "system", "user", 10)


def test_run_session_generates_grades_and_returns_usage(store_with, fakes, scripted_input,
                                                        monkeypatch):
    store_with({"doc.md": rows(2)})
    client = fakes.Client(fakes.Response("Q: First?\nQ: Second?"),
                          fakes.Response("VERDICT: STRONG — ok."))
    monkeypatch.setattr(interview, "get_client", lambda: client)
    scripted_input("My answer to the first.", "skip")
    usage = interview.run_session(source="doc.md", n_questions=2)
    assert usage.calls == 2
    assert client.messages.calls[1]["model"] == config.EVALUATION_MODEL
    assert "My answer to the first." in client.messages.calls[1]["messages"][0]["content"]


def test_grading_gets_its_own_budget_and_effort(fakes, make_hit):
    """
    Grading runs on a thinking model, whose reasoning comes out of the same
    max_tokens budget as the grade. Sized like an answer, it returned nothing.
    """
    client = fakes.Client(fakes.Response("VERDICT: WEAK — thin."))
    interview.grade_answer("q", "a", [make_hit()], client=client)
    call = client.messages.calls[0]
    assert call["max_tokens"] == config.EVALUATION_MAX_TOKENS
    assert call["output_config"] == {"effort": config.EVALUATION_EFFORT}


def test_question_generation_sends_no_effort_setting(fakes, make_hit):
    # Haiku 4.5 rejects the effort parameter outright.
    client = fakes.Client(fakes.Response("Q: a?"))
    interview.generate_questions([make_hit()], n=1, client=client)
    assert "output_config" not in client.messages.calls[0]


def test_empty_grade_raises_instead_of_printing_nothing(fakes, make_hit):
    client = fakes.Client(fakes.Response(blocks=[fakes.Block("", type="thinking")],
                                         stop_reason="max_tokens"))
    with pytest.raises(RuntimeError, match="no grade text"):
        interview.grade_answer("q", "a", [make_hit()], client=client)
