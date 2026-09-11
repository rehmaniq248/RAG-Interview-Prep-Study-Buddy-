"""Prompt assembly, the free-refusal path, and API error handling."""

import anthropic
import pytest

from rag_buddy import config, generate


def test_format_context_numbers_and_labels_excerpts(make_hit):
    hits = [make_hit(text="Alpha body", source="a.md", section="A"),
            make_hit(text="Beta body", source="b.md", section="B")]
    context = generate.format_context(hits)
    assert '<excerpt id="1" source="a.md" section="A">' in context
    assert '<excerpt id="2" source="b.md" section="B">' in context
    assert context.index("Alpha body") < context.index("Beta body")
    assert context.count("</excerpt>") == 2


def test_build_messages_is_one_user_turn_ending_with_the_question(make_hit):
    messages = generate.build_messages("Why PostGIS?", [make_hit(text="Because spatial.")])
    assert len(messages) == 1 and messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert "Because spatial." in content
    assert content.rstrip().endswith("Question: Why PostGIS?")


def test_system_prompt_enforces_grounding_and_citations():
    prompt = generate.SYSTEM_PROMPT
    assert "ONLY the information in the provided excerpts" in prompt
    assert "[2]" in prompt
    assert "do not contain enough information" in prompt


class TestAnswer:
    def test_cost_uses_model_pricing(self, fakes, make_hit):
        answer = generate.Answer("x", [make_hit()], fakes.Response().usage, "claude-haiku-4-5")
        assert answer.cost == pytest.approx(config.price_of("claude-haiku-4-5", 100, 50))

    def test_no_usage_means_free_and_labelled(self, make_hit):
        answer = generate.Answer("x", [make_hit()], None, "claude-haiku-4-5")
        assert (answer.input_tokens, answer.output_tokens, answer.cost) == (0, 0, 0.0)
        assert answer.model == "(no API call)"

    def test_sources_block_maps_numbers_to_citations(self, make_hit):
        answer = generate.Answer("x", [make_hit(source="a.md", section="A"),
                                       make_hit(source="b.md", section="B")], None, "m")
        block = answer.sources_block()
        assert "[1] a.md § A" in block
        assert "[2] b.md § B" in block


@pytest.fixture
def hits_with(monkeypatch, make_hit):
    """Make retrieval return hits carrying the given rerank scores."""
    def install(*scores):
        hits = [make_hit(text=f"passage {i}", rerank_score=s) for i, s in enumerate(scores)]
        monkeypatch.setattr(generate, "retrieve", lambda question, top_k=None: hits)
        return hits
    return install


class TestAnswerQuestion:
    def test_sends_the_expected_request(self, hits_with, fakes):
        hits_with(3.0, 1.0)
        client = fakes.Client(fakes.Response("You chose it [1]."))
        answer = generate.answer_question("Why?", client=client)
        call = client.messages.calls[0]
        assert call["model"] == config.GENERATION_MODEL
        assert call["system"] == generate.SYSTEM_PROMPT
        assert call["max_tokens"] == config.MAX_OUTPUT_TOKENS
        assert "passage 0" in call["messages"][0]["content"]
        assert answer.text == "You chose it [1]."

    def test_only_text_blocks_become_the_answer(self, hits_with, fakes):
        hits_with(3.0)
        blocks = [fakes.Block("", type="thinking"), fakes.Block("Part one. "), fakes.Block("Part two.")]
        client = fakes.Client(fakes.Response(blocks=blocks))
        assert generate.answer_question("Why?", client=client).text == "Part one. Part two."

    def test_free_refusal_is_off_by_default(self, hits_with, fakes):
        """
        Measured, not cautious: the integration benchmark found answerable
        questions scoring below the original -7.0 cutoff. A false refusal hides
        real experience; a false send costs about $0.001. So by default every
        question reaches the model, whose prompt handles the refusal.
        """
        assert config.RERANK_MIN_SCORE is None
        hits_with(-11.0, -11.5)
        client = fakes.Client(fakes.Response("Your documents don't cover this."))
        generate.answer_question("q", client=client)
        assert len(client.messages.calls) == 1

    def test_when_enabled_low_scores_answer_without_calling_the_api(self, hits_with, monkeypatch):
        monkeypatch.setattr(config, "RERANK_MIN_SCORE", -7.0)
        hits_with(-9.5, -10.2, -11.0)
        # No client passed: building a real one would trip the conftest blocker,
        # so this passing proves no client was ever constructed.
        answer = generate.answer_question("What is my Kubernetes experience?")
        assert "don't appear to cover" in answer.text
        assert answer.cost == 0.0

    def test_when_enabled_one_confident_hit_is_enough_to_call_the_api(self, hits_with, fakes,
                                                                     monkeypatch):
        monkeypatch.setattr(config, "RERANK_MIN_SCORE", -7.0)
        hits_with(-10.0, -6.5)
        client = fakes.Client(fakes.Response("ok"))
        generate.answer_question("q", client=client)
        assert len(client.messages.calls) == 1

    def test_disabled_threshold_always_calls_the_api(self, hits_with, fakes, monkeypatch):
        monkeypatch.setattr(config, "RERANK_MIN_SCORE", None)
        hits_with(-11.0)
        client = fakes.Client(fakes.Response("ok"))
        generate.answer_question("q", client=client)
        assert len(client.messages.calls) == 1

    def test_without_rerank_scores_calls_the_api(self, hits_with, fakes):
        hits_with(None, None)
        client = fakes.Client(fakes.Response("ok"))
        generate.answer_question("q", client=client)
        assert len(client.messages.calls) == 1

    @pytest.mark.parametrize("cls, status, headers, message", [
        (anthropic.AuthenticationError, 401, None, "rejected the API key"),
        (anthropic.RateLimitError, 429, {"retry-after": "30"}, "Try again in 30s"),
        (anthropic.InternalServerError, 500, None, r"server error \(500\)"),
        (anthropic.BadRequestError, 400, None, "API error 400"),
        (anthropic.APIConnectionError, 0, None, "Could not reach"),
    ])
    def test_api_errors_become_readable_messages(self, hits_with, fakes, api_error,
                                                 cls, status, headers, message):
        hits_with(5.0)
        client = fakes.Client(api_error(cls, status, headers))
        with pytest.raises(RuntimeError, match=message):
            generate.answer_question("q", client=client)


def test_get_client_requires_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        generate.get_client()
