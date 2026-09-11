"""Configuration invariants and the pricing helper."""

import pytest

from rag_buddy import config


def test_price_of_known_model():
    # Haiku: $1 / 1M input, $5 / 1M output
    assert config.price_of("claude-haiku-4-5", 1_000_000, 0) == pytest.approx(1.00)
    assert config.price_of("claude-haiku-4-5", 0, 1_000_000) == pytest.approx(5.00)
    assert config.price_of("claude-haiku-4-5", 855, 86) == pytest.approx(0.001285)


def test_price_of_unknown_model_is_zero_not_an_error():
    assert config.price_of("some-future-model", 1000, 1000) == 0.0


def test_both_configured_models_have_prices():
    # A missing price would silently report every request as free.
    assert config.GENERATION_MODEL in config.MODEL_PRICING
    assert config.EVALUATION_MODEL in config.MODEL_PRICING


def test_get_api_key_returns_key_when_set(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abc")
    assert config.get_api_key() == "sk-ant-abc"


def test_get_api_key_explains_setup_when_missing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="cp .env.example .env"):
        config.get_api_key()


def test_chunk_budget_leaves_room_for_heading_inside_model_window():
    # The breadcrumb is embedded alongside the body; together they must fit.
    assert config.CHUNK_TARGET_TOKENS + config.HEADING_TOKEN_RESERVE <= config.EMBEDDING_MAX_TOKENS


def test_min_chunk_is_below_target():
    assert 0 < config.CHUNK_MIN_TOKENS < config.CHUNK_TARGET_TOKENS


def test_distance_metric_is_cosine():
    assert config.DISTANCE_METRIC == "cosine"


def test_reranker_casts_wider_net_than_final_context():
    assert config.RERANK_CANDIDATES >= config.TOP_K


def test_grading_has_more_room_than_answering():
    # Grading runs on a thinking model: the reasoning and the grade share one
    # budget. Sizing it like an answer produced empty grades.
    assert config.EVALUATION_MAX_TOKENS > config.MAX_OUTPUT_TOKENS


def test_evaluation_effort_is_a_valid_level():
    assert config.EVALUATION_EFFORT in {"low", "medium", "high"}
