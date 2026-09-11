"""
The real local models, and the pieces that must agree with them.

These catch what no fake can: the chunker loading a different tokenizer from
the embedding model, a model whose window no longer matches config, or a
dependency upgrade that quietly changes how text is embedded.
"""

import numpy as np

from corpus_data import LEDGER_FACTS
from rag_buddy import config, github, ingest, retrieve, store

SAMPLES = [
    "Rebuilt the nightly ETL job in Apache Airflow.",
    "data/sample_shopify_import.csv",
    "ledger_sync/matching/fuzzy_payee_03.py",
    "Café résumé — 0.87 AUC at 6.2 µg/m³ 🚀",
    "",
]


def test_chunker_counts_tokens_exactly_like_the_embedding_model():
    """
    The chunker loads its tokenizer separately from the embedding model, so
    chunking does not need the whole model. If the two ever disagree, every
    token budget in the project is quietly wrong.
    """
    tokenizer = store.get_embedder().tokenizer
    for text in SAMPLES:
        assert ingest.count_tokens(text) == len(tokenizer(text)["input_ids"]), repr(text)


def test_chunker_uses_the_real_tokenizer_not_the_word_count_fallback():
    ingest.count_tokens("warm up")
    assert ingest._tokenizer is not None
    assert not ingest._tokenizer_failed


def test_configured_window_matches_the_model():
    assert store.get_embedder().max_seq_length == config.EMBEDDING_MAX_TOKENS


def test_file_paths_really_are_token_heavy():
    # The premise of the file-tree truncation bug, against the real tokenizer.
    path = "data/sample_shopify_import.csv"
    assert len(path.split()) == 1
    assert ingest.count_tokens(path) >= 8


def test_embeddings_have_the_model_dimension_and_are_normalised():
    model = store.get_embedder()
    vectors = np.array(store.embed_texts(["one", "a longer sentence about databases"],
                                         show_progress=False))
    assert vectors.shape == (2, model.get_embedding_dimension())
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)


def test_embedding_is_deterministic():
    first = store.embed_texts(["the same text"], show_progress=False)[0]
    second = store.embed_texts(["the same text"], show_progress=False)[0]
    assert np.allclose(first, second, atol=1e-6)


def test_paraphrase_lands_closer_than_an_unrelated_sentence():
    a, b, c = np.array(store.embed_texts([
        "I optimised a slow database query",
        "I sped up our Postgres reads",
        "I baked sourdough bread at the weekend",
    ], show_progress=False))
    assert a @ b > a @ c


def test_reranker_prefers_the_passage_that_answers():
    question = "Why did you choose PostGIS?"
    scores = retrieve.get_reranker().predict([
        (question, "I chose PostGIS because the core query was spatial and needed an index."),
        (question, "The team held a retrospective every second Friday."),
    ])
    assert scores[0] > scores[1]


def test_generated_repo_document_fits_the_model_window(tmp_path):
    """The file-tree truncation regression, end to end with the real tokenizer."""
    facts = {**LEDGER_FACTS,
             "tree": [f"src/module_{i:03d}/component_file_{i:03d}.py" for i in range(150)]}
    path = tmp_path / "repo_big.md"
    path.write_text(github.facts_to_markdown(facts), encoding="utf-8")

    chunks = ingest.chunk_document(path)
    assert all(c.estimated_tokens() <= config.EMBEDDING_MAX_TOKENS for c in chunks), \
        max(c.estimated_tokens() for c in chunks)
    assert store.check_truncation(chunks).ok
