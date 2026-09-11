"""Real ChromaDB: building, rebuilding, and reading back the index."""

import pytest

from corpus_data import install_corpus
from rag_buddy import config, ingest, retrieve, store


@pytest.fixture
def built(fresh_workspace):
    install_corpus(config.DOCUMENTS_DIR)
    return store.build_index()


def test_every_chunk_is_stored_with_cosine_distance(built):
    collection = store.get_collection()
    assert collection.count() == built["chunks"] > 0
    assert collection.metadata["hnsw:space"] == "cosine"
    assert built["dimensions"] == 384
    assert built["truncation"].ok


def test_stored_documents_are_chunk_bodies_not_embedding_text(built):
    # The heading trail is embedded to help search, but storing it would mean
    # paying for it again as input tokens in every prompt.
    expected = {c.chunk_id: c.text for c in ingest.load_and_chunk_all()}
    rows = store.get_collection().get(include=["documents"])
    assert dict(zip(rows["ids"], rows["documents"])) == expected


def test_rebuilding_does_not_duplicate_rows(built):
    store.build_index()
    assert store.get_collection().count() == built["chunks"]


def test_rebuild_while_a_client_is_already_open(built):
    """
    Regression: the CLI's status screen opened a Chroma client, then ingest
    deleted the database directory underneath it. Chroma caches clients by
    path, so the next write hit a deleted SQLite file: "attempt to write a
    readonly database". Rebuild now drops the collection through the API.
    """
    assert store.collection_stats()["count"] > 0      # opens and caches a client
    store.build_index()
    assert store.collection_stats()["count"] == built["chunks"]


def test_rebuild_drops_chunks_from_deleted_documents(built):
    assert "resume.md" in store.collection_stats()["sources"]
    (config.DOCUMENTS_DIR / "resume.md").unlink()
    store.build_index()
    assert "resume.md" not in store.collection_stats()["sources"]


def test_collection_stats_lists_every_source(built):
    assert store.collection_stats()["sources"] == [
        "example_project_writeup.md", "repo_ledger-sync.md", "resume.md"]


def test_chunks_by_source_come_back_in_document_order(built):
    rows = store.get_chunks_by_source("resume.md")
    indexes = [int(row["id"].rsplit("::", 1)[1]) for row in rows]
    assert indexes == list(range(len(rows)))


def test_querying_with_a_chunks_own_text_finds_it_with_near_perfect_similarity(built):
    chunk = next(c for c in ingest.load_and_chunk_all() if "Airflow" in c.text)
    hits = retrieve.retrieve(chunk.embedding_text(), rerank=False, top_k=1)
    assert hits[0].text == chunk.text
    assert hits[0].similarity > 0.99


def test_renamed_pdf_is_skipped_and_reported(fresh_workspace):
    install_corpus(config.DOCUMENTS_DIR)
    (config.DOCUMENTS_DIR / "old_resume.md").write_bytes(b"%PDF-1.7\nbinary junk")
    stats = store.build_index()
    assert [name for name, _ in stats["skipped"]] == ["old_resume.md"]
    assert "old_resume.md" not in store.collection_stats()["sources"]


def test_unfinished_draft_moved_into_documents_is_flagged(fresh_workspace):
    install_corpus(config.DOCUMENTS_DIR)
    (config.DOCUMENTS_DIR / "project_half.md").write_text(
        "# Half\n\nSome real content about the project.\n\n> **TODO:** why?\n")
    assert store.build_index()["unfinished"] == {"project_half.md": 1}


def test_empty_documents_folder_explains_itself(fresh_workspace):
    with pytest.raises(RuntimeError, match="No chunks produced"):
        store.build_index()
