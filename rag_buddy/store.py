"""
Step 3 — Local embedding and vector storage.

Turns each chunk into a vector with a model running on your own machine, and
stores those vectors in ChromaDB on disk. No API, no network after the first
model download, no cost.

WHAT IS AN EMBEDDING?
A function from text to a fixed-length list of numbers — here, 384 floats.
The model is trained so that texts meaning similar things land near each other
in that 384-dimensional space, regardless of the words used. "I optimised a
slow database query" and "I sped up our Postgres reads" share almost no
vocabulary but land close together; "I optimised a slow database query" and "I
optimised our onboarding flow" share more vocabulary but land further apart.

That is the entire trick behind semantic search, and it is why this beats
keyword matching for interview prep: you will ask "tell me about a time I
handled scale" and your write-up will say "cut median page load from 2.1s to
340ms" — no shared keywords, but close in meaning-space.

WHY STORE THE TEXT TOO?
Embedding is one-way; you cannot turn 384 floats back into prose. So each row
holds the vector (for searching), the original text (to feed the model in step
5), and metadata (to cite the source). The vector finds it, the text answers
with it.

WHY PERSIST TO DISK?
Embedding is the slow part of this pipeline. Doing it once and saving the
result means asking questions later is instant.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass

import chromadb

from . import config
from .ingest import Chunk, load_and_chunk_all

# The model is heavy to construct (~90 MB of weights) but reusable, so we load
# it once per process and hold it here rather than rebuilding it per call.
_embedder = None


def get_embedder():
    """
    Load the local sentence-transformers model, downloading it on first use.

    The import is deliberately inside the function: importing
    sentence-transformers pulls in torch, which takes a couple of seconds. That
    cost belongs to the steps that actually embed, not to every `import config`
    anywhere in the project.
    """
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        print(f"Loading embedding model '{config.EMBEDDING_MODEL_NAME}' (local, free)…")
        print("  First run downloads ~90 MB; afterwards it loads from cache and works offline.")
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _embedder


def embed_texts(texts: list[str], show_progress: bool = True) -> list[list[float]]:
    """
    Embed a batch of texts locally.

    normalize_embeddings=True scales every vector to length 1. Cosine
    similarity only cares about the ANGLE between vectors, so pre-normalising
    makes the comparison a plain dot product — slightly faster, and it keeps
    the numbers in a predictable range. It also means the distances Chroma
    reports land in a clean 0–2 range, which matters in step 4 when we turn
    distance into a human-readable relevance score.
    """
    model = get_embedder()
    vectors = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=show_progress and len(texts) > 32,
    )
    return [v.tolist() for v in vectors]


# ---------------------------------------------------------------------------
# Truncation check
# ---------------------------------------------------------------------------


@dataclass
class TruncationReport:
    checked: int
    limit: int
    longest: int
    offenders: list[tuple[str, int]]

    @property
    def ok(self) -> bool:
        return not self.offenders


def check_truncation(chunks: list[Chunk]) -> TruncationReport:
    """
    Verify our word-count estimate against the model's REAL tokenizer.

    Step 2 sized chunks with a heuristic (1 word ≈ 1.3 word-pieces) to avoid
    loading the model just to split text. Here the model is loaded anyway, so
    we can check the estimate for real.

    This matters because truncation is silent. Hand this model 400 word-pieces
    and it embeds the first 256 and discards the rest — no error, no warning.
    The vector then represents only part of the chunk, while the full text
    still gets sent to Claude later, so a chunk can be un-findable for content
    it demonstrably contains. That is a miserable bug to diagnose from the
    outside, so we check for it at build time instead.
    """
    model = get_embedder()
    tokenizer = model.tokenizer
    limit = model.max_seq_length

    longest = 0
    offenders: list[tuple[str, int]] = []
    for chunk in chunks:
        # The text we actually embed includes the heading breadcrumb.
        n = len(tokenizer(chunk.embedding_text())["input_ids"])
        longest = max(longest, n)
        if n > limit:
            offenders.append((chunk.chunk_id, n))

    return TruncationReport(
        checked=len(chunks), limit=limit, longest=longest, offenders=offenders
    )


# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------


def get_client() -> chromadb.ClientAPI:
    """A client that persists to a folder on disk. No server, no Docker."""
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def get_collection(client: chromadb.ClientAPI | None = None):
    """
    Fetch (or create) the collection that holds our chunks.

    Two arguments here are load-bearing:

    metadata={"hnsw:space": "cosine"}
        Chroma defaults to squared L2 (straight-line) distance. We want cosine,
        which compares direction and ignores magnitude — the right choice for
        text, where a one-line note and a full paragraph on the same topic
        should count as similar. This can only be set when the collection is
        CREATED; changing it later means rebuilding.

    embedding_function=None
        Chroma will happily download and run its own embedding model if you
        let it. We pass None and hand it vectors we computed ourselves, so
        there is exactly one embedding model in this project and no surprise
        second download.
    """
    client = client or get_client()
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": config.DISTANCE_METRIC},
        embedding_function=None,
    )


def build_index(rebuild: bool = True) -> dict:
    """
    Chunk every document, embed the chunks, and write them to the store.

    Default is a full rebuild: drop the collection and recreate it. Embedding
    a few hundred chunks locally takes seconds, and a clean rebuild guarantees
    no stale rows survive from text you have since edited or deleted — which
    is exactly the kind of ghost that makes a RAG tool quietly cite something
    you rewrote a week ago.
    """
    chunks = load_and_chunk_all()
    if not chunks:
        raise RuntimeError(
            f"No chunks produced — is {config.DOCUMENTS_DIR} empty?\n"
            "Drop some .txt or .md files in there and try again."
        )

    report = check_truncation(chunks)

    if rebuild and config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)

    collection = get_collection()

    started = time.perf_counter()
    vectors = embed_texts([c.embedding_text() for c in chunks])
    embed_seconds = time.perf_counter() - started

    # Chroma writes in batches; a few hundred chunks fit comfortably in one.
    collection.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=vectors,
        # `documents` is the text returned at query time and handed to Claude.
        # Note this is chunk.text, NOT embedding_text() — the heading trail
        # helps the SEARCH, but it is already captured in metadata, so there is
        # no reason to pay for it again as input tokens in every prompt.
        documents=[c.text for c in chunks],
        metadatas=[
            {
                "source": c.source,
                "section": c.section,
                "citation": c.citation,
                "word_count": c.word_count,
            }
            for c in chunks
        ],
    )

    return {
        "chunks": len(chunks),
        "documents": len({c.source for c in chunks}),
        "dimensions": len(vectors[0]),
        "embed_seconds": embed_seconds,
        "truncation": report,
    }


def collection_stats() -> dict:
    """Summarise what is currently in the store, for the CLI in step 7."""
    try:
        collection = get_collection()
    except Exception:
        return {"exists": False, "count": 0, "sources": []}

    count = collection.count()
    sources: list[str] = []
    if count:
        rows = collection.get(include=["metadatas"])
        sources = sorted({m["source"] for m in rows["metadatas"]})
    return {"exists": True, "count": count, "sources": sources}


# ---------------------------------------------------------------------------
# `python -m rag_buddy.store`
# ---------------------------------------------------------------------------


def main() -> None:
    print("Building the vector index — everything here runs locally.\n")
    stats = build_index(rebuild=True)
    report: TruncationReport = stats["truncation"]

    print(f"\nEmbedded {stats['chunks']} chunks from {stats['documents']} document(s)")
    print(f"  vector size     : {stats['dimensions']} dimensions")
    print(f"  time            : {stats['embed_seconds']:.1f}s "
          f"({stats['chunks'] / max(stats['embed_seconds'], 1e-6):.0f} chunks/sec)")
    print(f"  stored at       : {config.CHROMA_DIR}")
    print(f"  distance metric : {config.DISTANCE_METRIC}")

    print(f"\nTruncation check against the real tokenizer "
          f"(limit {report.limit} word-pieces):")
    if report.ok:
        print(f"  OK — longest chunk is {report.longest} word-pieces, "
              f"{report.limit - report.longest} to spare.")
    else:
        print(f"  ⚠ {len(report.offenders)} chunk(s) exceed the limit and will be "
              f"silently truncated when embedded:")
        for chunk_id, n in report.offenders[:10]:
            print(f"      {chunk_id}: {n} word-pieces")
        print(f"  Lower CHUNK_TARGET_WORDS in config.py and re-run.")

    print("\nCost of this step: $0.00 — no API calls.")


if __name__ == "__main__":
    main()
