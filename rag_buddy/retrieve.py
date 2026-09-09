"""
Step 4 — Retrieval.

Given a question, find the handful of chunks most likely to contain the answer.

THE WHOLE IDEA IN ONE PARAGRAPH
The question and the chunks are embedded by the SAME model into the SAME
384-dimensional space. A question about database choices lands near passages
about database choices, because that is what the model was trained to do. So
"find relevant text" becomes "find the nearest points", which is a geometry
problem a vector database solves in milliseconds.

This step is still completely free and completely local. Nothing here calls an
API — retrieval is just arithmetic on vectors you already computed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from . import config
from .store import embed_texts, get_collection


@dataclass
class RetrievedChunk:
    """One search hit: the text, where it came from, and how close it was."""

    text: str
    source: str
    section: str
    citation: str
    distance: float
    rank: int
    rerank_score: float | None = None

    @property
    def similarity(self) -> float:
        """
        Cosine similarity, converted from the distance Chroma reports.

        Chroma returns cosine DISTANCE = 1 - cosine_similarity, so similarity
        is simply 1 - distance. Because our vectors are normalised, that lands
        in the range -1 (opposite) through 0 (unrelated) to 1 (identical).

        TREAT THIS AS A RANKING SIGNAL, NOT A CONFIDENCE SCORE. Feeding a
        chunk's own text back as a query scores about 0.88, not 1.0 — the
        stored vector includes the heading breadcrumb the query lacks. Genuine
        good matches on real questions typically sit around 0.4–0.7. Any fixed
        "good enough" cutoff you pick from intuition will be wrong; see
        min_similarity below.
        """
        return 1.0 - self.distance


_reranker = None


def get_reranker():
    """
    Load the cross-encoder, once per process.

    Imported lazily for the same reason as the embedding model: it costs a
    couple of seconds and a ~80 MB download the first time, and only the
    retrieval path needs it.
    """
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(config.RERANK_MODEL)
    return _reranker


def _apply_diversity_cap(hits: list[RetrievedChunk], max_per_source: int,
                         limit: int) -> list[RetrievedChunk]:
    """
    Take the best `limit` hits, allowing at most `max_per_source` per document.

    Hits must already be in rank order. Anything skipped by the cap is held
    back and used to fill remaining slots if we run out of diverse candidates —
    returning fewer chunks than asked for would be a worse outcome than a
    slightly lopsided context.
    """
    chosen: list[RetrievedChunk] = []
    overflow: list[RetrievedChunk] = []
    seen: dict[str, int] = {}

    for hit in hits:
        if len(chosen) >= limit:
            break
        if seen.get(hit.source, 0) < max_per_source:
            seen[hit.source] = seen.get(hit.source, 0) + 1
            chosen.append(hit)
        else:
            overflow.append(hit)

    for hit in overflow:
        if len(chosen) >= limit:
            break
        chosen.append(hit)

    for i, hit in enumerate(chosen, start=1):
        hit.rank = i
    return chosen


def retrieve(
    query: str,
    top_k: int | None = None,
    min_similarity: float | None = None,
    rerank: bool | None = None,
    max_per_source: int | None = None,
    source: str | None = None,
) -> list[RetrievedChunk]:
    """
    Embed the question and return the top-k nearest chunks.

    WHY top_k IS SMALL (4 by default)
    This is the single biggest cost lever in the project. Every retrieved chunk
    is sent to Claude as input tokens on every question you ask. At ~160 words
    (~210 tokens) per chunk, k=4 costs roughly 840 tokens of context per
    question; k=20 costs 4,200. Quality does not improve in step with that —
    past a handful of chunks you are mostly adding near-misses, and a model
    handed twenty passages of which three are relevant answers less precisely
    than one handed the three. Retrieve few, retrieve well.

    WHY min_similarity IS OFF BY DEFAULT
    A threshold that drops weak matches sounds obviously good and is a common
    way to break a RAG system. Absolute similarity values depend on the model,
    the phrasing of the question, and how long your chunks are — the scores
    cluster in a narrow band that shifts from corpus to corpus. Set it to 0.7
    on intuition and the tool silently answers "I don't know" to everything.
    It is exposed here so you can calibrate it against YOUR documents once you
    can see real scores (run this module from the command line to do that),
    not so you can guess at it now.
    """
    top_k = top_k or config.TOP_K
    rerank = config.RERANK_ENABLED if rerank is None else rerank
    max_per_source = max_per_source or config.MAX_CHUNKS_PER_SOURCE
    collection = get_collection()

    if collection.count() == 0:
        raise RuntimeError(
            "The vector store is empty. Run `python -m rag_buddy.store` first "
            "to ingest and embed the files in documents/."
        )

    # The query MUST be embedded by the same model that embedded the chunks.
    # Different models produce incompatible coordinate systems, and mixing them
    # fails silently: no error, just permanently nonsensical results. This is
    # why the model name lives in config.py and is never passed in by a caller.
    query_vector = embed_texts([query], show_progress=False)

    # With reranking on, cast a wider net first: vector search only has to get
    # the right chunk into the candidate pool, not to the top of it.
    wanted = config.RERANK_CANDIDATES if rerank else top_k
    results = collection.query(
        query_embeddings=query_vector,
        # Never ask for more rows than exist, or Chroma pads the result.
        n_results=min(wanted, collection.count()),
        include=["documents", "metadatas", "distances"],
        **({"where": {"source": source}} if source else {}),
    )

    # Chroma returns a list-per-query; we only ever send one query.
    hits: list[RetrievedChunk] = []
    for rank, (doc, meta, dist) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0]),
        start=1,
    ):
        chunk = RetrievedChunk(
            text=doc,
            source=meta["source"],
            section=meta["section"],
            citation=meta["citation"],
            distance=dist,
            rank=rank,
        )
        if min_similarity is not None and chunk.similarity < min_similarity:
            continue
        hits.append(chunk)

    if rerank and len(hits) > 1:
        # The cross-encoder reads (question, passage) together and scores how
        # well the passage answers the question — a different and much better
        # judgement than the cosine distance between two independent vectors.
        scores = get_reranker().predict([(query, hit.text) for hit in hits])
        for hit, score in zip(hits, scores):
            hit.rerank_score = float(score)
        hits.sort(key=lambda h: h.rerank_score, reverse=True)

    return _apply_diversity_cap(hits, max_per_source, top_k)


def estimate_context_tokens(hits: list[RetrievedChunk]) -> int:
    """
    Rough input-token count for the retrieved context.

    Used by step 5 to show what a question costs before it is asked. Same
    words x 1.3 heuristic as chunking — good enough for a cost estimate, and
    free, unlike asking the API to count for us.
    """
    words = sum(len(h.text.split()) for h in hits)
    return int(words * config.WORDS_TO_TOKENS_RATIO)


# ---------------------------------------------------------------------------
# `python -m rag_buddy.retrieve "your question"`
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Inspect what retrieval returns, without generating an answer.

    Worth using whenever an answer disappoints you. It separates the two
    failure modes: if the right passage is not in this list, the problem is
    retrieval or your chunking, and no amount of prompting will fix it. If the
    right passage IS here but the answer was still poor, the problem is in
    step 5.
    """
    if len(sys.argv) < 2:
        print('Usage: python -m rag_buddy.retrieve "your question here"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    hits = retrieve(query)

    print(f'\nQuery: "{query}"')
    print(f"Top {len(hits)} of {get_collection().count()} chunks\n")

    for hit in hits:
        score = (f"rerank {hit.rerank_score:+.2f}, " if hit.rerank_score is not None else "")
        print(f"  [{hit.rank}] {score}similarity {hit.similarity:.3f}  —  {hit.citation}")
        text = " ".join(hit.text.split())
        print(f"      {text[:300]}{'…' if len(text) > 300 else ''}\n")

    tokens = estimate_context_tokens(hits)
    print(f"Context size: ~{tokens} input tokens.")
    print(f"At Claude Haiku input pricing ($1.00 per 1M tokens), the retrieved "
          f"context for this question would cost about ${tokens * 1.0 / 1_000_000:.6f}.")
    print("Retrieval itself: $0.00 — it never leaves your machine.")


if __name__ == "__main__":
    main()
