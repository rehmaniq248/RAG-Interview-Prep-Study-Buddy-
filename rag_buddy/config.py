"""
Central configuration for the study buddy.

WHY a config module: every later step (ingestion, embedding, retrieval,
generation) needs to agree on the same paths and the same embedding model.
If ingestion embedded with one model and retrieval queried with another, the
vectors would live in different "meaning spaces" and search results would be
garbage. Keeping these constants in one file makes that mismatch impossible.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load ANTHROPIC_API_KEY (and anything else) from a local .env file, if present.
# This keeps secrets out of the source code and out of git.
load_dotenv()

# --- Paths -----------------------------------------------------------------
# Anchored to this file's location, not the current working directory, so the
# tool behaves the same no matter which folder you run it from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where YOU drop your resume, project write-ups, and experience notes.
DOCUMENTS_DIR = PROJECT_ROOT / "documents"

# Where ChromaDB persists the vector store on disk. Safe to delete — you can
# always rebuild it by re-running ingestion.
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# File types we read. Plain text formats only: no PDF parsing, no encoding
# guesswork, no extra dependencies.
SUPPORTED_EXTENSIONS = {".txt", ".md"}

# --- Embedding model (LOCAL and FREE) --------------------------------------
# all-MiniLM-L6-v2 is a small, fast, well-benchmarked sentence embedding model.
# It maps a piece of text to a 384-dimensional vector. Downloaded once (~90 MB)
# and then runs on your CPU — no API calls, no per-token charges.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# The model was trained with a 256-word-piece input window; text longer than
# that gets silently truncated. This is the real reason our chunks must stay
# small — see the chunking step.
EMBEDDING_MAX_TOKENS = 256

# --- Chunking --------------------------------------------------------------
# We measure chunk size in WORDS, not tokens, because counting real tokens
# would mean loading the embedding model just to split text. The conversion is
# roughly 1 word ≈ 1.3 word-pieces for ordinary English prose, so we keep a
# deliberate safety margin below EMBEDDING_MAX_TOKENS: 180 words ≈ 234
# word-pieces. Step 3 verifies this against the real tokenizer and warns if
# any chunk would actually be truncated.
# Measured against the real tokenizer on sample prose: 1.25 word-pieces per
# word. We budget at 1.3 to stay conservative.
WORDS_TO_TOKENS_RATIO = 1.3

# The heading breadcrumb is embedded ALONGSIDE the body text, so it eats into
# the same 256-token window. Measured at 8–29 tokens on real write-ups; we
# reserve 40 so a document with long headings still fits. Forgetting this
# reserve is what pushed our first build to 245/256 — uncomfortably close to
# silent truncation.
HEADING_TOKEN_RESERVE = 40

# Body budget: (256 - 40 reserved) / 1.3 ≈ 166 words. Rounded down.
CHUNK_TARGET_WORDS = 160

# Chunks shorter than this are weak retrieval units — a lone sentence rarely
# carries enough context to answer anything — so we pack neighbouring
# paragraphs together until we clear this floor.
CHUNK_MIN_WORDS = 40

# Carry the last N sentences of each chunk into the start of the next one.
# This is "overlap": it stops an idea that straddles a chunk boundary from
# being lost by both chunks. Costs a little duplicated storage, which is free
# here because storage is local.
CHUNK_OVERLAP_SENTENCES = 1

# How many of the deepest heading levels to prepend to the embedded text.
# The full trail ("Example > Project: Trailhead — a route-planning app for day
# hikers > Architecture") is mostly redundant: the document title is already
# in the `source` metadata, and every extra heading word competes with the
# body for room in a 384-number vector, diluting what the chunk is "about".
# The deepest two levels carry the useful signal. The FULL trail is still kept
# for citations — this only trims what gets embedded.
EMBED_HEADING_LEVELS = 2

# --- Vector store ----------------------------------------------------------
COLLECTION_NAME = "interview_prep"

# Cosine similarity measures the ANGLE between two vectors, ignoring their
# length. That is what we want for text: a one-sentence chunk and a paragraph
# about the same topic should count as similar even though the paragraph's
# vector is "bigger". Chroma defaults to squared L2 distance, so we set this
# explicitly at collection-creation time.
DISTANCE_METRIC = "cosine"

# --- Generation (the ONLY paid step) ---------------------------------------
# Claude Haiku 4.5: the cheapest current Claude model.
# Pricing at time of writing: $1.00 per 1M input tokens, $5.00 per 1M output.
GENERATION_MODEL = "claude-haiku-4-5"

# Interview mode (step 6) grades your spoken/typed answer against what your
# write-up actually says. That is a far harder judgement call than answering a
# question from supplied text, and a weak grader is a generous grader — it
# will tell you an answer was fine when you left out the tradeoff you actually
# made. This knob exists so you can measure that rather than guess: leave it on
# Haiku, and if grading feels soft, point it at "claude-sonnet-5" ($2.00 per
# 1M input / $10.00 per 1M output) for evaluation only, while Q&A stays cheap.
EVALUATION_MODEL = GENERATION_MODEL

# How many chunks we feed the model as context. Keeping this small is the main
# cost control in the whole project: each extra chunk is extra input tokens on
# every single question you ask.
TOP_K = 4

# Hard ceiling on the answer length, which caps the (more expensive) output
# token spend per question.
MAX_OUTPUT_TOKENS = 1024

# Published per-million-token prices, used to show you the real cost of every
# question. Update these if Anthropic's pricing changes — they are display
# only and do not affect any request.
MODEL_PRICING = {
    # model id            (input $/1M, output $/1M)
    "claude-haiku-4-5":   (1.00, 5.00),
    "claude-sonnet-5":    (2.00, 10.00),
    "claude-opus-5":      (5.00, 25.00),
}


def price_of(model: str, input_tokens: int, output_tokens: int) -> float:
    """Dollar cost of one request. Returns 0.0 for a model we have no price for."""
    rates = MODEL_PRICING.get(model)
    if not rates:
        return 0.0
    in_rate, out_rate = rates
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


def get_api_key() -> str:
    """
    Return the Anthropic API key, with a friendly error if it is missing.

    Called only by the generation step — ingestion, embedding, and retrieval
    all work with no key at all.
    """
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set.\n"
            "  1. cp .env.example .env\n"
            "  2. paste your key from https://console.anthropic.com/settings/keys\n"
            "Ingesting documents and searching them works without a key — "
            "only answer generation needs one."
        )
    return key
