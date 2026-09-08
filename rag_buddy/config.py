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

# How many chunks we feed the model as context. Keeping this small is the main
# cost control in the whole project: each extra chunk is extra input tokens on
# every single question you ask.
TOP_K = 4

# Hard ceiling on the answer length, which caps the (more expensive) output
# token spend per question.
MAX_OUTPUT_TOKENS = 1024


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
