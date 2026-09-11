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
# Fallback only. The chunker uses the embedding model's REAL tokenizer when it
# can load it; this ratio is used only if that fails.
#
# It is a fallback rather than the default because it is wrong in exactly the
# case that hurts most. Prose runs ~1.25 word-pieces per word, but a file path
# like "data/sample_shopify_import.csv" is one word and about eight tokens. A
# 60-line repository tree measured 60 "words" — well inside a 160-word budget —
# while actually being 952 word-pieces, nearly four times the model's limit.
# Estimating token counts from word counts silently breaks on any structured
# text: paths, code, tables, URLs.
WORDS_TO_TOKENS_RATIO = 1.3

# The heading breadcrumb is embedded ALONGSIDE the body text, so it eats into
# the same 256-token window. Measured at 8–29 tokens on real write-ups; we
# reserve 40 so a document with long headings still fits. Forgetting this
# reserve is what pushed our first build to 245/256 — uncomfortably close to
# silent truncation.
HEADING_TOKEN_RESERVE = 40

# Body budget in TOKENS: 256 limit - 40 reserved for the heading breadcrumb,
# minus a small margin.
CHUNK_TARGET_TOKENS = 200

# Chunks shorter than this are weak retrieval units — a lone sentence rarely
# carries enough context to answer anything — so we pack neighbouring
# paragraphs together until we clear this floor.
CHUNK_MIN_TOKENS = 50

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
# made. So we measured it, on three answers to the same question:
#
#                     Haiku 4.5    Sonnet 5     correct
#   vague answer      WEAK         WEAK         both
#   wrong numbers     WEAK         WEAK         both
#   strong answer     ADEQUATE     STRONG       Sonnet
#
# Haiku compresses the top of the scale: it could not tell a good answer from a
# mediocre one, and invented faults to justify the gap — including penalising a
# true detail the notes happened not to contain, after being told explicitly not
# to. A grader that calls a strong answer "adequate" trains you to over-explain
# and stops being believed.
#
# So grading alone uses Sonnet 5. Q&A and question generation stay on Haiku.
# Measured cost: ~$0.010 per graded answer vs ~$0.003 on Haiku — about 3x, on
# the one call in the project where judgement is the whole product. A ten
# question session costs roughly $0.10.
#
# To revert, set this back to GENERATION_MODEL.
EVALUATION_MODEL = "claude-sonnet-5"

# Grading runs on a model that THINKS before it answers, and thinking tokens
# come out of the same max_tokens budget as the grade itself. At 1024 — fine
# for Haiku answers — Sonnet 5 spent the whole budget reasoning and returned no
# grade at all in 2 of 3 trials. The CLI printed nothing after "Grading…",
# which is indistinguishable from a hang, and the run was billed in full.
#
# Measured on one grading request (same prompt, same excerpts):
#
#   budget  effort   time   output   result
#   1024    high      11s     1024   EMPTY in 2 of 3 runs
#   4096    high      21s    ~2000   works, slow, ~$0.02
#   4096    medium     7s     ~630   works
#   2048    low        6s     ~500   works
#
# At both medium and low the grader still scored a vague answer WEAK, a
# factually wrong one WEAK and a strong one STRONG, so low keeps the
# discrimination that made Sonnet worth using while being faster and cheaper.
# Raise EVALUATION_EFFORT to "medium" if grades ever feel shallow.
EVALUATION_MAX_TOKENS = 2048
EVALUATION_EFFORT = "low"        # low | medium | high

# How many chunks we feed the model as context. Keeping this small is the main
# cost control in the whole project: each extra chunk is extra input tokens on
# every single question you ask.
TOP_K = 4

# --- Reranking --------------------------------------------------------------
# Vector search is fast but coarse: it compares two vectors that were computed
# independently, so it can only measure "are these about the same topic",
# never "does this passage answer this question". A cross-encoder reads the
# question and the passage TOGETHER and scores the pair directly. Far more
# accurate, far too slow to run over a whole corpus — which is exactly why it
# goes second: vector search cheaply narrows 89 chunks to 20, the cross-encoder
# carefully picks the best few from those.
#
# Runs locally on CPU. Free, like everything before the generation step.
RERANK_ENABLED = True
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# How many candidates vector search hands the reranker. Larger casts a wider
# net at some CPU cost; the API cost is unchanged either way, because only
# TOP_K chunks are ever sent to Claude.
RERANK_CANDIDATES = 20

# At most this many chunks from any one document in the final results.
#
# WHY: the same project now appears in up to three places — a resume bullet, a
# generated repo document, and a write-up. Without a cap, one project's
# near-duplicate chunks can take every slot, so a question that touches two
# projects only ever sees one. The cap forces the context to spread.
MAX_CHUNKS_PER_SOURCE = 2

# Optional: below this cross-encoder score, say the documents don't cover the
# question WITHOUT calling the API. OFF by default (None) — because of a
# measured failure, not caution.
#
# It was first calibrated at -7.0 on one real corpus, where the gap looked
# comfortable:
#
#   answerable    -4.9 … +6.8
#   unanswerable  -11.3 … -9.2
#
# The integration benchmark (tests/integration/test_retrieval_quality.py) then
# ran the same model over a different corpus, and -7.0 refused two questions
# the documents DID answer, scoring -7.6 and -9.25. Across the two corpora the
# ranges overlap outright — an answerable question scored -9.25 while an
# unanswerable one scored -9.2 — so no single value separates them in general.
#
# The two possible errors are not equal. Sending an unanswerable question to
# Claude costs about $0.0013, and the system prompt still makes it decline.
# Refusing an answerable one silently hides your real experience from you,
# which is the one thing this tool must never do. So it is off.
#
# To turn it on, calibrate against YOUR documents and choose a value well
# below the lowest score of any question you know they answer. -10.5 refused
# no answerable question on either corpus measured so far — but with only
# about 1.25 points of margin, so treat it as a starting point, not a
# guarantee.
RERANK_MIN_SCORE: float | None = None

# --- Interview mode (step 6) -----------------------------------------------
# How many questions to generate per round.
QUESTIONS_PER_ROUND = 5

# How many of a document's chunks to use as the raw material for those
# questions. This is a cost ceiling: a long resume could be fifty chunks, and
# sending all of them would make a single question-generation call cost more
# than a hundred ordinary questions. Six chunks gives the model enough material
# to ask something substantive without paying for the whole document.
MAX_QUESTION_SOURCE_CHUNKS = 6

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
