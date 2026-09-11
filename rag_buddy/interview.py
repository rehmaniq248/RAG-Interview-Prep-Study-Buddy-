"""
Step 6 — Interview mode.

Two jobs, and they use the retrieved context in opposite directions.

  Q&A mode (step 5):  you ask -> the documents answer.
  Interview mode:     the documents ask -> you answer -> the documents grade you.

That inversion is the whole point. Reading your own write-up back to yourself
tells you nothing about whether you can SAY it under pressure. This makes you
produce the answer from memory first, then shows you exactly what your written
account contains that your spoken one missed.

WHY GRADING IS THE HARD PART
Asking a model to evaluate an answer invites the failure mode of a generous
grader. A model handed "your answer" and "the source material" will, left to
itself, find something encouraging to say about almost anything. A grader that
tells you a vague answer was fine is worse than no grader at all, because you
walk into the interview believing you covered it. The prompt below therefore
spends most of its length forbidding that.
"""

from __future__ import annotations

import random
import sys

import anthropic

from . import config
from .generate import extract_text, format_context, get_client
from .retrieve import RetrievedChunk
from .store import collection_stats, get_chunks_by_source

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

QUESTION_SYSTEM_PROMPT = """\
You are an experienced technical interviewer. You will be given excerpts from \
one of a candidate's own project write-ups. Generate interview questions drawn \
from that material.

Rules:
1. Every question must be answerable from the excerpts. Do not ask about \
technologies or events that are not mentioned there.
2. Ask what a real interviewer asks: open questions about decisions, \
tradeoffs, failures, and what they would do differently. "Walk me through why \
you chose X over Y" is a good question. "What database did you use?" is not — \
it is a trivia lookup with a one-word answer.
3. Favour the parts of the write-up with the most substance: a decision with a \
stated reason, a tradeoff with a stated cost, a mistake with a stated lesson.
4. Ask about ONE thing per question. No multi-part questions.
5. Phrase them in the second person, as if speaking to the candidate.

Output format: each question on its own line, prefixed with "Q: ". Nothing \
else — no numbering, no preamble, no commentary."""


GRADER_SYSTEM_PROMPT = """\
You evaluate a candidate's spoken interview answer against their own written \
project notes. The notes are the ground truth for what actually happened.

Be exacting. A generous grade is worse than useless here — it sends someone \
into an interview believing they covered something they did not. Do not open \
with praise. Do not soften findings.

Judge only against the excerpts. Three distinct cases, and keep them separate:
- CONTRADICTS: the answer states something the excerpts say otherwise. This is \
the most serious finding — flag it prominently, and quote what the notes say.
- MISSING: the excerpts contain something substantive the answer left out — \
especially a number, a named tradeoff, a rejected alternative, or a stated \
consequence. Interviewers probe exactly these.
- UNVERIFIABLE: the answer includes something plausible that simply is not in \
the notes. This is NOT an error — the candidate may know more than they wrote \
down. Say only that you cannot check it from the notes, and suggest they add \
it if it matters.

Two rules that constrain your criticism, because an over-harsh grader is as \
useless as a flattering one:
- Every MISSING item must quote actual text from the excerpts. If you cannot \
quote it, it is not missing — the notes do not contain it either, and the \
candidate cannot be faulted for omitting what they never wrote down. Never \
fault an answer for lacking a number, a comparison, or a detail that the \
excerpts do not themselves contain.
- UNVERIFIABLE material is not a fault and must never lower the verdict. A \
candidate who adds a true detail they forgot to write down has done nothing \
wrong.

Verdict scale — apply it literally:
- STRONG: covers the substantive points the excerpts contain, with no \
contradictions. Small omissions do not disqualify a STRONG.
- ADEQUATE: gets the main point but omits specifics an interviewer would \
probe — a number, a named tradeoff, a rejected alternative.
- WEAK: contradicts the notes, or misses most of their substance.

Respond in exactly this structure, with no other sections:

VERDICT: one of STRONG / ADEQUATE / WEAK, then one sentence saying why.

COVERED:
- what the answer got right, briefly. Omit this section entirely if nothing was.

MISSING:
- each substantive omission, quoting the specific detail from the notes.

CONTRADICTS:
- each conflict, quoting the notes. Write "None." if there are none.

SHARPEN THIS:
- one concrete sentence the candidate should have said, drawn from the notes.

Grade the substance, not the delivery. Do not comment on filler words, \
nervousness, or phrasing unless the meaning is genuinely unclear."""


# ---------------------------------------------------------------------------
# Turning stored rows back into excerpts
# ---------------------------------------------------------------------------


def _as_excerpts(rows: list[dict]) -> list[RetrievedChunk]:
    """
    Adapt stored Chroma rows to the same shape retrieval produces.

    Interview mode selects chunks by document rather than by search, so there
    is no distance to report — but reusing RetrievedChunk means format_context()
    from step 5 works unchanged for both paths.
    """
    return [
        RetrievedChunk(
            text=row["text"],
            source=row["metadata"]["source"],
            section=row["metadata"]["section"],
            citation=row["metadata"]["citation"],
            distance=0.0,
            rank=i,
        )
        for i, row in enumerate(rows, start=1)
    ]


def pick_source_excerpts(
    source: str | None = None, max_chunks: int | None = None
) -> tuple[str, list[RetrievedChunk]]:
    """
    Choose a document and a sample of its chunks to build questions from.

    Sampling rather than sending everything is a deliberate cost ceiling — see
    MAX_QUESTION_SOURCE_CHUNKS in config. Chunks are sampled but then restored
    to document order, so the model reads them as a coherent narrative rather
    than shuffled fragments.
    """
    max_chunks = max_chunks or config.MAX_QUESTION_SOURCE_CHUNKS
    stats = collection_stats()

    if not stats["count"]:
        raise RuntimeError(
            "The vector store is empty. Run `python -m rag_buddy.store` first."
        )

    source = source or random.choice(stats["sources"])
    if source not in stats["sources"]:
        raise RuntimeError(
            f"No document named {source!r}. Available: {', '.join(stats['sources'])}"
        )

    rows = get_chunks_by_source(source)
    if len(rows) > max_chunks:
        picked = random.sample(range(len(rows)), max_chunks)
        rows = [rows[i] for i in sorted(picked)]

    return source, _as_excerpts(rows)


# ---------------------------------------------------------------------------
# The two API calls
# ---------------------------------------------------------------------------


class Usage:
    """Running total of what a session has cost."""

    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = 0.0

    def add(self, response, model: str) -> None:
        self.calls += 1
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens
        self.cost += config.price_of(
            model, response.usage.input_tokens, response.usage.output_tokens
        )

    def summary(self) -> str:
        return (
            f"{self.calls} API call(s), {self.input_tokens} in / "
            f"{self.output_tokens} out — ${self.cost:.6f}"
        )


def _call(client, model, system, user, max_tokens, effort=None):
    """
    One message request, with the same typed error handling as step 5.

    `effort` controls how much a thinking model reasons before replying. It is
    only sent when given: Haiku 4.5 rejects the parameter outright, so question
    generation must not pass it.
    """
    extra = {"output_config": {"effort": effort}} if effort else {}
    try:
        return client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            **extra,
        )
    except anthropic.AuthenticationError:
        raise RuntimeError("Anthropic rejected the API key — check .env") from None
    except anthropic.RateLimitError as exc:
        wait = exc.response.headers.get("retry-after", "60")
        raise RuntimeError(f"Rate limited. Try again in {wait}s.") from None
    except anthropic.APIStatusError as exc:
        if exc.status_code >= 500:
            raise RuntimeError(f"Anthropic server error ({exc.status_code}).") from None
        raise RuntimeError(f"API error {exc.status_code}: {exc.message}") from None
    except anthropic.APIConnectionError:
        raise RuntimeError("Could not reach the Anthropic API.") from None


def generate_questions(
    excerpts: list[RetrievedChunk],
    n: int | None = None,
    client=None,
    usage: Usage | None = None,
) -> list[str]:
    """Ask the model for n interview questions grounded in these excerpts."""
    n = n or config.QUESTIONS_PER_ROUND
    client = client or get_client()

    response = _call(
        client,
        config.GENERATION_MODEL,
        QUESTION_SYSTEM_PROMPT,
        f"Here are excerpts from the candidate's write-up:\n\n"
        f"{format_context(excerpts)}\n\n"
        f"Generate exactly {n} interview questions.",
        max_tokens=config.MAX_OUTPUT_TOKENS,
    )
    if usage:
        usage.add(response, config.GENERATION_MODEL)

    text = extract_text(response, "question")
    # Tolerate the model formatting "Q:" lines inconsistently. Strip BEFORE
    # removing the prefix: models sometimes indent the list, and "  Q: Why?"
    # does not start with "Q:" until the leading whitespace is gone — which
    # left the prefix inside the question text.
    questions = [
        line.strip().removeprefix("Q:").strip()
        for line in text.splitlines()
        if line.strip().startswith("Q:")
    ]
    return questions[:n]


def grade_answer(
    question: str,
    answer: str,
    excerpts: list[RetrievedChunk],
    client=None,
    usage: Usage | None = None,
) -> str:
    """
    Grade one answer against the excerpts the question was built from.

    Note we grade against the SOURCE excerpts, not a fresh retrieval of the
    answer text. The question was constructed from this passage, so this
    passage is by definition the ground truth for it. Re-retrieving would cost
    an extra search and risk grading against a passage the question never came
    from — marking you down for omitting something you were never asked about.
    """
    client = client or get_client()
    model = config.EVALUATION_MODEL

    response = _call(
        client,
        model,
        GRADER_SYSTEM_PROMPT,
        f"The candidate's written notes:\n\n{format_context(excerpts)}\n\n"
        f"Question asked:\n{question}\n\n"
        f"The candidate's spoken answer:\n{answer}",
        # Grading needs room for the model's reasoning AND the grade; see the
        # measurement table beside EVALUATION_MAX_TOKENS in config.
        max_tokens=config.EVALUATION_MAX_TOKENS,
        effort=config.EVALUATION_EFFORT,
    )
    if usage:
        usage.add(response, model)

    return extract_text(response, "grade")


# ---------------------------------------------------------------------------
# The interactive session
# ---------------------------------------------------------------------------


def run_session(source: str | None = None, n_questions: int | None = None) -> Usage:
    """
    Generate questions, take your answers, grade each one.

    Returns the session's Usage so a caller (the CLI menu) can fold this into a
    running total across modes.
    """
    usage = Usage()
    client = get_client()

    source, excerpts = pick_source_excerpts(source)
    print(f"\nDrawing questions from: {source}")
    print(f"Using {len(excerpts)} excerpt(s) as ground truth.\n")
    print("Generating questions…")

    questions = generate_questions(excerpts, n_questions, client=client, usage=usage)
    if not questions:
        print("The model returned no usable questions. Try running it again.")
        return usage

    print(f"{len(questions)} question(s) ready. "
          f"Type your answer and press Enter. "
          f"Enter 'skip' to move on, 'quit' to stop.\n")

    graded = 0
    for i, question in enumerate(questions, start=1):
        print("=" * 78)
        print(f"Question {i} of {len(questions)}")
        print("=" * 78)
        print(f"\n{question}\n")

        try:
            answer = input("Your answer > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nStopping.")
            break

        if answer.lower() == "quit":
            break
        if answer.lower() == "skip" or not answer:
            print("Skipped.\n")
            continue

        print("\nGrading…\n")
        print(grade_answer(question, answer, excerpts, client=client, usage=usage))
        print()
        graded += 1

    print("=" * 78)
    print(f"Session over — {graded} answer(s) graded.")
    print(f"Cost: {usage.summary()}")
    if graded:
        print(f"Average per graded answer: ${usage.cost / graded:.6f}")
    return usage


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else None
    run_session(source=source)


if __name__ == "__main__":
    main()
