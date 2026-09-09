"""
Step 5 — Grounded answer generation.

The first and only step that costs money. Everything before this ran on your
machine for free; here we send the question plus the retrieved chunks to
Claude Haiku and ask for an answer built strictly from that context.

WHAT "RETRIEVAL-AUGMENTED" ACTUALLY MEANS
The model is not being asked what it knows. It is being handed a few passages
and asked to answer *from them*, the way you would answer an open-book exam
question by pointing at the page. Claude has never seen your resume and has no
opinion about your career; it is reading four paragraphs you wrote and
summarising them. That is the entire safety property of this design, and it is
why the answers can be trusted for interview prep — the tool cannot credit you
with a project you never did, because the project is not in the context.

WHERE GROUNDING IS ENFORCED
Not in retrieval. Retrieval has no notion of relevance thresholds — ask it
about Kubernetes when your documents only cover a hiking app and it will
cheerfully return the four nearest hiking passages. It always returns
something. So the "I don't know" behaviour has to be built HERE, in the system
prompt, and it is the most important text in this file.
"""

from __future__ import annotations

import sys

import anthropic

from . import config
from .retrieve import RetrievedChunk, retrieve

# ---------------------------------------------------------------------------
# The system prompt
# ---------------------------------------------------------------------------

# Design notes, because this text is doing the real work:
#
# * "Only the context" is stated positively AND negatively. Models comply more
#   reliably when told what to do when the answer is absent, not merely
#   forbidden from inventing.
# * Sources are numbered [1]…[n] rather than cited by filename. Filenames and
#   heading trails are long; repeating them inside the answer would spend
#   output tokens on bookkeeping. The number maps back to a citation we print
#   ourselves, at no token cost.
# * The prompt explicitly says some passages may be irrelevant, because with a
#   fixed top-k they frequently are. Without this, a model tries to find a use
#   for every passage it is given and pads the answer with near-misses.
# * Second person throughout: this is the user's own material, and answers
#   read better as "you built X" than "the candidate built X".
SYSTEM_PROMPT = """\
You help someone prepare for technical interviews using their own resume and \
project write-ups. You will be given numbered excerpts from those documents \
and a question.

Rules:
1. Answer using ONLY the information in the provided excerpts. Do not use \
outside knowledge, and do not infer details that are not written down.
2. Cite the excerpt number in square brackets after each claim, like [2]. If a \
sentence draws on two excerpts, cite both, like [1][3].
3. If the excerpts do not contain enough information to answer, say so plainly \
and state what is missing. Do not guess, and do not pad the answer with \
loosely related material. It is genuinely useful to hear "your documents \
don't cover this" — it tells the person what to go write down.
4. Some excerpts may be irrelevant to the question. Ignore them rather than \
finding a use for them.
5. Answer in the second person ("you built…", "you chose…"), since this is the \
person's own experience.
6. Be specific and concise. Prefer the concrete details from the excerpts — \
numbers, tradeoffs, names — over general summary."""


def format_context(hits: list[RetrievedChunk]) -> str:
    """
    Turn retrieved chunks into the numbered context block sent to the model.

    Each excerpt is delimited and labelled with its source. Clear delimiters
    matter: they keep the model from confusing your document text with the
    instructions around it.
    """
    parts = []
    for i, hit in enumerate(hits, start=1):
        parts.append(
            f"<excerpt id=\"{i}\" source=\"{hit.source}\" section=\"{hit.section}\">\n"
            f"{hit.text}\n"
            f"</excerpt>"
        )
    return "\n\n".join(parts)


def build_messages(question: str, hits: list[RetrievedChunk]) -> list[dict]:
    """Assemble the single user turn: the excerpts, then the question."""
    return [
        {
            "role": "user",
            "content": (
                f"Here are the relevant excerpts from your documents:\n\n"
                f"{format_context(hits)}\n\n"
                f"Question: {question}"
            ),
        }
    ]


# ---------------------------------------------------------------------------
# Calling the API
# ---------------------------------------------------------------------------


class Answer:
    """An answer plus what it cost and what it was based on."""

    def __init__(self, text: str, hits: list[RetrievedChunk], usage, model: str):
        self.text = text
        self.hits = hits
        self.input_tokens = usage.input_tokens
        self.output_tokens = usage.output_tokens
        self.model = model

    @property
    def cost(self) -> float:
        return config.price_of(self.model, self.input_tokens, self.output_tokens)

    def sources_block(self) -> str:
        """The [n] → document mapping, printed locally at zero token cost."""
        lines = ["Sources:"]
        for i, hit in enumerate(self.hits, start=1):
            lines.append(f"  [{i}] {hit.citation}  (similarity {hit.similarity:.3f})")
        return "\n".join(lines)


def get_client() -> anthropic.Anthropic:
    """
    Build the API client, failing with a useful message if no key is set.

    config.get_api_key() raises with setup instructions rather than letting the
    SDK fail later with a less obvious authentication error.
    """
    return anthropic.Anthropic(api_key=config.get_api_key())


def answer_question(
    question: str,
    top_k: int | None = None,
    model: str | None = None,
    client: anthropic.Anthropic | None = None,
) -> Answer:
    """
    Retrieve, then generate. The complete RAG loop.

    Note what is NOT here: no prompt caching. Caching only pays off above a
    minimum cacheable prefix of a few thousand tokens, and our system prompt is
    a few hundred. Adding cache_control would silently do nothing while looking
    like an optimisation. If interview mode later grows a much larger stable
    prompt, that is the moment to revisit it.

    Also no extended thinking: Haiku 4.5 does not accept the effort parameter,
    and summarising four supplied paragraphs is not a reasoning-heavy task.
    Both would add cost without adding accuracy.
    """
    model = model or config.GENERATION_MODEL
    hits = retrieve(question, top_k=top_k)
    client = client or get_client()

    try:
        response = client.messages.create(
            model=model,
            max_tokens=config.MAX_OUTPUT_TOKENS,
            system=SYSTEM_PROMPT,
            messages=build_messages(question, hits),
        )
    except anthropic.AuthenticationError:
        raise RuntimeError(
            "Anthropic rejected the API key. Check ANTHROPIC_API_KEY in your .env "
            "file at https://console.anthropic.com/settings/keys"
        ) from None
    except anthropic.RateLimitError as exc:
        retry_after = exc.response.headers.get("retry-after", "60")
        raise RuntimeError(f"Rate limited by the API. Try again in {retry_after}s.") from None
    except anthropic.APIStatusError as exc:
        if exc.status_code >= 500:
            raise RuntimeError(f"Anthropic server error ({exc.status_code}). Try again.") from None
        raise RuntimeError(f"API error {exc.status_code}: {exc.message}") from None
    except anthropic.APIConnectionError:
        raise RuntimeError("Could not reach the Anthropic API. Check your connection.") from None

    text = "".join(block.text for block in response.content if block.type == "text")
    return Answer(text=text, hits=hits, usage=response.usage, model=model)


# ---------------------------------------------------------------------------
# `python -m rag_buddy.generate "question"` (add --dry-run to spend nothing)
# ---------------------------------------------------------------------------


def dry_run(question: str, top_k: int | None = None) -> None:
    """
    Show exactly what WOULD be sent, and what it would cost, without sending it.

    Useful for seeing how the prompt is assembled, and for sanity-checking cost
    before running a long interview session.
    """
    hits = retrieve(question, top_k=top_k)
    messages = build_messages(question, hits)
    body = messages[0]["content"]

    print("=" * 78)
    print("SYSTEM PROMPT")
    print("=" * 78)
    print(SYSTEM_PROMPT)
    print()
    print("=" * 78)
    print("USER MESSAGE")
    print("=" * 78)
    print(body)
    print()

    words = len((SYSTEM_PROMPT + body).split())
    est_in = int(words * config.WORDS_TO_TOKENS_RATIO)
    est_out = 250  # a typical grounded answer
    cost = config.price_of(config.GENERATION_MODEL, est_in, est_out)
    print("=" * 78)
    print(f"Estimated: ~{est_in} input tokens + ~{est_out} output tokens")
    print(f"Estimated cost with {config.GENERATION_MODEL}: ${cost:.6f}")
    print("Actually sent: nothing. This was a dry run — $0.00 spent.")


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    is_dry = "--dry-run" in sys.argv

    if not args:
        print('Usage: python -m rag_buddy.generate "your question" [--dry-run]')
        sys.exit(1)

    question = " ".join(args)

    if is_dry:
        dry_run(question)
        return

    answer = answer_question(question)

    print(f'\nQ: {question}\n')
    print(answer.text)
    print()
    print(answer.sources_block())
    print()
    print(f"Tokens: {answer.input_tokens} in, {answer.output_tokens} out "
          f"— cost ${answer.cost:.6f} ({answer.model})")


if __name__ == "__main__":
    main()
