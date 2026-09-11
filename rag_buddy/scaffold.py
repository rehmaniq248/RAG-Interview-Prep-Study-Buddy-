"""
Step 8b — Drafting project write-ups from your repositories.

THE PROBLEM THIS SOLVES
Repository facts make the tool broader but not deeper. A README says what a
project does; commits say what changed. Neither says why you chose what you
chose, what you rejected, or what went wrong — and those are the questions
interviewers actually ask. That reasoning exists only in your head, so no
amount of ingestion will surface it.

WHAT THIS DOES INSTEAD
Turns the repo into a draft write-up with the reasoning left as questions for
you. Claude fills in everything the repository can evidence, and wherever the
answer would require knowing your intent, it writes a TODO asking you directly.
Answering twenty pointed questions about a project you built is far easier than
facing a blank page, and the result is the document shape interview mode grades
best.

WHY DRAFTS ARE NOT INDEXED
They land in drafts/ rather than documents/, and nothing reads them until you
move them across. A draft is part evidence and part open question; indexing it
unedited would put unanswered TODOs into the store as though they were facts
about your career. You promote a draft by editing it and copying it into
documents/ — a deliberate act, because from then on the tool treats it as true.
"""

from __future__ import annotations

import sys
from pathlib import Path

import anthropic

from . import config
from .generate import get_client
from .github import GITHUB_DIR

# Drafts live outside documents/ so they are never indexed by accident.
DRAFTS_DIR = config.PROJECT_ROOT / "drafts"

SCAFFOLD_SYSTEM_PROMPT = """\
You help someone turn a code repository into an interview-ready project \
write-up. You will be given facts extracted from the repository: its README, \
file structure, dependencies, and commit history.

Produce a markdown write-up with exactly these sections:

# <Project name>

## The problem
## My role
## How it works
## Architecture and the tradeoffs
## What went wrong
## Outcome

Two kinds of content, and never blur them:

1. GROUNDED — anything the repository evidences. State it plainly. Prefer \
specifics: file names, libraries, counts, the actual pipeline stages.

2. TODO — anything requiring knowledge the repository does not contain: why a \
decision was made, what alternative was rejected, what broke, what the impact \
was, what they would change. Write these as:

> **TODO:** <a specific, answerable question>

Never invent an answer to a TODO. Never write a plausible-sounding reason the \
repository does not support — a fabricated rationale is worse than a blank, \
because they will read it back later and believe they wrote it.

Make the TODOs pointed and specific to THIS project. "TODO: why did you choose \
Next.js over a plain React SPA for a 24-hour hackathon build?" is useful. \
"TODO: describe your technical decisions" is not.

Every section needs at least one TODO except where the repository genuinely \
covers it. "What went wrong" will be almost entirely TODOs — repositories \
rarely record failures — and that section matters most, so ask several sharp \
questions there.

Write in the second person. Be concise; this is a working draft they will \
edit, not prose to admire."""


def draft_from_facts(facts_markdown: str, client=None) -> tuple[str, object]:
    """Ask Claude to turn one repo's facts into a write-up draft."""
    client = client or get_client()
    response = client.messages.create(
        model=config.GENERATION_MODEL,
        max_tokens=2048,
        system=SCAFFOLD_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Repository facts:\n\n{facts_markdown}\n\n"
                       f"Write the draft project write-up.",
        }],
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    return text, response.usage


HEADER = """<!--
DRAFT — generated from your repository, then left for you to finish.

Everything not marked TODO is evidenced by the repo. Every **TODO:** is a
question only you can answer; that is where the interview value is.

Easiest: answer them one at a time, and it moves the file for you when done:

    python -m rag_buddy.todos drafts/{filename}

Or by hand: delete this header and the TODO lines, then move and re-ingest:

    mv drafts/{filename} documents/
    python -m rag_buddy.store

Until you move it, nothing here is indexed and the tool cannot see it.
-->

"""


def scaffold_all(paths: list[Path] | None = None) -> tuple[list[Path], float, int]:
    """Draft a write-up for each repo facts document."""
    paths = paths or sorted(GITHUB_DIR.glob("repo_*.md"))
    if not paths:
        raise RuntimeError(
            f"No repository documents in {GITHUB_DIR}.\n"
            "Fetch some first: python -m rag_buddy.github owner/repo"
        )

    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    client = get_client()
    written: list[Path] = []
    cost = 0.0
    todos = 0

    for path in paths:
        name = path.stem.removeprefix("repo_")
        out = DRAFTS_DIR / f"project_{name}.md"
        if out.exists():
            print(f"  {name}: draft already exists, skipping (delete it to redo)")
            continue

        print(f"  drafting {name} …", end=" ", flush=True)
        try:
            text, usage = draft_from_facts(path.read_text(encoding="utf-8"), client)
        except anthropic.APIStatusError as exc:
            print(f"failed ({exc.status_code})")
            continue

        n = text.count("**TODO:**")
        todos += n
        cost += config.price_of(config.GENERATION_MODEL,
                                usage.input_tokens, usage.output_tokens)
        out.write_text(HEADER.format(filename=out.name) + text, encoding="utf-8")
        written.append(out)
        print(f"ok ({n} questions for you, ${config.price_of(config.GENERATION_MODEL, usage.input_tokens, usage.output_tokens):.4f})")

    return written, cost, todos


def main() -> None:
    args = [Path(a) for a in sys.argv[1:]] or None
    written, cost, todos = scaffold_all(args)

    print(f"\n{len(written)} draft(s) in {DRAFTS_DIR}")
    print(f"{todos} questions waiting for you across them.")
    print(f"Cost: ${cost:.4f}")
    print("\nAnswer them one at a time with:  python -m rag_buddy.todos")
    print("Drafts are NOT indexed until you move them.")


if __name__ == "__main__":
    main()
