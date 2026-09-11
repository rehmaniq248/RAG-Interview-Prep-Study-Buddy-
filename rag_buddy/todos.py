"""
Step 8d — Answering draft questions, one at a time.

Scaffolding leaves each draft full of TODO questions only you can answer. You
could edit the markdown by hand, but that means finding each question, writing
prose around blockquote syntax, and remembering to delete the question line.
This walks you through them instead: it shows a question, you type the answer,
it writes your answer into the right place and removes the question.

No API calls. It is file editing driven by your typing, so it is free.

WHAT KEEPS YOUR ANSWERS SAFE
  • every answer is saved the moment you submit it, so quitting halfway loses
    nothing — run it again and it picks up where you stopped
  • saves are atomic: written to a temporary file then swapped in, so a crash
    mid-write cannot leave the draft half-written
  • the untouched original is kept alongside as <name>.md.orig
  • a draft is only moved into documents/ once no questions remain, because an
    unanswered TODO in documents/ would be indexed as if it were a fact
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import config
from .scaffold import DRAFTS_DIR

TODO_MARKER = "**TODO:**"

_TODO_RE = re.compile(r"^(?P<indent>[ \t]*)>\s*\*\*TODO:\*\*\s*(?P<question>.*)$")
_QUOTE_RE = re.compile(r"^[ \t]*>\s?(?P<text>.*)$")
_HEADING_RE = re.compile(r"^#{1,6}\s+(?P<title>.+)$")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_DRAFT_HEADER_RE = re.compile(r"\A\s*<!--\s*DRAFT.*?-->\s*", re.DOTALL)

# Sentinels for the three commands, distinct from any typed answer.
SKIP, DELETE, QUIT = object(), object(), object()


@dataclass
class Todo:
    start: int        # first line of the TODO block
    end: int          # one past its last line
    indent: str       # leading whitespace — non-empty when inside a list item
    question: str
    section: str      # nearest heading above it
    context: str      # the grounded paragraph just above it


# ---------------------------------------------------------------------------
# Finding questions
# ---------------------------------------------------------------------------


def _context_before(lines: list[str], index: int, limit: int = 300) -> str:
    """
    The nearest paragraph of real content above a question.

    Shown with each question so you remember what it is asking about without
    having to open the file — a question about "the handle slugification" means
    more when you can see the sentence it was attached to.
    """
    def is_boundary(line: str) -> bool:
        s = line.strip()
        return not s or bool(_HEADING_RE.match(line)) or s.startswith("```")

    j = index - 1
    while j >= 0 and (not lines[j].strip() or _QUOTE_RE.match(lines[j])):
        j -= 1
    if j < 0 or is_boundary(lines[j]):
        return ""
    end = j
    while j >= 0 and not is_boundary(lines[j]) and not _QUOTE_RE.match(lines[j]):
        # List items sit on consecutive lines with no blank between them, so
        # stop at the start of this item rather than swallowing the whole list.
        if _LIST_ITEM_RE.match(lines[j]):
            j -= 1
            break
        j -= 1
    text = " ".join(line.strip() for line in lines[j + 1 : end + 1])
    return text if len(text) <= limit else "…" + text[-limit:]


def find_todos(lines: list[str]) -> list[Todo]:
    """
    Every TODO question in a draft, in document order.

    A question is a blockquote line starting with **TODO:**, plus any quoted
    lines that continue it. Text inside the draft's header comment and inside
    code fences is ignored — the header itself mentions **TODO:** while
    explaining what they are, and must not be mistaken for one.
    """
    todos: list[Todo] = []
    section = ""
    in_comment = in_fence = False
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if in_comment:
            if "-->" in line:
                in_comment = False
            i += 1
            continue
        if stripped.startswith("<!--") and "-->" not in stripped[4:]:
            in_comment = True
            i += 1
            continue
        if stripped.startswith("```"):
            in_fence = not in_fence
            i += 1
            continue
        if in_fence:
            i += 1
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            section = heading.group("title").strip()
            i += 1
            continue

        match = _TODO_RE.match(line)
        if match:
            parts = [match.group("question").strip()]
            j = i + 1
            while j < len(lines) and not _TODO_RE.match(lines[j]):
                cont = _QUOTE_RE.match(lines[j])
                if not cont or not cont.group("text").strip():
                    break
                parts.append(cont.group("text").strip())
                j += 1
            todos.append(Todo(
                start=i, end=j, indent=match.group("indent"),
                question=" ".join(parts), section=section,
                context=_context_before(lines, i),
            ))
            i = j
            continue

        i += 1

    return todos


# ---------------------------------------------------------------------------
# Editing the draft
# ---------------------------------------------------------------------------


def _replacement(todo: Todo, answer: str | None) -> list[str]:
    """
    The lines that take a question's place.

    Placement matters for retrieval, not just for looks. Chunking splits on
    blank lines, so:

      • a question INSIDE a list item gets its answer attached directly to
        that item, with no blank line. "Handle slugification. … It was
        deliberate because …" then stays one paragraph and gets retrieved
        together, keeping the answer next to what it is about.
      • a top-level question becomes its own paragraph, since the text above
        it is often an unrelated intro sentence.
    """
    if answer is None:
        return []
    if todo.indent:
        return [f"{todo.indent}{answer}"]
    return ["", answer, ""]


def _splice(lines: list[str], todo: Todo, block: list[str]) -> list[str]:
    """Swap a question for its replacement, tidying doubled blank lines at the seam."""
    new = lines[: todo.start] + block + lines[todo.end :]

    # Only normalise around the edit. Collapsing blank lines across the whole
    # file would also alter code fences, which may contain them on purpose.
    lo = max(todo.start - 1, 0)
    hi = min(todo.start + len(block) + 1, len(new))
    out = new[:lo]
    prev_blank = bool(out) and not out[-1].strip()
    for line in new[lo:hi]:
        blank = not line.strip()
        if blank and prev_blank:
            continue
        out.append(line)
        prev_blank = blank
    rest = new[hi:]
    if rest and prev_blank and not rest[0].strip():
        rest = rest[1:]
    return out + rest


def _save(path: Path, lines: list[str]) -> None:
    """
    Write atomically, keeping the untouched original on the first save.

    os.replace is atomic on the same filesystem: the draft is either fully the
    old version or fully the new one, never a partial write.
    """
    backup = path.with_name(path.name + ".orig")
    if not backup.exists():
        shutil.copy2(path, backup)

    text = "\n".join(lines).rstrip("\n") + "\n"
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".saving_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Promoting a finished draft
# ---------------------------------------------------------------------------


def finish(path: Path, documents_dir: Path | None = None, reindex: bool = True) -> Path:
    """
    Strip the draft header, move the file into documents/, and re-index.

    Refuses if any question remains or if the destination exists. Moving a
    draft is the point at which the tool starts treating its contents as true,
    so it should never happen with open questions in it or overwrite a file you
    already finished.
    """
    documents_dir = documents_dir or config.DOCUMENTS_DIR
    text = path.read_text(encoding="utf-8")

    remaining = find_todos(text.splitlines())
    if remaining:
        raise RuntimeError(
            f"{len(remaining)} question(s) are still open in {path.name}. "
            "Answer or delete them first — unanswered TODOs in documents/ would "
            "be indexed as if they were facts."
        )

    cleaned = _DRAFT_HEADER_RE.sub("", text, count=1).lstrip()
    if TODO_MARKER in cleaned:
        raise RuntimeError(
            f"{path.name} still contains a {TODO_MARKER} marker outside the usual "
            "question format. Open it and remove it by hand."
        )

    dest = documents_dir / path.name
    if dest.exists():
        raise RuntimeError(
            f"{dest} already exists. Not overwriting it — rename one of them first."
        )

    # Write the destination before removing the draft: if the write fails, the
    # draft is still exactly where it was.
    dest.write_text(cleaned, encoding="utf-8")
    path.unlink()

    if reindex:
        from .store import build_index
        build_index(rebuild=True)
    return dest


# ---------------------------------------------------------------------------
# The interactive session
# ---------------------------------------------------------------------------


def _read_answer():
    """Read one answer: lines until an empty one, or a single-letter command."""
    try:
        first = input("Your answer > ")
    except (EOFError, KeyboardInterrupt):
        print()
        return QUIT

    command = first.strip().lower()
    if command in {"", "s", "skip"}:
        return SKIP
    if command in {"d", "delete"}:
        return DELETE
    if command in {"q", "quit"}:
        return QUIT

    parts = [first.strip()]
    while True:
        try:
            line = input("            ")
        except (EOFError, KeyboardInterrupt):
            break
        if not line.strip():
            break
        parts.append(line.strip())
    return " ".join(parts)


INTRO = """
For each question, type your answer and press Enter on an empty line to save it.

   s   skip for now — it stays in the draft for next time
   d   delete it — it doesn't apply, or you don't remember
   q   stop — everything you've answered is already saved

Two rules worth keeping in mind:

 • If you're not sure, press d. Once this file is in documents/ it is the
   ground truth you get graded against. A detail you half-remember and get
   wrong here will mark you WRONG in practice for telling the truth.

 • Name the thing you're talking about. "Slugifying the handles was
   deliberate because…" can be found again later. "It was deliberate
   because…" can't — that paragraph gets retrieved on its own.
"""


def answer_draft(path: Path) -> None:
    """Walk through every open question in one draft."""
    lines = path.read_text(encoding="utf-8").splitlines()
    total = len(find_todos(lines))
    name = path.stem.removeprefix("project_")

    if total == 0:
        print(f"\n{path.name} has no open questions.")
        _offer_finish(path)
        return

    print(f"\n{name} — {total} open question(s).")
    print(INTRO)

    skipped: set[str] = set()
    answered = deleted = 0

    while True:
        lines = path.read_text(encoding="utf-8").splitlines()
        pending = [t for t in find_todos(lines) if t.question not in skipped]
        if not pending:
            break
        todo = pending[0]

        number = answered + deleted + len(skipped) + 1
        print("─" * 72)
        print(f"Question {number} of {total}  ·  {todo.section or 'untitled section'}")
        print("─" * 72)
        if todo.context:
            print(f"Context: {todo.context}\n")
        print(f"{todo.question}\n")

        result = _read_answer()
        if result is QUIT:
            break
        if result is SKIP:
            skipped.add(todo.question)
            print("Skipped — it stays in the draft.\n")
            continue
        if result is DELETE:
            _save(path, _splice(lines, todo, _replacement(todo, None)))
            deleted += 1
            print("Deleted and saved.\n")
            continue

        _save(path, _splice(lines, todo, _replacement(todo, result)))
        answered += 1
        print("Saved.\n")

    remaining = find_todos(path.read_text(encoding="utf-8").splitlines())
    print("─" * 72)
    print(f"Answered {answered}, deleted {deleted}, skipped {len(skipped)}. "
          f"{len(remaining)} still open.")

    if remaining:
        print(f"\nRun this again to pick up the rest — your answers are saved.")
        print("It stays in drafts/ until nothing is left open, so none of it is "
              "indexed yet.")
        return

    _offer_finish(path)


def _offer_finish(path: Path) -> None:
    """Ask before promoting — this is the step that makes the draft 'true'."""
    print("\nNo questions left. Give it a quick read-through first if you like:")
    print(f"  open -e {path}")
    try:
        choice = input("\nMove it into documents/ and re-index now? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        choice = ""

    if choice not in {"y", "yes"}:
        print("Left in drafts/. When you're ready:")
        print(f"  python -m rag_buddy.todos {path} --finish")
        return

    try:
        dest = finish(path)
    except RuntimeError as exc:
        print(f"\n{exc}")
        return
    print(f"\nMoved to {dest} and re-indexed. Interview mode can now use it.")


def pick_draft() -> Path:
    """Choose a draft interactively, showing how many questions each has left."""
    drafts = sorted(DRAFTS_DIR.glob("*.md")) if DRAFTS_DIR.is_dir() else []
    if not drafts:
        raise RuntimeError(
            "No drafts to answer. Create some first: python -m rag_buddy.scaffold"
        )

    counts = [(d, len(find_todos(d.read_text(encoding="utf-8").splitlines()))) for d in drafts]
    if len(counts) == 1:
        return counts[0][0]

    print("\nDrafts:\n")
    for i, (draft, n) in enumerate(counts, start=1):
        label = f"{n} open" if n else "ready to finish"
        print(f"  {i}. {draft.stem.removeprefix('project_'):<44} {label}")
    while True:
        try:
            choice = input("\nWhich one? > ").strip()
        except (EOFError, KeyboardInterrupt):
            raise RuntimeError("Cancelled.")
        if choice.isdigit() and 1 <= int(choice) <= len(counts):
            return counts[int(choice) - 1][0]
        print("Enter one of the numbers above.")


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--finish"]
    try:
        path = Path(args[0]) if args else pick_draft()
        if not path.is_file():
            raise RuntimeError(f"No such draft: {path}")
        if "--finish" in sys.argv:
            dest = finish(path)
            print(f"Moved to {dest} and re-indexed.")
            return
        answer_draft(path)
    except RuntimeError as exc:
        print(f"\n{exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
