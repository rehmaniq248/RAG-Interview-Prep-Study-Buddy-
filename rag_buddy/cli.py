"""
Step 7 — The menu that ties it together.

Nothing new happens here; this is a front door to the four things the other
modules already do. Building it last was deliberate — each piece was testable
on its own from the command line first, so when something misbehaves you can
still reach past this menu and run that one step in isolation:

    python -m rag_buddy.ingest              # see how documents get chunked
    python -m rag_buddy.store               # rebuild the index
    python -m rag_buddy.retrieve "..."      # see what a question retrieves
    python -m rag_buddy.generate "..."      # one grounded answer
    python -m rag_buddy.interview           # a practice session
"""

from __future__ import annotations

import sys

from . import config
from .interview import Usage, run_session
from .store import collection_stats


def _key_is_set() -> bool:
    try:
        config.get_api_key()
        return True
    except RuntimeError:
        return False


def show_status() -> None:
    """What is indexed, and what is ready to run."""
    stats = collection_stats()
    have_key = _key_is_set()

    print("\n" + "=" * 60)
    print("  Interview Prep Study Buddy")
    print("=" * 60)

    if stats["count"]:
        print(f"  Indexed : {stats['count']} chunks from "
              f"{len(stats['sources'])} document(s)")
        for source in stats["sources"]:
            print(f"            · {source}")
    else:
        print(f"  Indexed : nothing yet")
        print(f"            Put .txt or .md files in {config.DOCUMENTS_DIR.name}/ "
              f"and choose option 1.")

    print(f"  API key : {'set' if have_key else 'NOT SET — options 2 and 3 need one'}")
    print(f"  Models  : {config.GENERATION_MODEL} for answers, "
          f"{config.EVALUATION_MODEL} for grading")
    print("=" * 60)


def do_ingest() -> None:
    """Chunk, embed and store every document. Local and free."""
    # Imported here so the menu opens instantly: importing store pulls in
    # torch, which takes a couple of seconds.
    from .store import build_index

    print("\nReading documents/, chunking, and embedding locally…")
    try:
        stats = build_index(rebuild=True)
    except RuntimeError as exc:
        print(f"\n{exc}")
        return

    report = stats["truncation"]
    print(f"\nIndexed {stats['chunks']} chunks from {stats['documents']} document(s) "
          f"in {stats['embed_seconds']:.1f}s.")
    if report.ok:
        print(f"All chunks fit the embedding model's {report.limit}-token window "
              f"(longest: {report.longest}).")
    else:
        print(f"\n⚠ {len(report.offenders)} chunk(s) exceed the "
              f"{report.limit}-token window and will be silently truncated:")
        for chunk_id, n in report.offenders[:5]:
            print(f"    {chunk_id}: {n} tokens")
        print("  Lower CHUNK_TARGET_WORDS in config.py and re-ingest.")
    print("Cost: $0.00 — this step never leaves your machine.")


def do_ask(usage: Usage) -> None:
    """Ask questions about your own background until you're done."""
    from .generate import answer_question

    if not collection_stats()["count"]:
        print("\nNothing is indexed yet — choose option 1 first.")
        return
    if not _key_is_set():
        print("\nThis needs an API key. Put one in .env (see .env.example).")
        return

    print("\nAsk anything about your documents. Blank line returns to the menu.")
    while True:
        try:
            question = input("\nQuestion > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            return

        try:
            answer = answer_question(question)
        except RuntimeError as exc:
            print(f"\n{exc}")
            return

        print(f"\n{answer.text}\n")
        print(answer.sources_block())

        usage.calls += 1
        usage.input_tokens += answer.input_tokens
        usage.output_tokens += answer.output_tokens
        usage.cost += answer.cost
        print(f"\n({answer.input_tokens} in / {answer.output_tokens} out — "
              f"${answer.cost:.6f})")


def do_interview(usage: Usage) -> None:
    """Practice session: it asks, you answer, it grades."""
    if not collection_stats()["count"]:
        print("\nNothing is indexed yet — choose option 1 first.")
        return
    if not _key_is_set():
        print("\nThis needs an API key. Put one in .env (see .env.example).")
        return

    stats = collection_stats()
    sources = stats["sources"]
    source = None

    if len(sources) > 1:
        print("\nWhich document should the questions come from?")
        print("  0. Any (picked at random)")
        for i, name in enumerate(sources, start=1):
            print(f"  {i}. {name}")
        try:
            choice = input("Choice > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if choice.isdigit() and 1 <= int(choice) <= len(sources):
            source = sources[int(choice) - 1]

    try:
        session = run_session(source=source)
    except RuntimeError as exc:
        print(f"\n{exc}")
        return

    # run_session tracks its own usage; fold it into the session total.
    if session:
        usage.calls += session.calls
        usage.input_tokens += session.input_tokens
        usage.output_tokens += session.output_tokens
        usage.cost += session.cost


MENU = """
  1. Ingest documents      (chunk + embed everything in documents/ — free)
  2. Ask a question        (grounded answer with citations)
  3. Interview mode        (it asks, you answer, it grades you)
  4. Refresh status
  5. Quit
"""


def main() -> None:
    usage = Usage()
    show_status()

    while True:
        print(MENU)
        try:
            choice = input("Choose > ").strip()
        except (EOFError, KeyboardInterrupt):
            choice = "5"
            print()

        if choice == "1":
            do_ingest()
        elif choice == "2":
            do_ask(usage)
        elif choice == "3":
            do_interview(usage)
        elif choice == "4":
            show_status()
        elif choice in {"5", "q", "quit", "exit"}:
            if usage.calls:
                print(f"\nThis session: {usage.summary()}")
            print("Good luck in the interview.\n")
            return
        else:
            print(f"\n'{choice}' isn't one of the options.")


if __name__ == "__main__":
    sys.exit(main())
