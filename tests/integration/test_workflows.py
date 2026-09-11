"""Multi-step flows through the real stack, the way a user actually runs them."""

from corpus_data import install_corpus
from rag_buddy import cli, config, retrieve, scaffold, store, todos

DRAFT_BODY = """# ledger-sync

## The problem

Month-end reconciliation took two days of manual spreadsheet matching, and
split payments were the part people most often got wrong.

## Architecture and the tradeoffs

- **Storage.** Reconciled results are written to SQLite.
- **Caching.** Parsed bank statements are cached for one hour.
  > **TODO:** Why one hour?
"""


def test_answered_draft_is_promoted_reindexed_and_searchable(fresh_workspace):
    todos.DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    draft = todos.DRAFTS_DIR / "project_ledger-sync.md"
    draft.write_text(scaffold.HEADER.format(filename=draft.name) + DRAFT_BODY, encoding="utf-8")

    lines = draft.read_text().splitlines()
    question = todos.find_todos(lines)[0]
    answer = ("An hour matched how often the bank export refreshes, so a longer "
              "cache would have shown stale balances.")
    todos._save(draft, todos._splice(lines, question, todos._replacement(question, answer)))

    dest = todos.finish(draft)                     # real re-index
    assert dest.exists() and not draft.exists()

    hits = retrieve.retrieve("Why were bank statements only cached for an hour?", top_k=1)
    assert "bank export refreshes" in hits[0].text
    assert "Parsed bank statements are cached" in hits[0].text   # answer kept with its bullet
    assert hits[0].section.endswith("Architecture and the tradeoffs")


def test_scaffold_to_answered_to_indexed(fresh_workspace, fakes, monkeypatch, scripted_input):
    """The whole draft workflow: repo document → draft → answer → promote → search."""
    install_corpus(config.DOCUMENTS_DIR)           # includes documents/github/repo_ledger-sync.md
    draft_text = ("# ledger-sync\n\n## What went wrong\n\n"
                  "The commit history records one timezone fix.\n\n"
                  "> **TODO:** What caused the timezone drift?\n")
    monkeypatch.setattr(scaffold, "get_client", lambda: fakes.Client(fakes.Response(draft_text)))

    written, _, question_count = scaffold.scaffold_all()
    assert question_count == 1

    scripted_input(
        "The drift came from treating posted dates as UTC when the bank exported local time.", "",
        "y",                                        # promote and re-index
    )
    todos.answer_draft(written[0])

    assert (config.DOCUMENTS_DIR / written[0].name).exists()
    hits = retrieve.retrieve("What caused the timezone drift?")
    assert any("bank exported local time" in h.text for h in hits)


def test_cli_ingest_after_the_status_screen(fresh_workspace, scripted_input, capsys):
    """
    Regression for the readonly-database crash, through the real menu: the
    status screen opens the store, then ingest rebuilds it underneath.
    """
    install_corpus(config.DOCUMENTS_DIR)
    store.build_index()                            # a returning user already has an index

    scripted_input("1", "7", "8")                  # ingest, refresh status, quit
    cli.main()

    out = capsys.readouterr().out
    assert "document(s) in" in out                 # the ingest summary line
    assert out.count("Interview Prep Study Buddy") == 2
    assert "Good luck in the interview." in out
