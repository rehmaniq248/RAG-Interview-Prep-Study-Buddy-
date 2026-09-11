"""
The guided TODO editor.

Everything here rewrites files full of the user's own answers, so the bar is:
never lose an answer, never corrupt a draft, never promote an unfinished one.
"""

import pytest

from rag_buddy import config, scaffold, todos

BODY = """# Widget Pipeline

## Architecture and the tradeoffs

- **Storage.** Data lives in SQLite.
- **Caching.** Results are cached for an hour.
  > **TODO:** Why one hour rather than a day?
- **Batching.** Writes are batched.

> **TODO:** How were failures
> retried?

## What went wrong

The repository records no failures.

> **TODO:** What broke first?
> **TODO:** What did you change after?
"""

QUESTIONS = [
    "Why one hour rather than a day?",
    "How were failures retried?",
    "What broke first?",
    "What did you change after?",
]


@pytest.fixture
def draft(isolate):
    todos.DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    path = todos.DRAFTS_DIR / "project_widget.md"
    path.write_text(scaffold.HEADER.format(filename=path.name) + BODY, encoding="utf-8")
    return path


def lines_of(path):
    return path.read_text(encoding="utf-8").splitlines()


def open_questions(path):
    return todos.find_todos(lines_of(path))


def resolve_all(path):
    """Delete every question, the way pressing d repeatedly would."""
    while found := open_questions(path):
        todos._save(path, todos._splice(lines_of(path), found[0],
                                        todos._replacement(found[0], None)))


# --- finding questions --------------------------------------------------------


class TestFindTodos:
    def test_finds_every_real_question_in_order(self, draft):
        assert [t.question for t in open_questions(draft)] == QUESTIONS

    def test_header_mention_of_marker_is_not_a_question(self, draft):
        assert "**TODO:**" in scaffold.HEADER          # the header does mention it…
        assert len(open_questions(draft)) == 4         # …and it is not counted

    def test_marker_inside_code_fence_is_not_a_question(self):
        lines = ["```", "> **TODO:** not real", "```", "> **TODO:** real?"]
        assert [t.question for t in todos.find_todos(lines)] == ["real?"]

    def test_sections_recorded(self, draft):
        assert [t.section for t in open_questions(draft)] == [
            "Architecture and the tradeoffs", "Architecture and the tradeoffs",
            "What went wrong", "What went wrong"]

    def test_list_item_indent_recorded(self, draft):
        assert [t.indent for t in open_questions(draft)] == ["  ", "", "", ""]

    def test_multiline_question_spans_both_lines(self, draft):
        q = open_questions(draft)[1]
        assert q.end - q.start == 2

    def test_context_for_list_item_is_only_its_own_bullet(self, draft):
        context = open_questions(draft)[0].context
        assert "Caching" in context
        assert "Storage" not in context

    def test_context_for_top_level_question_is_paragraph_above(self, draft):
        assert open_questions(draft)[2].context == "The repository records no failures."

    def test_no_context_directly_under_a_heading(self):
        assert todos.find_todos(["## H", "> **TODO:** q?"])[0].context == ""


# --- editing ------------------------------------------------------------------


def edit(path, index, answer):
    q = open_questions(path)[index]
    todos._save(path, todos._splice(lines_of(path), q, todos._replacement(q, answer)))


class TestEditing:
    def test_list_item_answer_attaches_to_bullet_without_blank_line(self, draft):
        edit(draft, 0, "An hour matched the upstream refresh.")
        lines = lines_of(draft)
        i = next(n for n, line in enumerate(lines) if "**Caching.**" in line)
        # No blank line: chunking keeps the answer in the same chunk as its bullet.
        assert lines[i + 1] == "  An hour matched the upstream refresh."

    def test_top_level_answer_is_its_own_paragraph(self, draft):
        edit(draft, 2, "The importer broke first.")
        lines = lines_of(draft)
        i = lines.index("The importer broke first.")
        assert lines[i - 1] == "" and lines[i + 1] == ""

    def test_delete_removes_continuation_lines_too(self, draft):
        edit(draft, 1, None)
        text = draft.read_text()
        assert "How were failures" not in text
        assert "retried?" not in text

    def test_other_questions_survive_an_edit(self, draft):
        edit(draft, 2, "Answered.")
        assert [t.question for t in open_questions(draft)] == [
            QUESTIONS[0], QUESTIONS[1], QUESTIONS[3]]

    @pytest.mark.parametrize("answer", [None, "An answer."])
    def test_no_doubled_blank_lines_at_any_seam(self, draft, answer):
        while found := open_questions(draft):
            todos._save(draft, todos._splice(lines_of(draft), found[0],
                                             todos._replacement(found[0], answer)))
        lines = lines_of(draft)
        doubled = [n for n in range(1, len(lines))
                   if not lines[n].strip() and not lines[n - 1].strip()]
        assert doubled == []


class TestSave:
    def test_keeps_the_untouched_original_exactly_once(self, draft):
        original = draft.read_text()
        todos._save(draft, ["first edit"])
        todos._save(draft, ["second edit"])
        assert draft.with_name(draft.name + ".orig").read_text() == original
        assert draft.read_text() == "second edit\n"

    def test_leaves_no_temporary_files(self, draft):
        todos._save(draft, ["x"])
        assert [p.name for p in draft.parent.iterdir() if p.name.startswith(".saving_")] == []

    def test_backup_is_not_listed_as_a_draft(self, draft):
        todos._save(draft, ["x"])
        assert [p.name for p in todos.DRAFTS_DIR.glob("*.md")] == ["project_widget.md"]


# --- promoting ----------------------------------------------------------------


class TestFinish:
    def test_refuses_while_questions_remain(self, draft):
        with pytest.raises(RuntimeError, match="still open"):
            todos.finish(draft, reindex=False)
        assert draft.exists()
        assert not (config.DOCUMENTS_DIR / draft.name).exists()

    def test_refuses_to_overwrite_an_existing_document(self, draft):
        resolve_all(draft)
        existing = config.DOCUMENTS_DIR / draft.name
        existing.write_text("already finished")
        with pytest.raises(RuntimeError, match="already exists"):
            todos.finish(draft, reindex=False)
        assert existing.read_text() == "already finished"
        assert draft.exists()

    def test_refuses_stray_marker_outside_question_format(self, draft):
        resolve_all(draft)
        draft.write_text(draft.read_text() + "\nRemember **TODO:** this later.\n")
        with pytest.raises(RuntimeError, match="marker"):
            todos.finish(draft, reindex=False)

    def test_promotes_a_finished_draft(self, draft):
        resolve_all(draft)
        dest = todos.finish(draft, reindex=False)
        assert dest == config.DOCUMENTS_DIR / "project_widget.md"
        assert not draft.exists()
        text = dest.read_text()
        assert text.startswith("# Widget Pipeline")
        assert "<!--" not in text
        assert "**TODO:**" not in text


# --- the interactive session --------------------------------------------------


class TestAnswerDraft:
    def test_answer_skip_delete_then_quit_saves_progress(self, draft, scripted_input):
        scripted_input(
            "An hour matched the upstream refresh.", "",   # Q1 answered
            "s",                                           # Q2 skipped
            "d",                                           # Q3 deleted
            "q",                                           # stop at Q4
        )
        todos.answer_draft(draft)
        assert [t.question for t in open_questions(draft)] == [QUESTIONS[1], QUESTIONS[3]]
        text = draft.read_text()
        assert "  An hour matched the upstream refresh." in text
        assert QUESTIONS[2] not in text
        assert draft.with_name(draft.name + ".orig").exists()

    def test_resume_then_decline_the_move(self, draft, scripted_input):
        scripted_input("s", "d", "d", "q")
        todos.answer_draft(draft)                                  # leaves Q1 and Q4
        scripted_input("Cached hourly because the source refreshes hourly.", "", "d", "n")
        todos.answer_draft(draft)
        assert open_questions(draft) == []
        assert "Cached hourly because" in draft.read_text()
        assert draft.exists()                                       # "n" kept it in drafts/

    def test_accepting_the_move_calls_finish(self, draft, scripted_input, monkeypatch):
        resolve_all(draft)
        calls = []
        monkeypatch.setattr(todos, "finish", lambda path: calls.append(path) or path)
        scripted_input("y")
        todos.answer_draft(draft)
        assert calls == [draft]

    def test_multiline_answer_joined_into_one_paragraph(self, draft, scripted_input):
        scripted_input("s", "The first thing to break", "was the importer.", "", "q")
        todos.answer_draft(draft)
        assert "The first thing to break was the importer." in lines_of(draft)

    def test_blank_entries_skip_and_change_nothing(self, draft, scripted_input):
        before = draft.read_text()
        scripted_input("", "", "", "")
        todos.answer_draft(draft)
        assert draft.read_text() == before
        assert not draft.with_name(draft.name + ".orig").exists()

    def test_end_of_input_quits_cleanly(self, draft, scripted_input):
        scripted_input()
        todos.answer_draft(draft)
        assert len(open_questions(draft)) == 4


class TestPickDraft:
    def test_no_drafts_explains_how_to_make_some(self):
        with pytest.raises(RuntimeError, match="rag_buddy.scaffold"):
            todos.pick_draft()

    def test_single_draft_chosen_without_prompting(self, draft, scripted_input):
        scripted_input()             # a prompt would hit EOF and raise
        assert todos.pick_draft() == draft

    def test_reprompts_until_a_valid_choice(self, draft, scripted_input):
        other = todos.DRAFTS_DIR / "project_another.md"
        other.write_text(BODY)
        scripted_input("9", "abc", "1")
        assert todos.pick_draft() == other    # sorted: another < widget
