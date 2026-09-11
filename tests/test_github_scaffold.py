"""GitHub fact extraction and write-up scaffolding — no network, no API."""

import anthropic
import pytest

from rag_buddy import config, github, ingest, scaffold

FACTS = {
    "name": "widget-pipeline-", "full_name": "someone/widget-pipeline-",
    "description": "Moves widgets.", "languages": {"Python": 18094},
    "created": "2026-01-01", "updated": "2026-02-01",
    "readme": "# Widget Pipeline\n\n## Usage\n\n```bash\n# a shell comment\n```",
    "tree": ["README.md", "main.py"], "commits": ["Add importer", "Fix size parsing"],
    "manifests": {"requirements.txt": "openpyxl\n"}, "private": False,
}


class TestDemoteHeadings:
    def test_every_heading_pushed_down_one_level(self):
        assert github._demote_headings("# A\ntext\n## B") == "## A\ntext\n### B"

    def test_hash_lines_inside_fences_untouched(self):
        assert github._demote_headings("# A\n```bash\n# comment\n```") == "## A\n```bash\n# comment\n```"


class TestFactsToMarkdown:
    def test_sections_appear_in_order(self):
        md = github.facts_to_markdown(FACTS)
        headings = ["# widget-pipeline- (GitHub repository)", "## Repository overview",
                    "## README", "## Repository structure", "## Dependencies and stack",
                    "## Commit history"]
        positions = [md.index(h) for h in headings]
        assert positions == sorted(positions)

    def test_readme_cannot_reset_the_top_level_heading(self):
        # Citations come from the heading trail, so a README's own "# Title"
        # must never replace the repository as the root of that trail. Checked
        # through the real chunker rather than by scanning for "# " lines: a
        # "#" inside a code fence is a shell comment, not a heading, and only
        # the chunker knows which is which.
        md = github.facts_to_markdown(FACTS)
        sections = [section for section, _ in ingest.split_into_blocks(md)]
        assert sections
        assert all(s.startswith("widget-pipeline- (GitHub repository)") for s in sections)
        assert "## Widget Pipeline" in md

    def test_shell_comment_in_readme_code_is_not_demoted(self):
        assert "\n# a shell comment\n" in github.facts_to_markdown(FACTS)

    def test_commits_listed_as_bullets(self):
        md = github.facts_to_markdown(FACTS)
        assert "- Add importer" in md and "- Fix size parsing" in md

    def test_long_tree_truncated_with_a_count(self):
        facts = {**FACTS, "tree": [f"f{i}.py" for i in range(github.MAX_TREE_ENTRIES + 5)]}
        assert "... and 5 more files" in github.facts_to_markdown(facts)

    def test_large_manifest_truncated(self):
        md = github.facts_to_markdown({**FACTS, "manifests": {"package.json": "x" * 5000}})
        assert "x" * 2000 + "\n..." in md
        assert "x" * 2001 not in md

    def test_empty_sections_are_omitted(self):
        md = github.facts_to_markdown({**FACTS, "readme": "", "tree": [], "commits": [],
                                       "manifests": {}})
        for heading in ("## README", "## Repository structure", "## Dependencies", "## Commit history"):
            assert heading not in md


class TestRequireGh:
    def test_missing_cli_explains_how_to_install(self, monkeypatch):
        monkeypatch.setattr(github.shutil, "which", lambda name: None)
        with pytest.raises(RuntimeError, match="brew install gh"):
            github.require_gh()

    def test_logged_out_cli_explains_how_to_log_in(self, monkeypatch):
        monkeypatch.setattr(github.shutil, "which", lambda name: "/usr/local/bin/gh")
        monkeypatch.setattr(github, "_gh", lambda *args: None)
        with pytest.raises(RuntimeError, match="gh auth login"):
            github.require_gh()


def test_sync_repos_writes_documents_and_skips_failures(monkeypatch):
    monkeypatch.setattr(github, "require_gh", lambda: None)

    def fetch(name):
        if name == "someone/broken":
            raise RuntimeError("no access")
        return {**FACTS, "name": name.split("/")[1], "full_name": name}

    monkeypatch.setattr(github, "fetch_repo_facts", fetch)
    written = github.sync_repos(["someone/Momentum-Tool-", "someone/broken"])
    assert [p.name for p in written] == ["repo_Momentum-Tool.md"]
    assert written[0].parent == github.GITHUB_DIR
    assert written[0].read_text().startswith("# Momentum-Tool- (GitHub repository)")


# --- scaffolding --------------------------------------------------------------

DRAFT_TEXT = ("# Widget\n\n## The problem\n\nIt moved widgets.\n\n"
              "> **TODO:** Why?\n> **TODO:** What broke?\n> **TODO:** Impact?\n")


@pytest.fixture
def repo_doc():
    github.GITHUB_DIR.mkdir(parents=True, exist_ok=True)
    path = github.GITHUB_DIR / "repo_widget.md"
    path.write_text(github.facts_to_markdown(FACTS))
    return path


class TestScaffold:
    def test_writes_draft_with_header_and_counts_questions(self, repo_doc, fakes, monkeypatch):
        monkeypatch.setattr(scaffold, "get_client", lambda: fakes.Client(fakes.Response(DRAFT_TEXT)))
        written, cost, question_count = scaffold.scaffold_all()
        assert [p.name for p in written] == ["project_widget.md"]
        text = written[0].read_text()
        assert text.startswith("<!--\nDRAFT")
        assert text.endswith(DRAFT_TEXT)
        assert question_count == 3
        assert cost > 0

    def test_drafts_land_outside_documents_so_are_never_indexed(self, repo_doc, fakes, monkeypatch):
        monkeypatch.setattr(scaffold, "get_client", lambda: fakes.Client(fakes.Response(DRAFT_TEXT)))
        written, _, _ = scaffold.scaffold_all()
        assert config.DOCUMENTS_DIR not in written[0].parents

    def test_existing_draft_is_neither_overwritten_nor_billed(self, repo_doc, fakes, monkeypatch):
        scaffold.DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
        existing = scaffold.DRAFTS_DIR / "project_widget.md"
        existing.write_text("my answers")
        client = fakes.Client()                      # no responses: any call fails
        monkeypatch.setattr(scaffold, "get_client", lambda: client)
        written, cost, _ = scaffold.scaffold_all()
        assert written == [] and cost == 0.0
        assert existing.read_text() == "my answers"
        assert client.messages.calls == []

    def test_api_failure_skips_that_repo_without_writing(self, repo_doc, fakes, api_error, monkeypatch):
        client = fakes.Client(api_error(anthropic.InternalServerError, 500))
        monkeypatch.setattr(scaffold, "get_client", lambda: client)
        written, _, _ = scaffold.scaffold_all()
        assert written == []
        assert not (scaffold.DRAFTS_DIR / "project_widget.md").exists()

    def test_no_repo_documents_explains_how_to_fetch(self):
        with pytest.raises(RuntimeError, match="rag_buddy.github"):
            scaffold.scaffold_all()

    def test_draft_request_uses_generation_model(self, fakes):
        client = fakes.Client(fakes.Response("x"))
        scaffold.draft_from_facts("facts here", client=client)
        call = client.messages.calls[0]
        assert call["model"] == config.GENERATION_MODEL
        assert call["system"] == scaffold.SCAFFOLD_SYSTEM_PROMPT
        assert "facts here" in call["messages"][0]["content"]

    def test_prompt_forbids_inventing_reasons(self):
        assert "Never invent an answer to a TODO" in scaffold.SCAFFOLD_SYSTEM_PROMPT
