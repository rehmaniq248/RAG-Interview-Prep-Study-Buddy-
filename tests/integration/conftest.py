"""
Integration fixtures: the real local models and a real ChromaDB.

What is real here: the embedding model, the reranker, the tokenizer, ChromaDB,
and file writes. What is still blocked (by the root conftest): the Anthropic
API and the GitHub CLI. Integration tests cost nothing to run.

Every test in this directory is marked `model` automatically, so it is skipped
by a plain `pytest` and run with `pytest -m model`.

Paths are redirected by a module-scoped `workspace` so that one index can be
built once and shared by every test in a module (building it is the slow part).
Tests that change documents or the index use `fresh_workspace` for their own.
"""

from pathlib import Path

import pytest

from rag_buddy import config, github, scaffold, todos

HERE = Path(__file__).parent


def pytest_collection_modifyitems(items):
    for item in items:
        if HERE in Path(item.path).parents:
            item.add_marker(pytest.mark.model)


def point_at(patcher, root: Path) -> Path:
    documents = root / "documents"
    documents.mkdir(parents=True, exist_ok=True)
    patcher.setattr(config, "DOCUMENTS_DIR", documents)
    patcher.setattr(config, "CHROMA_DIR", root / "chroma_db")
    for module in (scaffold, todos):
        patcher.setattr(module, "DRAFTS_DIR", root / "drafts")
    for module in (github, scaffold):
        patcher.setattr(module, "GITHUB_DIR", documents / "github")
    return root


@pytest.fixture(scope="module", autouse=True)
def workspace(tmp_path_factory):
    patcher = pytest.MonkeyPatch()
    root = point_at(patcher, tmp_path_factory.mktemp("workspace"))
    yield root
    patcher.undo()


@pytest.fixture
def fresh_workspace(tmp_path, monkeypatch):
    """An empty workspace for a test that modifies documents or the index."""
    return point_at(monkeypatch, tmp_path)
