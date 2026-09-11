"""
Shared fixtures.

The autouse `isolate` fixture is the most important thing in the test suite.

For EVERY test, core or integration, it blocks the Anthropic API and the
GitHub CLI and sets a fake API key. Nothing in the suite can spend money or
reach GitHub.

For core tests it additionally guarantees that no test can:

  • read or write your real documents/, drafts/ or chroma_db/ — every path is
    redirected into a per-test temporary directory
  • call the Anthropic API, the GitHub CLI, or ChromaDB — each is replaced with
    a blocker that fails the test loudly if anything reaches it
  • load the embedding model, the reranker, or the real tokenizer — core tests
    must run in seconds with no downloads

Token counting is replaced with a deterministic fake that mimics the two
properties of the real WordPiece tokenizer that matter here:

  • long unbroken strings cost many tokens — a file path is one "word" but
    about eight tokens, which is what caused the file-tree truncation bug, so
    regression tests for that bug still bite
  • every call adds a fixed 2 special tokens ([CLS] and [SEP]), so counting
    pieces separately over-estimates their joined total, just as it does for
    real. A naive characters/4 fake gets this backwards and reports budget
    overflows that cannot happen with the real tokenizer.
"""

import math
from pathlib import Path

import anthropic
import httpx2
import pytest

from rag_buddy import config, generate, github, ingest, interview, retrieve, scaffold, store, todos
from rag_buddy.retrieve import RetrievedChunk


def fake_count_tokens(text: str) -> int:
    words = text.split()
    return sum(math.ceil(len(w) / 4) for w in words) + 2 if words else 0


class _Blocked:
    def __init__(self, *args, **kwargs):
        raise AssertionError("A test tried to construct a real Anthropic client.")


def _blocked(name):
    def fail(*args, **kwargs):
        raise AssertionError(f"A test reached {name}, which must be faked in core tests.")
    return fail


def _refuse_real_folders():
    """
    Integration tests use a real ChromaDB and real file writes. Their paths are
    redirected by tests/integration/conftest.py; if a `model` test somehow runs
    without that redirect, fail rather than let it near the project's folders.
    """
    checks = {
        "documents/": (config.DOCUMENTS_DIR, config.PROJECT_ROOT / "documents"),
        "chroma_db/": (config.CHROMA_DIR, config.PROJECT_ROOT / "chroma_db"),
        "drafts/": (todos.DRAFTS_DIR, config.PROJECT_ROOT / "drafts"),
        "drafts/ (scaffold)": (scaffold.DRAFTS_DIR, config.PROJECT_ROOT / "drafts"),
        "documents/github/": (github.GITHUB_DIR, config.PROJECT_ROOT / "documents" / "github"),
    }
    for name, (live, project) in checks.items():
        if Path(live).resolve() == project.resolve():
            pytest.fail(
                f"A `model` test would use the project's real {name} folder. "
                "Integration tests must live under tests/integration/, whose "
                "workspace fixture redirects every path."
            )


@pytest.fixture(autouse=True)
def isolate(request, tmp_path, monkeypatch):
    # Every test: never the real API, never GitHub.
    monkeypatch.setattr(anthropic, "Anthropic", _Blocked)
    monkeypatch.setattr(github, "_gh", _blocked("the GitHub CLI"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")

    if request.node.get_closest_marker("model"):
        # Real models and ChromaDB; paths handled by the integration workspace.
        _refuse_real_folders()
        return tmp_path

    documents = tmp_path / "documents"
    documents.mkdir()
    drafts = tmp_path / "drafts"
    github_dir = documents / "github"

    monkeypatch.setattr(config, "DOCUMENTS_DIR", documents)
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma_db")
    for module in (scaffold, todos):
        monkeypatch.setattr(module, "DRAFTS_DIR", drafts)
    for module in (github, scaffold):
        monkeypatch.setattr(module, "GITHUB_DIR", github_dir)

    monkeypatch.setattr(ingest, "count_tokens", fake_count_tokens)
    monkeypatch.setattr(store, "get_client", _blocked("ChromaDB"))
    monkeypatch.setattr(store, "get_embedder", _blocked("the embedding model"))
    monkeypatch.setattr(retrieve, "get_reranker", _blocked("the reranker"))
    return tmp_path


# --- fake Anthropic responses -------------------------------------------------


class FakeBlock:
    def __init__(self, text, type="text"):
        self.type = type
        self.text = text


class FakeUsage:
    def __init__(self, input_tokens=100, output_tokens=50):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeResponse:
    def __init__(self, text="ok", input_tokens=100, output_tokens=50, blocks=None,
                 stop_reason="end_turn"):
        self.content = blocks if blocks is not None else [FakeBlock(text)]
        self.usage = FakeUsage(input_tokens, output_tokens)
        # Responses that ran out of room report "max_tokens". On a thinking
        # model that can mean the whole budget went on reasoning and no visible
        # text came back at all — which is what made grading look like a hang.
        self.stop_reason = stop_reason


class FakeMessages:
    def __init__(self, responses):
        self.calls = []
        self._responses = list(responses)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("FakeClient received more calls than it has responses.")
        result = self._responses.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class FakeClient:
    """Stands in for anthropic.Anthropic. Records every request it receives."""

    def __init__(self, *responses):
        self.messages = FakeMessages(responses)


@pytest.fixture
def fakes():
    """Access to the fake response classes from inside tests."""
    class Namespace:
        Client = FakeClient
        Response = FakeResponse
        Block = FakeBlock
    return Namespace


_REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


@pytest.fixture
def api_error():
    """Build a real anthropic exception carrying a given HTTP status."""
    def build(cls, status, headers=None):
        if cls is anthropic.APIConnectionError:
            return cls(request=_REQUEST)
        response = httpx2.Response(status, request=_REQUEST, headers=headers or {})
        return cls(f"HTTP {status}", response=response, body=None)
    return build


@pytest.fixture
def make_hit():
    def build(text="Some passage.", source="doc.md", section="Section",
              distance=0.5, rank=1, rerank_score=None):
        return RetrievedChunk(text=text, source=source, section=section,
                              citation=f"{source} § {section}", distance=distance,
                              rank=rank, rerank_score=rerank_score)
    return build


@pytest.fixture
def scripted_input(monkeypatch):
    """Feed a fixed sequence of lines to input(); EOF once they run out."""
    def install(*lines):
        remaining = iter(lines)

        def fake_input(prompt=""):
            try:
                return next(remaining)
            except StopIteration:
                raise EOFError
        monkeypatch.setattr("builtins.input", fake_input)
    return install
