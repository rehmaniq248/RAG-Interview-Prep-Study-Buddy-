"""
Shared fixtures.

The autouse `isolate` fixture is the most important thing in the test suite.
It runs before every test and guarantees that no test can:

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


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
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
    monkeypatch.setattr(anthropic, "Anthropic", _Blocked)
    monkeypatch.setattr(store, "get_client", _blocked("ChromaDB"))
    monkeypatch.setattr(store, "get_embedder", _blocked("the embedding model"))
    monkeypatch.setattr(retrieve, "get_reranker", _blocked("the reranker"))
    monkeypatch.setattr(github, "_gh", _blocked("the GitHub CLI"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")
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
    def __init__(self, text="ok", input_tokens=100, output_tokens=50, blocks=None):
        self.content = blocks if blocks is not None else [FakeBlock(text)]
        self.usage = FakeUsage(input_tokens, output_tokens)


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
