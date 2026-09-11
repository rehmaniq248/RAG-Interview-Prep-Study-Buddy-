"""Reading documents and chunking them — the layer every later step depends on."""

import pytest

from rag_buddy import config, ingest
from rag_buddy.ingest import Chunk


def tok(text):
    return ingest.count_tokens(text)  # the conftest fake tokenizer


# --- sentences and blocks -----------------------------------------------------


def test_split_sentences():
    assert ingest.split_sentences("One. Two! Three? Four") == ["One.", "Two!", "Three?", "Four"]


class TestSplitIntoBlocks:
    def test_paragraphs_carry_heading_breadcrumb(self):
        text = "# Title\n\nIntro para.\n\n## Sub\n\nLine one\nline two\n\nSecond para."
        assert ingest.split_into_blocks(text) == [
            ("Title", "Intro para."),
            ("Title > Sub", "Line one\nline two"),
            ("Title > Sub", "Second para."),
        ]

    def test_new_heading_replaces_same_level_and_deeper(self):
        text = "# A\n## B\np1\n\n## C\np2\n\n# D\np3"
        sections = [s for s, _ in ingest.split_into_blocks(text)]
        assert sections == ["A > B", "A > C", "D"]

    def test_text_before_any_heading_has_empty_section(self):
        assert ingest.split_into_blocks("Just text.") == [("", "Just text.")]

    def test_code_fence_is_one_atomic_block_even_with_blank_lines(self):
        text = "# A\n\n```\ncode line\n\nmore code\n```\n\nafter"
        blocks = ingest.split_into_blocks(text)
        assert blocks[0] == ("A", "```\ncode line\n\nmore code\n```")
        assert blocks[1] == ("A", "after")

    def test_heading_syntax_inside_fence_is_not_a_heading(self):
        text = "# Real\n\n```\n# not a heading\n```"
        blocks = ingest.split_into_blocks(text)
        assert len(blocks) == 1
        assert blocks[0][0] == "Real"
        assert "# not a heading" in blocks[0][1]


# --- packing ------------------------------------------------------------------


def para(n_chars):
    """A paragraph of exactly n_chars characters with no sentence breaks."""
    return ("x" * (n_chars - 1)) + "."


class TestPackBlocksIntoChunks:
    def test_small_paragraphs_packed_together_under_budget(self):
        blocks = [("S", para(160))] * 3                     # 40 tokens each
        chunks = ingest.pack_blocks_into_chunks(blocks, "a.md", target_tokens=100,
                                                min_tokens=10, overlap_sentences=0)
        assert len(chunks) == 2
        assert all(c.token_count <= 100 for c in chunks)

    def test_never_merges_across_a_heading(self):
        blocks = [("A", "tiny a."), ("B", "tiny b.")]
        chunks = ingest.pack_blocks_into_chunks(blocks, "a.md")
        assert [c.section for c in chunks] == ["A", "B"]

    def test_overlap_carries_last_sentence_into_next_chunk(self):
        p1 = "First alpha sentence here. Second alpha sentence here."   # 18 tokens
        p2 = "First beta sentence here. Second beta sentence here."     # 16 tokens
        # Budget 30: both paragraphs together (34) do not fit, but the carried
        # sentence (10) plus the next paragraph (16) does — so overlap is kept.
        # (An earlier version of this test used a budget of 20, where the
        # overlap could only be carried by breaking the budget. It passed
        # because of the overflow bug, not despite it.)
        chunks = ingest.pack_blocks_into_chunks([("S", p1), ("S", p2)], "a.md",
                                                target_tokens=30, min_tokens=1,
                                                overlap_sentences=1)
        assert len(chunks) == 2
        assert chunks[1].text.startswith("Second alpha sentence here.")
        assert all(c.token_count <= 30 for c in chunks)

    def test_no_overlap_when_disabled(self):
        p1 = "First alpha sentence here. Second alpha sentence here."
        p2 = "First beta sentence here. Second beta sentence here."
        chunks = ingest.pack_blocks_into_chunks([("S", p1), ("S", p2)], "a.md",
                                                target_tokens=20, min_tokens=1,
                                                overlap_sentences=0)
        assert chunks[1].text.startswith("First beta")

    def test_oversized_paragraph_is_split_into_fitting_chunks(self):
        big = " ".join(f"Sentence number {i} is here." for i in range(80))
        chunks = ingest.pack_blocks_into_chunks([("S", big)], "a.md",
                                                target_tokens=100, min_tokens=10)
        assert len(chunks) > 1
        assert all(c.token_count <= 100 for c in chunks)

    def test_indexes_are_contiguous_from_zero(self):
        blocks = [("A", para(400)), ("B", para(400)), ("C", para(400))]
        chunks = ingest.pack_blocks_into_chunks(blocks, "a.md", target_tokens=120, min_tokens=10)
        assert [c.index for c in chunks] == list(range(len(chunks)))

    def test_no_text_is_lost(self):
        blocks = [("S", f"Paragraph {i} has some words in it.") for i in range(30)]
        chunks = ingest.pack_blocks_into_chunks(blocks, "a.md", target_tokens=60,
                                                min_tokens=5, overlap_sentences=0)
        joined = " ".join(c.text for c in chunks)
        for i in range(30):
            assert f"Paragraph {i} has" in joined


# --- oversized splitting: the three fallbacks ---------------------------------


class TestSplitOversized:
    def test_prose_splits_on_sentence_boundaries(self):
        prose = " ".join(f"This is sentence {i}." for i in range(60))
        pieces = ingest._split_oversized(prose, 50)
        assert all(tok(p) <= 50 for p in pieces)
        assert all(p.endswith(".") for p in pieces)

    def test_file_tree_regression_splits_on_lines_not_words(self):
        """
        The bug: a 60-file tree was 60 words — inside a 160 word budget — while
        being ~4x the embedding model's token limit, and got silently truncated.
        """
        tree = "\n".join(f"src/module_{i:02d}/component_file_{i:02d}.py" for i in range(60))
        assert len(tree.split()) == 60
        assert tok(tree) > 200

        pieces = ingest._split_oversized(tree, 200)
        assert len(pieces) > 1
        assert all(tok(p) <= 200 for p in pieces)
        assert "\n".join(pieces).split() == tree.split()   # nothing lost or reordered

    def test_single_line_longer_than_budget_is_hard_cut(self):
        line = " ".join(["word"] * 500)                    # no sentences, no newlines
        pieces = ingest._split_oversized(line, 200)
        assert len(pieces) > 1
        assert all(tok(p) <= 200 for p in pieces)
        assert " ".join(pieces).split() == line.split()


# --- orphan merging -----------------------------------------------------------


class TestMergeOrphans:
    def test_runt_merged_into_neighbour_in_same_section(self):
        chunks = [Chunk("tiny", "a.md", "A", 0), Chunk("y" * 200, "a.md", "A", 1)]
        merged = ingest._merge_orphans(chunks, target_tokens=100, min_tokens=10)
        assert len(merged) == 1
        assert merged[0].text.startswith("tiny")

    def test_not_merged_across_sections(self):
        chunks = [Chunk("tiny", "a.md", "A", 0), Chunk("y" * 200, "a.md", "B", 1)]
        assert len(ingest._merge_orphans(chunks, 100, 10)) == 2

    def test_not_merged_when_result_would_exceed_budget(self):
        chunks = [Chunk("tiny", "a.md", "A", 0), Chunk("y" * 396, "a.md", "A", 1)]
        assert len(ingest._merge_orphans(chunks, 100, 10)) == 2

    def test_indexes_renumbered_after_merge(self):
        chunks = [Chunk("z" * 200, "a.md", "A", 0), Chunk("tiny", "a.md", "A", 1),
                  Chunk("w" * 200, "a.md", "B", 2)]
        merged = ingest._merge_orphans(chunks, 100, 10)
        assert [c.index for c in merged] == [0, 1]


# --- the Chunk ----------------------------------------------------------------


class TestChunk:
    def test_citation_and_id(self):
        c = Chunk("body", "resume.md", "Experience > Acme", 3)
        assert c.citation == "resume.md § Experience > Acme"
        assert c.chunk_id == "resume.md::3"

    def test_citation_without_section_is_filename(self):
        assert Chunk("body", "notes.txt", "", 0).citation == "notes.txt"

    def test_embedding_text_uses_only_deepest_heading_levels(self, monkeypatch):
        monkeypatch.setattr(config, "EMBED_HEADING_LEVELS", 2)
        c = Chunk("body", "a.md", "Doc > Project > What went wrong", 0)
        assert c.embedding_text() == "Project > What went wrong\n\nbody"

    def test_embedding_text_without_section_is_body(self):
        assert Chunk("body", "a.md", "", 0).embedding_text() == "body"

    def test_counts_computed_on_creation(self):
        c = Chunk("four words right here", "a.md", "", 0)
        assert c.word_count == 4
        assert c.token_count == tok("four words right here")


# --- reading documents --------------------------------------------------------


class TestReadDocument:
    @pytest.mark.parametrize("header, kind", [
        (b"%PDF-1.7\nbinary", "a PDF"),
        (b"PK\x03\x04binary", "Word/Office"),
        (b"\xd0\xcf\x11\xe0binary", "old-style Word"),
        (b"{\\rtf1 hello", "RTF"),
    ])
    def test_rejects_binary_formats_despite_extension(self, tmp_path, header, kind):
        path = tmp_path / "resume.md"
        path.write_bytes(header)
        with pytest.raises(ValueError, match=kind):
            ingest.read_document(path)

    def test_rejects_text_that_mostly_fails_to_decode(self, tmp_path):
        path = tmp_path / "weird.txt"
        path.write_bytes(b"\xff\xfe" * 200 + b"abc")
        with pytest.raises(ValueError, match="readable text"):
            ingest.read_document(path)

    def test_accepts_ordinary_utf8_with_accents_and_emoji(self, tmp_path):
        path = tmp_path / "ok.md"
        path.write_text("Café résumé — shipped 🚀", encoding="utf-8")
        assert ingest.read_document(path) == "Café résumé — shipped 🚀"


class TestDocumentDiscovery:
    def test_finds_supported_files_recursively_and_skips_readme(self, tmp_path):
        for name in ["a.md", "b.txt", "UPPER.MD", "nested/c.md", "README.md",
                     "nested/README.md", "x.pdf", "y.docx"]:
            (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / name).write_text("text")
        found = [p.relative_to(tmp_path).as_posix() for p in ingest.iter_document_paths(tmp_path)]
        assert found == sorted(["UPPER.MD", "a.md", "b.txt", "nested/c.md"])

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ingest.iter_document_paths(tmp_path / "nope")

    def test_bad_file_collected_when_skipped_list_given(self, tmp_path):
        (tmp_path / "good.md").write_text("# G\n\nA real paragraph.")
        (tmp_path / "bad.md").write_bytes(b"%PDF-1.7 junk")
        skipped = []
        chunks = ingest.load_and_chunk_all(tmp_path, skipped=skipped)
        assert {c.source for c in chunks} == {"good.md"}
        assert [name for name, _ in skipped] == ["bad.md"]

    def test_bad_file_raises_when_no_skipped_list(self, tmp_path):
        (tmp_path / "bad.md").write_bytes(b"%PDF-1.7 junk")
        with pytest.raises(ValueError):
            ingest.load_and_chunk_all(tmp_path)

    def test_find_unfinished_drafts(self, tmp_path):
        (tmp_path / "done.md").write_text("# Done\n\nAll answered.")
        (tmp_path / "half.md").write_text("# Half\n\n> **TODO:** one?\n> **TODO:** two?")
        (tmp_path / "bad.md").write_bytes(b"%PDF-1.7")
        assert ingest.find_unfinished_drafts(tmp_path) == {"half.md": 2}


def test_chunk_document_end_to_end(tmp_path):
    path = tmp_path / "project.md"
    path.write_text(
        "# Project\n\n## Problem\n\n" + "The problem was real. " * 40 +
        "\n\n## Outcome\n\nIt shipped.\n", encoding="utf-8")
    chunks = ingest.chunk_document(path)
    assert chunks, "expected chunks"
    assert {c.source for c in chunks} == {"project.md"}
    assert all(c.text.strip() for c in chunks)
    assert all(c.estimated_tokens() <= config.EMBEDDING_MAX_TOKENS for c in chunks)
    assert any(c.section == "Project > Outcome" for c in chunks)


def test_overlap_never_pushes_a_chunk_over_budget():
    """
    The sentence carried over from the previous chunk was appended without
    checking the budget, so a long final sentence plus a near-budget paragraph
    produced a chunk past the limit — the same silent-truncation risk the
    token budget exists to prevent.
    """
    p1 = "Hi. " + " ".join(["abcdefgh"] * 9) + "."       # long final sentence
    p2 = " ".join(["abcdefgh"] * 21) + "."               # nearly fills the budget alone
    chunks = ingest.pack_blocks_into_chunks([("S", p1), ("S", p2)], "a.md",
                                            target_tokens=50, min_tokens=1,
                                            overlap_sentences=1)
    assert all(c.token_count <= 50 for c in chunks), [c.token_count for c in chunks]
