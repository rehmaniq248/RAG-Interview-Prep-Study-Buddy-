"""
Step 2 — Ingestion and chunking.

Reads every .txt / .md file in documents/ and splits it into chunks: small,
self-contained passages that are the actual unit of retrieval later on.

WHY CHUNK AT ALL?
Two independent reasons, and both are hard limits rather than preferences.

1. The embedding model physically cannot read a whole document. It has a
   256-word-piece input window and silently truncates anything longer. Hand it
   your entire resume and it embeds the first paragraph and throws the rest
   away — with no error.

2. One vector has to stand for one idea. An embedding is a single point in
   meaning-space. If a chunk covers your database choice AND your caching
   strategy AND a hiring anecdote, its vector lands at the meaningless average
   of all three and matches no query strongly. Retrieval quality is mostly
   decided here, before any model runs.

Chunking is also the main cost lever in the whole project: chunk size sets how
many tokens each retrieved passage costs you on every question you ever ask.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from . import config


# ---------------------------------------------------------------------------
# The chunk
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """One retrievable passage, plus the metadata needed to cite it."""

    text: str
    source: str          # filename, e.g. "project_trailhead.md"
    section: str         # heading breadcrumb, e.g. "Trailhead > What went wrong"
    index: int           # position within its source document
    word_count: int = field(default=0)

    def __post_init__(self) -> None:
        if not self.word_count:
            self.word_count = len(self.text.split())

    @property
    def citation(self) -> str:
        """Human-readable source label, shown to you under every answer."""
        return f"{self.source} § {self.section}" if self.section else self.source

    @property
    def chunk_id(self) -> str:
        """Stable unique id, so re-ingesting updates rows instead of duplicating them."""
        return f"{self.source}::{self.index}"

    def embedding_text(self) -> str:
        """
        The text we actually embed — the heading breadcrumb prepended to the body.

        WHY: consider a real chunk from a project write-up:

            "My first version computed the estimate using a flat average
             speed of 3 mph. It was badly wrong on steep trails."

        Nothing in those words says which project this is. Ask "what went wrong
        in Trailhead?" and this chunk — the correct answer — may not surface,
        because the word "Trailhead" appears nowhere in it. Prepending
        "Trailhead > What went wrong" puts the subject back into the text being
        embedded, so the chunk becomes findable by the name of the thing it is
        actually about. Cheap fix, large quality gain.
        """
        if not self.section:
            return self.text
        # Deepest N levels only — see EMBED_HEADING_LEVELS in config.
        levels = self.section.split(" > ")[-config.EMBED_HEADING_LEVELS :]
        return f"{' > '.join(levels)}\n\n{self.text}"

    def estimated_tokens(self) -> int:
        """Rough word-piece count. Verified against the real tokenizer in step 3."""
        return int(self.word_count * config.WORDS_TO_TOKENS_RATIO)


# ---------------------------------------------------------------------------
# Reading files
# ---------------------------------------------------------------------------


def iter_document_paths(documents_dir: Path | None = None) -> list[Path]:
    """Every supported file in documents/, sorted so runs are reproducible."""
    directory = documents_dir or config.DOCUMENTS_DIR
    if not directory.is_dir():
        raise FileNotFoundError(f"No documents directory at {directory}")

    return sorted(
        p
        for p in directory.rglob("*")
        if p.is_file()
        and p.suffix.lower() in config.SUPPORTED_EXTENSIONS
        # documents/README.md is our instructions to you, not source material.
        and p.name != "README.md"
    )


def read_document(path: Path) -> str:
    """Read a file as UTF-8, tolerating the odd stray byte rather than crashing."""
    return path.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Splitting text into blocks (paragraphs, tagged with their heading trail)
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE_RE = re.compile(r"^\s*```")

# Naive sentence splitter: break after . ! or ? followed by whitespace.
# It mis-fires on abbreviations ("e.g. this") — acceptable here, because the
# only cost of a bad split is a slightly odd overlap sentence, not lost text.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_RE.split(text.strip()) if s]


def split_into_blocks(text: str) -> list[tuple[str, str]]:
    """
    Split a document into (section_breadcrumb, paragraph) pairs.

    WHY PARAGRAPHS RATHER THAN FIXED-SIZE SLICES — the core design choice here:

    The obvious approach is fixed-size chunking: cut every 800 characters,
    maybe with some overlap. It is simple and it is what most tutorials show.
    It is also blind to meaning. A cut at character 800 lands wherever it
    lands — routinely mid-sentence, splitting

        "I chose PostGIS over raw coordinate arrays because the core query
         is a spatial one"

    into two fragments, one ending at "because". Neither half retrieves well:
    the first states a decision with no reason, the second a reason with no
    decision. The information survives in storage but is destroyed for search.

    Paragraph boundaries are different: they are semantic boundaries that YOU
    already placed while writing. A paragraph is usually one idea, which is
    exactly the unit we want one vector to represent. Using them is free
    structure — the author did the segmentation work already.

    So the strategy is structure-first with a size guard:
      • split on blank lines (paragraphs) and markdown headings
      • pack small neighbouring paragraphs together up to a word budget
      • only fall back to blunt sentence-splitting for a single paragraph
        that is too big on its own
      • never merge across a heading, because a chunk that spans two sections
        cannot be honestly labelled as belonging to either

    Fixed-size chunking would still be the right call for input with no
    structure at all — an interview transcript with no paragraph breaks, say.
    Resumes and project write-ups have plenty of structure, so we use it.
    """
    blocks: list[tuple[str, str]] = []
    heading_stack: list[str] = []   # current heading trail, by markdown level
    buffer: list[str] = []
    in_code_fence = False

    def current_section() -> str:
        return " > ".join(heading_stack)

    def flush() -> None:
        """Emit whatever paragraph text has accumulated, then reset."""
        if buffer:
            paragraph = "\n".join(buffer).strip()
            if paragraph:
                blocks.append((current_section(), paragraph))
            buffer.clear()

    for line in text.splitlines():
        # Fenced code blocks are held together as one atomic paragraph — a
        # snippet cut in half is worse than useless.
        if _FENCE_RE.match(line):
            in_code_fence = not in_code_fence
            buffer.append(line)
            if not in_code_fence:
                flush()
            continue

        if in_code_fence:
            buffer.append(line)
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            # Replace the trail from this level down: an H2 ends the previous
            # H2 and every H3 beneath it.
            del heading_stack[level - 1 :]
            heading_stack.append(title)
            continue

        if line.strip():
            buffer.append(line)
        else:
            flush()   # blank line = paragraph break

    flush()
    return blocks


# ---------------------------------------------------------------------------
# Packing blocks into chunks
# ---------------------------------------------------------------------------


def _split_oversized(paragraph: str, target_words: int) -> list[str]:
    """
    Break a single over-long paragraph on sentence boundaries.

    Only reached when one paragraph exceeds the budget by itself. Sentence
    boundaries are the least damaging place to cut once we have to cut.
    """
    pieces: list[str] = []
    current: list[str] = []
    count = 0

    for sentence in split_sentences(paragraph):
        words = len(sentence.split())
        if current and count + words > target_words:
            pieces.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += words

    if current:
        pieces.append(" ".join(current))
    return pieces or [paragraph]


def _merge_orphans(
    chunks: list[Chunk], target_words: int, min_words: int
) -> list[Chunk]:
    """
    Fold undersized chunks into a neighbour.

    Greedy packing can still emit a runt — a two-line paragraph that happens to
    sit just before an oversized one gets flushed on its own. A six-word chunk
    ("Jane Doe / Software Engineer") is a poor retrieval unit: too little
    context to answer anything, but still a vector competing for a top-k slot
    it doesn't deserve.

    So we make one pass afterwards and merge each runt into an adjacent chunk,
    preferring the one that follows it, provided they share a section and the
    result still fits the budget. A runt with no eligible neighbour is kept —
    dropping text is never the right answer.
    """
    if not chunks:
        return chunks

    merged: list[Chunk] = []
    for chunk in chunks:
        prev = merged[-1] if merged else None
        can_merge_back = (
            prev is not None
            and (prev.word_count < min_words or chunk.word_count < min_words)
            and prev.source == chunk.source
            and prev.section == chunk.section
            and prev.word_count + chunk.word_count <= target_words
        )
        if can_merge_back:
            merged[-1] = Chunk(
                text=f"{prev.text}\n\n{chunk.text}",
                source=prev.source,
                section=prev.section,
                index=prev.index,
            )
        else:
            merged.append(chunk)

    # Re-number so chunk_id stays contiguous and stable per document.
    return [
        Chunk(text=c.text, source=c.source, section=c.section, index=i)
        for i, c in enumerate(merged)
    ]


def pack_blocks_into_chunks(
    blocks: list[tuple[str, str]],
    source: str,
    target_words: int | None = None,
    min_words: int | None = None,
    overlap_sentences: int | None = None,
) -> list[Chunk]:
    """
    Greedily pack paragraphs into chunks that sit under the word budget.

    Two rules do the real work:
      • accumulate paragraphs until the next one would blow the budget
      • never carry a chunk across a heading boundary

    The second rule is what keeps citations honest. If a chunk contained the
    tail of "Architecture" and the head of "What went wrong", any label we put
    on it would be wrong for half its content — and Step 5 shows that label to
    you as the source of the answer.
    """
    target_words = target_words or config.CHUNK_TARGET_WORDS
    min_words = min_words or config.CHUNK_MIN_WORDS
    if overlap_sentences is None:
        overlap_sentences = config.CHUNK_OVERLAP_SENTENCES

    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_words = 0
    buffer_section = ""

    def flush() -> None:
        nonlocal buffer, buffer_words
        if not buffer:
            return
        text = "\n\n".join(buffer).strip()
        if text:
            chunks.append(
                Chunk(text=text, source=source, section=buffer_section, index=len(chunks))
            )
        buffer, buffer_words = [], 0

    def overlap_tail() -> list[str]:
        """Last N sentences of the chunk just emitted, to seed the next one."""
        if not overlap_sentences or not chunks:
            return []
        sentences = split_sentences(chunks[-1].text)
        return sentences[-overlap_sentences:] if sentences else []

    for section, paragraph in blocks:
        # A heading change is a hard boundary — close the current chunk.
        if section != buffer_section:
            flush()
            buffer_section = section

        words = len(paragraph.split())

        # Case 1: this paragraph alone busts the budget. Emit what we have,
        # then sentence-split the paragraph into its own chunks.
        if words > target_words:
            flush()
            for piece in _split_oversized(paragraph, target_words):
                chunks.append(
                    Chunk(
                        text=piece,
                        source=source,
                        section=buffer_section,
                        index=len(chunks),
                    )
                )
            continue

        # Case 2: adding this paragraph would bust the budget — close the
        # current chunk first, then start a new one seeded with the overlap.
        if buffer and buffer_words + words > target_words:
            flush()
            tail = overlap_tail()
            if tail:
                buffer.append(" ".join(tail))
                buffer_words = sum(len(s.split()) for s in tail)

        buffer.append(paragraph)
        buffer_words += words

        # Case 3: comfortably over the minimum and near the budget — emit now
        # rather than letting the next paragraph force an awkward split.
        if buffer_words >= target_words - min_words:
            flush()

    flush()
    return _merge_orphans(chunks, target_words, min_words)


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------


def chunk_document(path: Path) -> list[Chunk]:
    """Read one file and return its chunks."""
    return pack_blocks_into_chunks(split_into_blocks(read_document(path)), source=path.name)


def load_and_chunk_all(documents_dir: Path | None = None) -> list[Chunk]:
    """Read and chunk every document. This is what step 3 will embed."""
    chunks: list[Chunk] = []
    for path in iter_document_paths(documents_dir):
        chunks.extend(chunk_document(path))
    return chunks


# ---------------------------------------------------------------------------
# Preview: `python -m rag_buddy.ingest`
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Print the chunks without embedding anything.

    Worth actually reading the output. Nearly every "the answers are bad"
    problem in a RAG system is visible right here, in chunks that are cut in
    odd places or that mix two unrelated topics.
    """
    paths = iter_document_paths()
    if not paths:
        print(f"No .txt or .md files found in {config.DOCUMENTS_DIR}")
        print("Drop your resume and project write-ups in there, then re-run.")
        return

    total_words = 0
    grand_total = 0

    for path in paths:
        chunks = chunk_document(path)
        grand_total += len(chunks)
        print(f"\n{'=' * 78}\n{path.name} — {len(chunks)} chunks\n{'=' * 78}")

        for chunk in chunks:
            total_words += chunk.word_count
            over = " ⚠ OVER LIMIT" if chunk.estimated_tokens() > config.EMBEDDING_MAX_TOKENS else ""
            preview = " ".join(chunk.text.split())
            if len(preview) > 220:
                preview = preview[:220] + "…"
            print(f"\n  [{chunk.index}] {chunk.section or '(no heading)'}")
            print(f"      {chunk.word_count} words ≈ {chunk.estimated_tokens()} tokens{over}")
            print(f"      {preview}")

    print(f"\n{'=' * 78}")
    print(f"{grand_total} chunks from {len(paths)} document(s), {total_words} words total.")
    print(f"Budget: {config.CHUNK_TARGET_WORDS} words/chunk "
          f"(≈{int(config.CHUNK_TARGET_WORDS * config.WORDS_TO_TOKENS_RATIO)} "
          f"of the model's {config.EMBEDDING_MAX_TOKENS} token limit).")
    print("Cost of this step: $0.00 — nothing here calls an API.")


if __name__ == "__main__":
    main()
