# RAG Interview Prep Study Buddy

A retrieval-augmented question-answering tool for your own background. Point it
at your resume and project write-ups, then ask it questions about your
experience or have it quiz you like an interviewer would.

Answers are grounded in your actual documents and cite which file they came
from, so it can't invent a project you never worked on.

**Everything runs locally except the final answer.** Embeddings and vector
search happen on your machine with open-source models, at no cost. Only the
last step — turning retrieved text into an answer — calls the Anthropic API,
using Claude Haiku, the cheapest current model.

## How it works

```
documents/*.md ──▶ chunk ──▶ embed (local) ──▶ ChromaDB (local, on disk)
                                                     │
                              your question ──▶ embed (local)
                                                     │
                                                     ▼
                                          top-k similar chunks
                                                     │
                                                     ▼
                                    Claude Haiku ──▶ grounded answer + citations
                                          (the only paid step)
```

## Setup

Requires Python 3.10 or newer.

```bash
git clone <this repo>
cd rag-study-buddy

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then paste your Anthropic API key into .env
```

Get an API key at [console.anthropic.com](https://console.anthropic.com/settings/keys).

Then drop your resume and project write-ups into `documents/` as `.txt` or
`.md` files. See [`documents/README.md`](documents/README.md) for what to write.

## Inspecting your chunks

Before embedding anything, see how your documents get split:

```bash
python -m rag_buddy.ingest
```

This prints every chunk with its heading trail and token estimate, and costs
nothing. It is worth reading: most "the answers are bad" problems in a RAG
system are visible right here, as chunks cut in odd places or mixing two
unrelated topics.

## Building the index

Once your documents are in `documents/`:

```bash
python -m rag_buddy.store
```

This embeds every chunk on your machine and writes them to `chroma_db/`. The
first run downloads the ~90 MB embedding model; after that it works offline.
Re-run it whenever you edit your documents — it rebuilds from scratch, so no
stale text survives.

It also verifies every chunk against the model's real tokenizer and warns you
if any would be silently truncated.

## Inspecting retrieval

See which chunks a question pulls up, without generating an answer:

```bash
python -m rag_buddy.retrieve "why did I choose that database"
```

Use this whenever an answer disappoints you — it separates the two failure
modes. If the right passage isn't in the list, the problem is retrieval or
chunking and no amount of prompting will fix it. If the right passage *is*
there and the answer was still poor, the problem is in generation.

## Asking a question

```bash
python -m rag_buddy.generate "why did I choose that database"
```

Prints the answer with `[n]` citations, the sources those numbers refer to,
and the exact token count and dollar cost of the request.

To see the assembled prompt and a cost estimate **without spending anything**:

```bash
python -m rag_buddy.generate "your question" --dry-run
```

## Cost

| Step | Runs where | Cost |
|---|---|---|
| Chunking documents | Your machine | Free |
| Embedding chunks | Your machine (`all-MiniLM-L6-v2`) | Free |
| Storing / searching vectors | Your machine (ChromaDB) | Free |
| Generating an answer | Anthropic API (Claude Haiku 4.5) | Fractions of a cent per question |

Only the last row costs money. The retrieved context is deliberately kept to a
handful of chunks rather than whole documents, which is what keeps each
question cheap.

## Privacy

Your documents never leave your machine except for the few retrieved chunks
that are sent as context with each question. `documents/` and `chroma_db/` are
gitignored, as is `.env`.

## Project layout

```
rag-study-buddy/
├── documents/          your resume and write-ups (gitignored)
├── rag_buddy/          the package
│   ├── config.py       paths, model names, and tuning knobs
│   ├── ingest.py       reads documents/ and splits them into chunks
│   ├── store.py        embeds chunks locally and stores them in ChromaDB
│   ├── retrieve.py     finds the top-k chunks for a question
│   └── generate.py     asks Claude Haiku to answer from those chunks
├── chroma_db/          local vector store (generated, gitignored)
├── requirements.txt
└── .env.example
```

## Status

Built step by step. Currently complete:

- [x] 1. Project setup
- [x] 2. Ingestion + chunking
- [x] 3. Local embedding + ChromaDB storage
- [x] 4. Retrieval
- [x] 5. Grounded answer generation with citations
- [ ] 6. Interview mode
- [ ] 7. CLI menu
