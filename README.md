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

## Running it

```bash
python -m rag_buddy
```

That opens a menu: ingest documents, ask a question, or start an interview
session. It shows what's currently indexed and totals up what the session cost
when you quit.

Each step is also runnable on its own, which is the fastest way to debug a
disappointing answer:

```bash
python -m rag_buddy.ingest                  # see how documents get chunked
python -m rag_buddy.store                   # rebuild the index
python -m rag_buddy.retrieve "a question"   # see what gets retrieved
python -m rag_buddy.generate "a question"   # one grounded answer
python -m rag_buddy.interview               # a practice session
```

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

## Pulling in your GitHub repos

A resume bullet is one line. Your repositories hold far more — and the tool
can use both.

```bash
python -m rag_buddy.github --list                    # see your repos
python -m rag_buddy.github owner/repo-one owner/two  # pull them in
python -m rag_buddy.store                            # re-index
```

This indexes each repo's **README, file structure, dependencies, and commit
history** — not source files. That is deliberate: interview questions ask
*why*, and code only records *what*. A chunk of your migration script cannot
say why you merged colour variants the way you did, and indexing thousands of
such chunks would bury the ones that can.

### Turning repos into write-ups

Repo facts make the tool broader, not deeper — the reasoning interviewers ask
about isn't written anywhere. So this drafts a write-up per project, filling in
everything the repo evidences and leaving the rest as pointed questions for you:

```bash
python -m rag_buddy.scaffold
```

> **TODO:** The README notes this is "a clean reconstruction" of the original
> migration. What broke or was lost in the original run that prompted the rebuild?

Then answer the questions one at a time:

```bash
python -m rag_buddy.todos
```

Each question is shown with the sentence it's about. Type your answer and it
gets written into the draft in the right place. `s` skips a question, `d`
deletes one that doesn't apply, and `q` stops. Every answer is saved as you
go, so you can finish a project across several sittings. Once nothing is left
open, it offers to move the file into `documents/` and re-index.

**If you're not sure of an answer, delete the question.** A finished write-up
is the ground truth you get graded against, so a half-remembered detail here
will mark you wrong in practice for telling the truth.

Drafts are **not** indexed until they're moved. An unfinished draft is half
open questions, and indexing it would store those as facts about your career.
If one ever gets moved into `documents/` by hand too early, ingestion warns you.

## Interview mode

Instead of you asking questions, the tool asks *you* — generating interview
questions from your own write-ups, then grading your typed answer against what
your notes actually say.

```bash
python -m rag_buddy.interview                       # random document
python -m rag_buddy.interview project_trailhead.md  # a specific one
```

Each answer gets a verdict (STRONG / ADEQUATE / WEAK) plus what you covered,
what you left out, anything that **contradicts** your notes, and one sharper
sentence you could have said. Things you say that aren't in your notes are
flagged as unverifiable rather than wrong — you may know more than you wrote
down, and that's a prompt to go write it down.

Type `skip` to pass on a question, `quit` to end the session.

## Cost

| Step | Runs where | Cost |
|---|---|---|
| Chunking documents | Your machine | Free |
| Embedding chunks | Your machine (`all-MiniLM-L6-v2`) | Free |
| Storing / searching vectors | Your machine (ChromaDB) | Free |
| Answering a question | Claude Haiku 4.5 | ~$0.0013 |
| Generating interview questions | Claude Haiku 4.5 | ~$0.002 per round of 5 |
| Grading one answer | Claude Sonnet 5 | ~$0.010 |
| Pulling GitHub repos | GitHub API | Free |
| Drafting one write-up | Claude Haiku 4.5 | ~$0.009 |
| Answering draft questions | Your machine | Free |
| Question your docs can't answer | Claude Haiku 4.5 | ~$0.0013 — it's told to say so, not guess |

Only the last three rows cost money. Retrieved context is deliberately kept to
a handful of chunks rather than whole documents, which is what keeps questions
cheap — roughly 770 questions per dollar.

Grading is the one place that doesn't use Haiku. It was measured: Haiku graded
a strong answer as merely "adequate" and manufactured faults to justify it,
while Sonnet 5 graded the same answer correctly. A grader you can't trust is
worse than none. See `EVALUATION_MODEL` in `config.py` — one line to revert.

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite runs in under a second and needs no API key, no network, and no
model downloads.

It is also safe to run on a machine with real data in it. A shared fixture
redirects `documents/`, `drafts/` and `chroma_db/` into a temporary folder for
every test, and replaces the Anthropic client, the GitHub CLI, ChromaDB and
both local models with blockers that fail the test if anything reaches them.
Running the tests can never spend money or touch your documents, drafts or
index.

What it covers:

- **Chunking** — token budgets, all three split fallbacks, overlap, heading
  boundaries, orphan merging, and rejecting binary files renamed to `.md`
- **The TODO editor** — answer placement, atomic saves and backups, resuming
  a session, and refusing to promote a draft with open questions
- **Retrieval** — candidate fetch, reranking order, the per-source diversity cap
- **Generation** — prompt assembly, answering unanswerable questions without
  an API call, and turning API errors into readable messages
- **Interview mode** — question parsing, grading model and pricing, sampling
- **GitHub and scaffolding** — document structure, truncation, and drafts
  never overwriting your answers

Several tests are regression tests for real bugs the suite caught: a chunk
overflowing its budget when overlap was carried over, the last-resort word
split overshooting by the tokenizer's special tokens, and indented interview
questions keeping their `Q:` prefix.

### Integration tests

These run the real embedding model, reranker, tokenizer and ChromaDB. The
Anthropic API and GitHub are still blocked, so they cost nothing. They take
about 20 seconds:

```bash
pytest -m model
```

They check that the pieces agree with the real models, since the chunker's
tokenizer must count exactly like the embedding model's. They build and
rebuild a real index, and walk through complete workflows: drafting a
write-up, answering its questions, promoting it, and finding it again by
search.

They also run a **retrieval benchmark** on a fixed, fictional corpus. It checks
that the right passage still reaches Claude, which is the thing a model or
dependency upgrade can quietly break. Add `-s` to see the per-question report.

That benchmark changed the tool's behaviour. Unanswerable questions used to be
refused for free, using a reranker score threshold calibrated on one corpus. On
the benchmark corpus, that threshold would have refused two questions the
documents *did* answer. The score ranges of the two corpora overlap, so no
single value is safe in general. Free refusal is now off by default: a wrong
refusal hides your real experience, while the cost of asking is a fraction of
a cent. `RERANK_MIN_SCORE` in `config.py` explains how to turn it back on.

## Privacy

Your documents never leave your machine except for the few retrieved chunks
that are sent as context with each question. `documents/` and `chroma_db/` are
gitignored, as is `.env`.

## Project layout

```
rag-study-buddy/
├── documents/          your resume and write-ups (gitignored)
├── examples/           a sample write-up to test with
├── rag_buddy/          the package
│   ├── config.py       paths, model names, and tuning knobs
│   ├── ingest.py       reads documents/ and splits them into chunks
│   ├── store.py        embeds chunks locally and stores them in ChromaDB
│   ├── retrieve.py     finds the top-k chunks for a question
│   ├── generate.py     asks Claude Haiku to answer from those chunks
│   ├── interview.py    generates questions and grades your answers
│   ├── github.py       pulls README / structure / commits from your repos
│   ├── scaffold.py     drafts project write-ups you finish
│   ├── todos.py        walks you through a draft's questions
│   └── cli.py          the menu tying it together
├── drafts/             generated write-ups awaiting your answers (gitignored)
├── chroma_db/          local vector store (generated, gitignored)
├── tests/              the test suite (pytest)
├── requirements.txt
├── requirements-dev.txt   adds pytest
└── .env.example
```

## Status

Complete:

- [x] 1. Project setup
- [x] 2. Ingestion + chunking
- [x] 3. Local embedding + ChromaDB storage
- [x] 4. Retrieval
- [x] 5. Grounded answer generation with citations
- [x] 6. Interview mode
- [ ] 7. CLI menu
