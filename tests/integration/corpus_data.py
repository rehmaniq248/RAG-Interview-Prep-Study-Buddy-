"""
A fixed, entirely FICTIONAL corpus for integration tests.

Shaped like the real thing — a resume with short bullet paragraphs, a long
prose project write-up, and a generated GitHub repository document with a file
tree — so the chunker and retrieval face the same structures they meet in real
use. No real person, company, or project is described.
"""

import shutil
from pathlib import Path

from rag_buddy import config, github

EXAMPLE_WRITEUP = config.PROJECT_ROOT / "examples" / "example_project_writeup.md"

RESUME = """# Sample Candidate

sample.candidate@example.com | Example City

## Education

### State University — B.S. Computer Science

Expected 2027. Coursework: Databases, Machine Learning, Distributed Systems.

## Experience

### Acme Rail Systems — Data Engineering Intern

Rebuilt the nightly ticket-sales ETL job in Apache Airflow, cutting its runtime from 6 hours to 40 minutes by replacing row-by-row inserts with bulk COPY loads.

Added data-quality checks to the pipeline that caught around 1,200 duplicate fare records per week before they reached the finance dashboard.

Migrated three years of track sensor logs from CSV dumps into partitioned Parquet files on S3, shrinking storage by 70%.

### Globex Analytics — Machine Learning Research Assistant

Trained a gradient-boosted churn prediction model on 90,000 customer records, reaching 0.87 AUC on a held-out test set.

Presented feature-importance findings to the retention team, which led them to revise their discount policy for at-risk customers.

## Projects

### StudyLoop

A spaced-repetition flashcard app built with React Native, with offline sync backed by SQLite so reviews work without a connection.

### Wildfire Smoke Forecaster

An LSTM model that forecasts PM2.5 air-quality levels 48 hours ahead from satellite imagery and weather data, with a mean absolute error of 6.2 µg/m³.

## Technical Skills

Languages: Python, TypeScript, Go.

Databases: PostgreSQL, DuckDB.
"""

_AREAS = ["parsing", "matching", "reporting", "ledger", "cli"]
_KINDS = ["reader", "normaliser", "validator", "rules", "helpers", "models", "formatters"]

LEDGER_FACTS = {
    "name": "ledger-sync",
    "full_name": "sample-candidate/ledger-sync",
    "description": "Reconciles bank CSV exports against an accounting ledger.",
    "languages": {"Python": 41230},
    "created": "2026-03-01",
    "updated": "2026-05-20",
    "readme": (
        "# ledger-sync\n\n"
        "Reconciles bank statement CSV exports against a double-entry accounting "
        "ledger and flags anything that does not match.\n\n"
        "## How matching works\n\n"
        "Each bank transaction is matched to a ledger entry by amount and date, then "
        "by payee name using fuzzy matching with a 0.9 similarity threshold. Split "
        "transactions, where one bank payment covers several ledger entries, are "
        "matched by searching for subsets of entries that sum to the payment amount.\n\n"
        "## Usage\n\n"
        "```bash\n# reconcile one month\npython -m ledger_sync bank.csv ledger.csv --report out.html\n```\n"
    ),
    # Path-heavy on purpose: one "word" per line, many tokens per line.
    "tree": ["README.md", "requirements.txt", "pyproject.toml"] + [
        f"ledger_sync/{area}/{kind}_{i:02d}.py"
        for area in _AREAS for i, kind in enumerate(_KINDS * 2)
    ],
    "commits": [
        "Handle split transactions by subset-sum search",
        "Fix timezone drift in posted dates (bank exports local time, ledger stores UTC)",
        "Add fuzzy payee matching with rapidfuzz",
        "Write HTML reconciliation report",
        "Initial CSV parser for bank exports",
    ],
    "manifests": {"requirements.txt": "rapidfuzz==3.9.0\npandas==2.2.2\n"},
    "private": False,
}

# (question, a phrase that appears only in the passage that answers it)
ANSWERABLE = [
    ("How did you speed up the nightly ETL pipeline?", "6 hours to 40 minutes"),
    ("Tell me about catching data quality problems", "1,200 duplicate"),
    ("What machine learning model did you train and how well did it perform?", "0.87 AUC"),
    ("Describe a mobile app you built", "React Native"),
    ("Have you worked on time series forecasting?", "PM2.5"),
    ("Why did you choose PostGIS?", "bounding box"),
    ("What was your caching strategy and what did it cost you?", "20 minutes"),
    ("What went wrong with the hiking time estimate?", "Naismith"),
    ("How does the bank transaction matching work?", "0.9 similarity"),
    ("Which databases have you used?", "DuckDB"),
    ("How did you fix timezone problems in the ledger tool?", "timezone drift"),
    ("How many people used the route planning app?", "400 monthly active users"),
]

UNANSWERABLE = [
    "What is your experience with Kubernetes?",
    "Tell me about your time working at Google.",
    "Describe your PhD research on protein folding.",
    "How many years of Rust experience do you have?",
    "What did you do during your military service?",
]


def install_corpus(documents_dir: Path) -> None:
    documents_dir.mkdir(parents=True, exist_ok=True)
    (documents_dir / "resume.md").write_text(RESUME, encoding="utf-8")
    shutil.copy(EXAMPLE_WRITEUP, documents_dir / "example_project_writeup.md")
    repo_dir = documents_dir / "github"
    repo_dir.mkdir(exist_ok=True)
    (repo_dir / "repo_ledger-sync.md").write_text(
        github.facts_to_markdown(LEDGER_FACTS), encoding="utf-8")
