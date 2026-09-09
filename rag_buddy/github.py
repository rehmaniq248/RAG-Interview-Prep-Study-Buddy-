"""
Step 8a — Pulling facts out of your GitHub repos.

WHY NOT JUST INDEX THE SOURCE CODE
Because interview questions ask *why* and source code only records *what*.
A chunk of your migration script cannot answer "why did you merge colour
variants instead of treating them as separate products" — that reasoning was
never written down anywhere in the repo. Indexing file bodies would balloon
the store from tens of chunks to thousands, none of which answer the questions
you actually get asked, while diluting the chunks that do.

So we take the parts of a repo that DO carry interview signal:

  README        the closest thing to a write-up you already have
  structure     the file tree — architecture at a glance
  dependencies  requirements.txt / package.json is your real stack
  commits       a timeline of what you built and what you fixed
  metadata      languages, size, dates

Everything here is fetched with the `gh` CLI, so it uses the login you already
have and works for your private repos too. No API cost — GitHub's API is free.
"""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

from . import config

# Where generated repo documents land. Inside documents/ so they are indexed,
# and gitignored along with everything else you put there.
GITHUB_DIR = config.DOCUMENTS_DIR / "github"

# Manifests worth quoting verbatim: they state your actual stack.
MANIFESTS = [
    "requirements.txt", "pyproject.toml", "package.json", "Gemfile",
    "go.mod", "Cargo.toml", "pom.xml", "build.gradle",
]

# Caps. A repo is unbounded; a useful document is not.
MAX_TREE_ENTRIES = 60
MAX_COMMITS = 60
MAX_README_CHARS = 12_000


def _gh(*args: str) -> str | None:
    """Run a gh command, returning None if it fails (missing repo, no access)."""
    try:
        result = subprocess.run(
            ["gh", *args], capture_output=True, text=True, timeout=30
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def require_gh() -> None:
    """Fail early and clearly if the GitHub CLI is missing or logged out."""
    if not shutil.which("gh"):
        raise RuntimeError(
            "The GitHub CLI (`gh`) is not installed.\n"
            "  macOS:  brew install gh\n"
            "  other:  https://cli.github.com\n"
            "Then run `gh auth login`."
        )
    if _gh("auth", "status") is None:
        raise RuntimeError("`gh` is installed but not logged in. Run: gh auth login")


def list_repos(owner: str | None = None) -> list[dict]:
    """Every repo you can see, newest first."""
    target = [owner] if owner else []
    out = _gh("repo", "list", *target, "--limit", "100", "--json",
              "name,nameWithOwner,description,primaryLanguage,updatedAt,isPrivate")
    if out is None:
        raise RuntimeError("Could not list repositories. Is `gh` logged in?")
    repos = json.loads(out)
    return sorted(repos, key=lambda r: r["updatedAt"], reverse=True)


def _demote_headings(markdown: str) -> str:
    """
    Push every README heading down one level.

    The generated document uses "## README" as a section, so a README with its
    own "# Title" would otherwise jump back to the top level and break the
    heading breadcrumb that chunking relies on for citations.
    """
    lines = []
    in_fence = False
    for line in markdown.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence and line.startswith("#"):
            line = "#" + line
        lines.append(line)
    return "\n".join(lines)


def fetch_repo_facts(name_with_owner: str) -> dict:
    """Gather everything we care about from one repo."""
    meta_raw = _gh("api", f"repos/{name_with_owner}")
    if meta_raw is None:
        raise RuntimeError(f"Could not read {name_with_owner} — check the name and your access.")
    meta = json.loads(meta_raw)
    branch = meta.get("default_branch", "main")

    readme = ""
    readme_raw = _gh("api", f"repos/{name_with_owner}/readme", "--jq", ".content")
    if readme_raw:
        try:
            readme = base64.b64decode(readme_raw.strip()).decode("utf-8", errors="replace")
        except Exception:
            readme = ""

    tree: list[str] = []
    tree_raw = _gh("api", f"repos/{name_with_owner}/git/trees/{branch}?recursive=1",
                   "--jq", '[.tree[] | select(.type=="blob") | .path] | .[]')
    if tree_raw:
        tree = [p for p in tree_raw.splitlines() if p.strip()]

    commits: list[str] = []
    commits_raw = _gh("api", f"repos/{name_with_owner}/commits?per_page={MAX_COMMITS}",
                      "--jq", ".[].commit.message")
    if commits_raw:
        # Keep only the subject line of each message, drop trailer noise.
        seen = set()
        for message in commits_raw.split("\n\n"):
            subject = message.strip().splitlines()[0].strip() if message.strip() else ""
            if (subject and subject not in seen
                    and not subject.startswith(("Co-Authored-By:", "Merge branch",
                                                "Merge pull request"))):
                seen.add(subject)
                commits.append(subject)

    manifests: dict[str, str] = {}
    for filename in MANIFESTS:
        if filename in tree:
            raw = _gh("api", f"repos/{name_with_owner}/contents/{filename}", "--jq", ".content")
            if raw:
                try:
                    manifests[filename] = base64.b64decode(raw.strip()).decode(
                        "utf-8", errors="replace")
                except Exception:
                    pass

    langs = json.loads(_gh("api", f"repos/{name_with_owner}/languages") or "{}")

    return {
        "name": meta["name"],
        "full_name": name_with_owner,
        "description": meta.get("description") or "",
        "languages": langs,
        "created": (meta.get("created_at") or "")[:10],
        "updated": (meta.get("pushed_at") or "")[:10],
        "readme": readme[:MAX_README_CHARS],
        "tree": tree,
        "commits": commits,
        "manifests": manifests,
        "private": meta.get("private", False),
    }


def facts_to_markdown(facts: dict) -> str:
    """
    Render the facts as a document shaped for chunking.

    Headings matter: chunking never merges across one, and the breadcrumb
    becomes the citation. "GrantMatcher > Commit history" is a useful thing to
    see under an answer.
    """
    langs = ", ".join(f"{k} ({v:,} bytes)" for k, v in facts["languages"].items()) or "not detected"
    parts = [
        f"# {facts['name']} (GitHub repository)",
        "",
        "## Repository overview",
        "",
        f"Repository: {facts['full_name']} "
        f"({'private' if facts['private'] else 'public'}).",
        "",
        f"Description: {facts['description'] or 'none given'}",
        "",
        f"Languages: {langs}.",
        "",
        f"Created {facts['created']}, last pushed {facts['updated']}. "
        f"{len(facts['tree'])} files, {len(facts['commits'])} commits recorded.",
    ]

    if facts["readme"]:
        parts += ["", "## README", "", _demote_headings(facts["readme"]).strip()]

    if facts["tree"]:
        shown = facts["tree"][:MAX_TREE_ENTRIES]
        extra = len(facts["tree"]) - len(shown)
        parts += ["", "## Repository structure", "",
                  "The files in this repository:", "",
                  "```", "\n".join(shown),
                  (f"... and {extra} more files" if extra > 0 else ""), "```"]

    if facts["manifests"]:
        parts += ["", "## Dependencies and stack", ""]
        for filename, content in facts["manifests"].items():
            body = content.strip()
            if len(body) > 2000:
                body = body[:2000] + "\n..."
            parts += [f"Declared in {filename}:", "", "```", body, "```", ""]

    if facts["commits"]:
        parts += ["", "## Commit history", "",
                  "Commit subjects, most recent first. These record what was "
                  "built and what was fixed over the life of the project.", ""]
        parts += [f"- {subject}" for subject in facts["commits"]]

    return "\n".join(parts) + "\n"


def sync_repos(names: list[str]) -> list[Path]:
    """Fetch each repo and write it as a markdown document."""
    require_gh()
    GITHUB_DIR.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for name in names:
        print(f"  fetching {name} …", end=" ", flush=True)
        try:
            facts = fetch_repo_facts(name)
        except RuntimeError as exc:
            print(f"skipped ({exc})")
            continue
        path = GITHUB_DIR / f"repo_{facts['name'].strip('-').replace(' ', '_')}.md"
        path.write_text(facts_to_markdown(facts), encoding="utf-8")
        written.append(path)
        print(f"ok ({len(facts['readme'].split())} README words, "
              f"{len(facts['commits'])} commits, {len(facts['tree'])} files)")
    return written


def main() -> None:
    args = sys.argv[1:]

    if not args or args[0] in {"-l", "--list"}:
        require_gh()
        repos = list_repos()
        print(f"\n{len(repos)} repositories:\n")
        for r in repos:
            lang = (r["primaryLanguage"] or {}).get("name", "—")
            print(f"  {r['nameWithOwner']:<52} {lang:<12} {r['updatedAt'][:10]}")
        print("\nFetch some with:")
        print("  python -m rag_buddy.github owner/repo-one owner/repo-two")
        print("\nThen re-index with: python -m rag_buddy.store")
        return

    written = sync_repos(args)
    print(f"\nWrote {len(written)} document(s) to {GITHUB_DIR}")
    print("Cost: $0.00 — GitHub's API is free.")
    print("Now re-index:  python -m rag_buddy.store")


if __name__ == "__main__":
    main()
