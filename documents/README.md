# Your documents go here

Drop `.txt` or `.md` files into this folder, then run ingestion. Anything you
put here is gitignored — your resume and notes stay on your machine.

## What to add

| File | What it should contain |
|---|---|
| `resume.md` | Your resume as plain text — roles, dates, bullet points |
| `project_<name>.md` | One file per project: the problem, your approach, the tech, the tradeoffs, the outcome |
| `experience_notes.md` | Stories you'd tell in a behavioral round: conflicts, failures, things you'd do differently |

## What makes retrieval work well

The tool can only answer from what you actually wrote down. Two habits help a lot:

- **Write in self-contained paragraphs.** Each paragraph gets retrieved on its
  own, without the ones around it. "We used Postgres because the access pattern
  was read-heavy" retrieves well; "We used it for that reason" does not.
- **Include the specifics you'd be asked about** — numbers, tradeoffs you
  rejected, what broke. Vague write-ups produce vague interview questions.

Use `example_project_writeup.md` as a shape to copy. Delete it once you have
your own files in here.
