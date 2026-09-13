# AICO-Intellig Agent Guide

This file is the operating interface for Codex and other code agents working on
`AICO-Intellig`.

## Mission

- Continue development on the AICO-Intellig layer.
- Treat `AICO1` as a separate, largely stable repository.
- Prioritize data consistency, analysis interfaces, backend selection,
  sequential stopping, risk-sensitive policy, and temporal generalization.

## First Read

Read these files before proposing non-trivial changes:

1. `README.md`
2. `docs/DATASET_INVENTORY.md`
3. `results/curated_datasets/README.md`
4. `results/curated_datasets/universes/summary.json`
5. relevant files in `results/manifests/`

Then move into the code:

6. `src/`
7. `scripts/`
8. `training/`
9. `tests/` and top-level `test_*.py`

## Preferred Data Interfaces

Use these as the default interfaces:

- `results/manifests/*.json` for experiment-level structure
- `results/curated_datasets/index.csv` and `index.jsonl` for export coverage
- `results/curated_datasets/universes/{paper,training,full_archive}/`
  summaries for split-aware work
- `docs/DATASET_INVENTORY.md` for global counts and directory roles

Conceptually, the canonical tables are:

- instance level: `problem_instance.jsonl`
- run summary level: `run_summary.jsonl`
- lightweight run level: `runs.jsonl`
- checkpoint cache: `run_checkpoint.jsonl`

In this repository, the large materialized exports are intentionally not tracked
by Git. If you need them, ask the server-side agent to query or summarize them.

## Do Not Do

- Do not scan the full `results/` tree as a first step.
- Do not assume `paper`, `training`, and `full_archive` universes are
  interchangeable.
- Do not add bulky result payloads or runtime assets to Git.
- Do not modify `folderA/` unless the task is explicitly about CORCA runtime
  integration.
- Do not rewrite stable AICO1 terminology such as `AICO-IR`,
  `capability contract`, or `ready/clarify/block`.

## Collaboration Contract

When Codex works locally and another agent works on the server:

1. Codex reads code and proposes the change set.
2. The server-side agent applies edits, runs commands, and reports outcomes.
3. Exchange:
   - target files
   - concrete diffs
   - executed commands
   - test results
   - tracebacks or log excerpts
4. Reconcile decisions against the current server state before the next round.

## Expected Output Style

- Make minimal, explicit patches.
- Prefer curated interfaces over raw directory traversal.
- State which universe is being used: `paper`, `training`, or `full_archive`.
- If a task depends on untracked large tables, request a targeted extract rather
  than assuming local availability.
