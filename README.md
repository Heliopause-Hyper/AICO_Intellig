# AICO-Intellig

AICO-Intellig is the analysis and decision layer built on top of the AICO
execution/archive ecosystem. The stable AICO framework remains in the separate
`AICO1` repository. This repository is for data curation, backend-selection
experiments, sequential stopping, risk-sensitive routing, and related theory
work built from the retained-run archive.

## Repository Scope

This Git repository tracks:

- source code under `src/`
- experiment and export scripts under `scripts/`
- lightweight configs under `training/`
- tests under `tests/` and the top-level `test_*.py`
- documentation under `docs/`
- lightweight result metadata under `results/manifests/`
- curated summary files needed to understand dataset coverage and universes

This Git repository does not track:

- raw or sharded experiment payloads under `results/`
- large curated tables such as `instances.jsonl`, `run_summaries.jsonl`,
  `runs_light.jsonl`, or `checkpoints.jsonl`
- industrial runtime assets under `folderA/`
- local secrets such as `.env`

## Recommended Read Order

1. `docs/DATASET_INVENTORY.md`
2. `results/curated_datasets/README.md`
3. `results/curated_datasets/universes/summary.json`
4. `results/manifests/`
5. `src/` and `scripts/`

The key rule is to use the curated interfaces and universe splits as the
default entry point. Do not treat the raw `results/` tree as the primary API.

## Directory Guide

- `src/`: core simulators, optimizers, CLI, and runtime abstractions
- `scripts/`: data inventory/export builders and paper/theory experiments
- `docs/`: dataset, submission, and theory notes
- `training/`: lightweight dataset-generation and split configs
- `results/manifests/`: canonical small metadata index for experiment bundles
- `results/curated_datasets/universes/`: paper/training/full-archive split
  summaries and path lists
- `folderA/`: CORCA/preciseFZ runtime assets; keep server-local

## Local vs Server Workflow

For local LLM/Codex review, sync this repository plus the tracked lightweight
metadata. When a task needs full JSONL tables or raw run outputs, keep the work
server-side and exchange diffs, summaries, and targeted extracts rather than
copying the entire `results/` tree.

See `AGENTS.md` for the Codex/remote-agent handoff rules.
