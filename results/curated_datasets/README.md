# Curated Dataset Exports

Generated at: `2026-09-05T07:33:19+00:00`

## Scope

These files provide a normalized export layer on top of the raw results tree.
The raw directories remain unchanged. Each exported row carries provenance fields
that identify the source experiment, manifest, file, and original line number.

## Included Manifest Classes

- `lightweight_dataset`: `3` manifests
- `raw_experiment`: `26` manifests
- `sharded_experiment`: `18` manifests

## Tables

- `instances.jsonl`: canonical instance table. Uses `problem_instance.jsonl` when available and falls back to `dataset.jsonl` only for lightweight directories.
- `run_summaries.jsonl`: canonical run summary export from `run_summary.jsonl`.
- `runs_light.jsonl`: lightweight run export from `runs.jsonl`.
- `checkpoints.jsonl`: checkpoint export from `run_checkpoint.jsonl`.
- `export_summary.json`: aggregate counts for the exported layer.

## Provenance Fields

- `_source_experiment`
- `_source_manifest`
- `_source_manifest_class`
- `_source_kind`
- `_source_file`
- `_source_line`
