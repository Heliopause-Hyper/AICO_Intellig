import json
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
MANIFEST_DIR = os.path.join(RESULTS_DIR, "manifests")
CURATED_DIR = os.path.join(RESULTS_DIR, "curated_datasets")

INSTANCES_PATH = os.path.join(CURATED_DIR, "instances.jsonl")
RUN_SUMMARIES_PATH = os.path.join(CURATED_DIR, "run_summaries.jsonl")
RUNS_LIGHT_PATH = os.path.join(CURATED_DIR, "runs_light.jsonl")
CHECKPOINTS_PATH = os.path.join(CURATED_DIR, "checkpoints.jsonl")
SUMMARY_PATH = os.path.join(CURATED_DIR, "export_summary.json")
README_PATH = os.path.join(CURATED_DIR, "README.md")

EXPORTABLE_CLASSES = {"raw_experiment", "sharded_experiment", "lightweight_dataset"}
SUMMARY_FILES = {"manifest_summary.json"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def relpath(path: str) -> str:
    return os.path.relpath(path, REPO_ROOT).replace(os.sep, "/")


def abs_from_rel(repo_rel: str) -> str:
    return os.path.join(REPO_ROOT, repo_rel.replace("/", os.sep))


def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: str, bad_rows: Optional[List[dict]] = None) -> Iterator[Tuple[int, dict]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                yield line_number, json.loads(text)
            except Exception as exc:
                if bad_rows is not None:
                    bad_rows.append(
                        {
                            "file": relpath(path),
                            "line": line_number,
                            "error": str(exc),
                            "preview": text[:200],
                        }
                    )
                continue


def load_manifests() -> List[Tuple[str, dict]]:
    manifests: List[Tuple[str, dict]] = []
    for name in sorted(os.listdir(MANIFEST_DIR)):
        if not name.endswith(".json") or name in SUMMARY_FILES:
            continue
        path = os.path.join(MANIFEST_DIR, name)
        manifest = read_json(path)
        if manifest.get("classification") not in EXPORTABLE_CLASSES:
            continue
        manifests.append((path, manifest))
    return manifests


def first_path(paths: List[str]) -> Optional[str]:
    return paths[0] if paths else None


def add_provenance(row: dict, manifest_path: str, manifest: dict, source_path: str, line_number: int, source_kind: str) -> dict:
    out = dict(row)
    out["_source_experiment"] = manifest["path"]
    out["_source_manifest"] = relpath(manifest_path)
    out["_source_manifest_class"] = manifest["classification"]
    out["_source_kind"] = source_kind
    out["_source_file"] = source_path
    out["_source_line"] = line_number
    return out


def export_instances(
    manifests: Iterable[Tuple[str, dict]], output_path: str, bad_rows: List[dict]
) -> Dict[str, int]:
    stats = Counter()
    seen_keys = set()
    with open(output_path, "w", encoding="utf-8") as out:
        for manifest_path, manifest in manifests:
            canonical_paths = manifest["canonical_paths"]
            problem_paths = canonical_paths.get("problem_instance.jsonl") or []
            dataset_paths = canonical_paths.get("dataset.jsonl") or []
            if problem_paths:
                for repo_rel in problem_paths:
                    abs_path = abs_from_rel(repo_rel)
                    for line_number, row in iter_jsonl(abs_path, bad_rows=bad_rows):
                        key = (manifest["path"], row.get("instance_id"), "problem_instance")
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                        out.write(
                            json.dumps(
                                add_provenance(
                                    row,
                                    manifest_path,
                                    manifest,
                                    repo_rel,
                                    line_number,
                                    "problem_instance",
                                ),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        stats["rows"] += 1
                        stats["problem_instance_rows"] += 1
                continue
            for repo_rel in dataset_paths:
                abs_path = abs_from_rel(repo_rel)
                for line_number, row in iter_jsonl(abs_path, bad_rows=bad_rows):
                    instance_id = row.get("instance_id")
                    key = (manifest["path"], instance_id, "dataset_fallback")
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    fallback_row = dict(row)
                    fallback_row["_instance_source"] = "dataset_fallback"
                    out.write(
                        json.dumps(
                            add_provenance(
                                fallback_row,
                                manifest_path,
                                manifest,
                                repo_rel,
                                line_number,
                                "dataset_fallback",
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    stats["rows"] += 1
                    stats["dataset_fallback_rows"] += 1
    stats["unique_keys"] = len(seen_keys)
    return dict(stats)


def export_generic_jsonl(
    manifests: Iterable[Tuple[str, dict]],
    output_path: str,
    canonical_name: str,
    key_fields: Tuple[str, ...],
    source_kind: str,
    bad_rows: List[dict],
) -> Dict[str, int]:
    stats = Counter()
    seen_keys = set()
    with open(output_path, "w", encoding="utf-8") as out:
        for manifest_path, manifest in manifests:
            for repo_rel in manifest["canonical_paths"].get(canonical_name) or []:
                abs_path = abs_from_rel(repo_rel)
                for line_number, row in iter_jsonl(abs_path, bad_rows=bad_rows):
                    key = tuple([manifest["path"]] + [row.get(field) for field in key_fields])
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    out.write(
                        json.dumps(
                            add_provenance(
                                row,
                                manifest_path,
                                manifest,
                                repo_rel,
                                line_number,
                                source_kind,
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    stats["rows"] += 1
    stats["unique_keys"] = len(seen_keys)
    return dict(stats)


def write_summary(summary: dict) -> None:
    with open(SUMMARY_PATH, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_readme(summary: dict) -> None:
    lines = [
        "# Curated Dataset Exports",
        "",
        f"Generated at: `{summary['generated_at']}`",
        "",
        "## Scope",
        "",
        "These files provide a normalized export layer on top of the raw results tree.",
        "The raw directories remain unchanged. Each exported row carries provenance fields",
        "that identify the source experiment, manifest, file, and original line number.",
        "",
        "## Included Manifest Classes",
        "",
    ]
    for name in sorted(summary["included_classes"].keys()):
        lines.append(f"- `{name}`: `{summary['included_classes'][name]}` manifests")
    lines.extend(
        [
            "",
            "## Tables",
            "",
            "- `instances.jsonl`: canonical instance table. Uses `problem_instance.jsonl` when available and falls back to `dataset.jsonl` only for lightweight directories.",
            "- `run_summaries.jsonl`: canonical run summary export from `run_summary.jsonl`.",
            "- `runs_light.jsonl`: lightweight run export from `runs.jsonl`.",
            "- `checkpoints.jsonl`: checkpoint export from `run_checkpoint.jsonl`.",
            "- `export_summary.json`: aggregate counts for the exported layer.",
            "",
            "## Provenance Fields",
            "",
            "- `_source_experiment`",
            "- `_source_manifest`",
            "- `_source_manifest_class`",
            "- `_source_kind`",
            "- `_source_file`",
            "- `_source_line`",
            "",
        ]
    )
    with open(README_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


def main() -> None:
    os.makedirs(CURATED_DIR, exist_ok=True)
    manifests = load_manifests()
    class_counter = Counter(manifest["classification"] for _, manifest in manifests)
    bad_rows: List[dict] = []
    summary = {
        "generated_at": utc_now(),
        "manifest_count": len(manifests),
        "included_classes": dict(sorted(class_counter.items())),
        "outputs": {},
    }
    summary["outputs"]["instances"] = export_instances(manifests, INSTANCES_PATH, bad_rows)
    summary["outputs"]["run_summaries"] = export_generic_jsonl(
        manifests,
        RUN_SUMMARIES_PATH,
        "run_summary.jsonl",
        ("run_id",),
        "run_summary",
        bad_rows,
    )
    summary["outputs"]["runs_light"] = export_generic_jsonl(
        manifests,
        RUNS_LIGHT_PATH,
        "runs.jsonl",
        ("run_id",),
        "runs_light",
        bad_rows,
    )
    summary["outputs"]["checkpoints"] = export_generic_jsonl(
        manifests,
        CHECKPOINTS_PATH,
        "run_checkpoint.jsonl",
        ("checkpoint_id",),
        "run_checkpoint",
        bad_rows,
    )
    summary["bad_rows"] = {"count": len(bad_rows), "examples": bad_rows[:20]}
    write_summary(summary)
    write_readme(summary)
    print(f"Exported curated tables from {len(manifests)} manifests")
    for name, stats in summary["outputs"].items():
        print(f"- {name}: {stats.get('rows', 0)} rows")


if __name__ == "__main__":
    main()
