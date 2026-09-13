import csv
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
DOCS_DIR = os.path.join(REPO_ROOT, "docs")
MANIFEST_DIR = os.path.join(RESULTS_DIR, "manifests")
CURATED_DIR = os.path.join(RESULTS_DIR, "curated_datasets")
INVENTORY_PATH = os.path.join(DOCS_DIR, "DATASET_INVENTORY.md")
INDEX_CSV_PATH = os.path.join(CURATED_DIR, "index.csv")
INDEX_JSONL_PATH = os.path.join(CURATED_DIR, "index.jsonl")
MANIFEST_SUMMARY_PATH = os.path.join(MANIFEST_DIR, "manifest_summary.json")

PRIMARY_DATA_FILES = [
    "dataset.jsonl",
    "problem_instance.jsonl",
    "runs.jsonl",
    "run_summary.jsonl",
    "iteration_event.jsonl",
    "run_checkpoint.jsonl",
    "optimizer_run.jsonl",
    "optimization_history.db",
]
OPTIONAL_META_FILES = [
    "config_used.json",
    "full_sweep_config.json",
    "stats.json",
    "invalid_instances.jsonl",
    "run_summary.json",
    "merge_summary.json",
    "filter_summary.json",
    "bucket_summary.json",
]
DISCOVERY_FILES = set(PRIMARY_DATA_FILES + OPTIONAL_META_FILES)
ROOT_ONLY_REPORT_PATTERNS = (".csv", ".png", ".txt", ".md")
SKIP_DIR_NAMES = {"__pycache__", "runs", "manifests", "curated_datasets"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def relpath(path: str) -> str:
    return os.path.relpath(path, REPO_ROOT).replace(os.sep, "/")


def list_dirnames(path: str) -> List[str]:
    return sorted(
        entry.name
        for entry in os.scandir(path)
        if entry.is_dir(follow_symlinks=False)
    )


def list_filenames(path: str) -> List[str]:
    return sorted(
        entry.name
        for entry in os.scandir(path)
        if entry.is_file(follow_symlinks=False)
    )


def is_shard_dir(name: str) -> bool:
    return name.startswith("shard_")


def count_lines(path: str) -> int:
    total = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            total += chunk.count(b"\n")
    return total


def file_size(path: str) -> int:
    try:
        return int(os.path.getsize(path))
    except OSError:
        return 0


def read_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def choose_paths_for_file(root: str, shard_dirs: List[str], filename: str) -> Tuple[List[str], str]:
    root_path = os.path.join(root, filename)
    shard_paths = [
        os.path.join(root, shard_name, filename)
        for shard_name in shard_dirs
        if os.path.exists(os.path.join(root, shard_name, filename))
    ]
    if os.path.exists(root_path):
        if shard_paths and filename not in {"dataset.jsonl", "runs.jsonl", "config_used.json", "run_summary.json"}:
            return shard_paths, "shard_aggregate"
        return [root_path], "root"
    if shard_paths:
        return shard_paths, "shard_aggregate"
    return [], "missing"


def aggregate_record_counts(root: str, shard_dirs: List[str]) -> Tuple[Dict[str, int], Dict[str, str], Dict[str, List[str]]]:
    counts: Dict[str, int] = {}
    sources: Dict[str, str] = {}
    canonical_paths: Dict[str, List[str]] = {}
    for filename in PRIMARY_DATA_FILES + OPTIONAL_META_FILES:
        paths, source = choose_paths_for_file(root, shard_dirs, filename)
        sources[filename] = source
        canonical_paths[filename] = [relpath(path) for path in paths]
        if filename.endswith(".jsonl"):
            counts[filename] = sum(count_lines(path) for path in paths)
        elif filename.endswith(".db"):
            counts[filename] = len(paths)
        elif filename.endswith(".json"):
            counts[filename] = len(paths)
        else:
            counts[filename] = len(paths)
    return counts, sources, canonical_paths


def classify_experiment(rel_dir: str, root_files: List[str], shard_dirs: List[str]) -> str:
    name = os.path.basename(rel_dir)
    if name.startswith("_"):
        return "derived_view"
    if "submission_snapshot" in rel_dir or "algo_landscape" in rel_dir or "paper_stats" in rel_dir:
        return "report_bundle"
    if "curve_policy_eval" in rel_dir:
        return "model_bundle"
    if any(filename in root_files for filename in ("run_summary.jsonl", "iteration_event.jsonl", "problem_instance.jsonl")):
        return "raw_experiment"
    if shard_dirs:
        return "sharded_experiment"
    if any(filename in root_files for filename in ("dataset.jsonl", "runs.jsonl")):
        return "lightweight_dataset"
    return "auxiliary"


def canonical_strategy(root_files: List[str], shard_dirs: List[str]) -> str:
    if not shard_dirs:
        return "single_dir"
    if "dataset.jsonl" in root_files or "runs.jsonl" in root_files:
        return "root_merged_core__shard_rich_tables"
    return "shard_only"


def extract_config_summary(root: str, shard_dirs: List[str]) -> Dict[str, Optional[object]]:
    config_paths, _ = choose_paths_for_file(root, shard_dirs, "config_used.json")
    fallback_paths, _ = choose_paths_for_file(root, shard_dirs, "full_sweep_config.json")
    config = None
    source_path = None
    for path in config_paths + fallback_paths:
        config = read_json(path)
        if config is not None:
            source_path = path
            break
    if not config:
        return {
            "source_path": None,
            "source_script": None,
            "seed_count": None,
            "max_iterations": None,
            "time_limit_s": None,
            "optimizer_total_config": None,
        }
    budget = config.get("budget") or {}
    meta = config.get("meta") or {}
    seeds = config.get("seeds")
    return {
        "source_path": relpath(source_path) if source_path else None,
        "source_script": meta.get("script"),
        "seed_count": len(seeds) if isinstance(seeds, list) else None,
        "max_iterations": budget.get("max_iterations"),
        "time_limit_s": budget.get("time_limit_s"),
        "optimizer_total_config": meta.get("optimizer_total_config"),
    }


def collect_notes(root_files: List[str], shard_dirs: List[str], sources: Dict[str, str]) -> List[str]:
    notes: List[str] = []
    if shard_dirs and ("dataset.jsonl" in root_files or "runs.jsonl" in root_files):
        notes.append("Root keeps merged dataset/runs while richer tables remain in shard directories.")
    if shard_dirs and sources.get("run_summary.jsonl") == "missing":
        notes.append("No canonical run_summary.jsonl is available at root or across all shards.")
    if "optimization_history.db" in root_files:
        notes.append("SQLite history is preserved alongside JSONL views.")
    if "merge_summary.json" in root_files or "filter_summary.json" in root_files or "bucket_summary.json" in root_files:
        notes.append("Directory acts as a derived training or filtered view rather than a raw sweep.")
    return notes


def discover_experiment_dirs(results_dir: str) -> List[str]:
    discovered: List[str] = []
    for root, dirs, files in os.walk(results_dir, topdown=True):
        dirs[:] = [
            name
            for name in dirs
            if name not in SKIP_DIR_NAMES and not name.startswith(".")
        ]
        if root == results_dir:
            continue
        shard_dirs = sorted(name for name in dirs if is_shard_dir(name))
        file_set = set(files)
        if shard_dirs or (file_set & set(PRIMARY_DATA_FILES)):
            discovered.append(root)
            if shard_dirs:
                dirs[:] = [name for name in dirs if not is_shard_dir(name)]
    return sorted(discovered)


def build_manifest(root: str) -> Dict[str, object]:
    root_files = list_filenames(root)
    dirnames = list_dirnames(root)
    shard_dirs = [name for name in dirnames if is_shard_dir(name)]
    counts, sources, canonical_paths = aggregate_record_counts(root, shard_dirs)
    rel_dir = relpath(root)
    manifest = {
        "manifest_version": 1,
        "generated_at": utc_now(),
        "path": rel_dir,
        "name": os.path.basename(root),
        "classification": classify_experiment(rel_dir, root_files, shard_dirs),
        "is_sharded": bool(shard_dirs),
        "shard_count": len(shard_dirs),
        "canonical_strategy": canonical_strategy(root_files, shard_dirs),
        "root_files": root_files,
        "shard_dirs": shard_dirs,
        "canonical_paths": canonical_paths,
        "canonical_sources": sources,
        "record_counts": counts,
        "config": extract_config_summary(root, shard_dirs),
        "notes": collect_notes(root_files, shard_dirs, sources),
    }
    return manifest


def manifest_filename(rel_dir: str) -> str:
    return rel_dir.replace("/", "__") + ".json"


def write_json(path: str, payload: object) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def classify_top_level_entry(path: str) -> str:
    name = os.path.basename(path)
    if os.path.isdir(path):
        files = list_filenames(path)
        dirs = list_dirnames(path)
        if any(filename.endswith(".joblib") for filename in files) or name.startswith("curve_policy_eval"):
            return "model_bundle"
        if name in {"sweep_logs", "logs"}:
            return "log_bundle"
        if any(filename.endswith(ROOT_ONLY_REPORT_PATTERNS) for filename in files):
            return "report_bundle"
        if any(is_shard_dir(dirname) for dirname in dirs) or (set(files) & DISCOVERY_FILES):
            return "experiment_bundle"
        return "container"
    if os.path.isfile(path):
        if path.endswith(".joblib"):
            return "model_file"
        if path.endswith(".log"):
            return "log_file"
        if path.endswith(".json"):
            return "config_or_summary_file"
        return "other_file"
    return "other"


def walk_size_and_count(path: str) -> Tuple[int, int]:
    total_bytes = 0
    total_files = 0
    for root, dirs, files in os.walk(path):
        dirs[:] = [name for name in dirs if name not in {"__pycache__"}]
        for filename in files:
            file_path = os.path.join(root, filename)
            total_bytes += file_size(file_path)
            total_files += 1
    return total_bytes, total_files


def summarize_precisefz() -> List[Dict[str, object]]:
    base = os.path.join(REPO_ROOT, "folderA", "preciseFZ")
    if not os.path.isdir(base):
        return []
    rows: List[Dict[str, object]] = []
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name)
        if not os.path.exists(path):
            continue
        if os.path.isdir(path):
            bytes_used, files = walk_size_and_count(path)
            rows.append(
                {
                    "name": name,
                    "kind": "dir",
                    "files": files,
                    "size_mb": round(bytes_used / (1024.0 * 1024.0), 2),
                }
            )
        else:
            rows.append(
                {
                    "name": name,
                    "kind": "file",
                    "files": 1,
                    "size_mb": round(file_size(path) / (1024.0 * 1024.0), 2),
                }
            )
    return rows


def write_index(manifests: List[Dict[str, object]]) -> None:
    os.makedirs(CURATED_DIR, exist_ok=True)
    fieldnames = [
        "path",
        "name",
        "classification",
        "is_sharded",
        "shard_count",
        "canonical_strategy",
        "dataset_lines",
        "problem_instance_lines",
        "runs_lines",
        "run_summary_lines",
        "iteration_event_lines",
        "run_checkpoint_lines",
        "optimizer_run_lines",
        "has_db",
        "source_script",
        "seed_count",
        "max_iterations",
        "time_limit_s",
        "optimizer_total_config",
    ]
    rows = []
    for manifest in manifests:
        counts = manifest["record_counts"]
        config = manifest["config"]
        row = {
            "path": manifest["path"],
            "name": manifest["name"],
            "classification": manifest["classification"],
            "is_sharded": int(bool(manifest["is_sharded"])),
            "shard_count": manifest["shard_count"],
            "canonical_strategy": manifest["canonical_strategy"],
            "dataset_lines": counts.get("dataset.jsonl", 0),
            "problem_instance_lines": counts.get("problem_instance.jsonl", 0),
            "runs_lines": counts.get("runs.jsonl", 0),
            "run_summary_lines": counts.get("run_summary.jsonl", 0),
            "iteration_event_lines": counts.get("iteration_event.jsonl", 0),
            "run_checkpoint_lines": counts.get("run_checkpoint.jsonl", 0),
            "optimizer_run_lines": counts.get("optimizer_run.jsonl", 0),
            "has_db": int(bool(counts.get("optimization_history.db", 0))),
            "source_script": config.get("source_script"),
            "seed_count": config.get("seed_count"),
            "max_iterations": config.get("max_iterations"),
            "time_limit_s": config.get("time_limit_s"),
            "optimizer_total_config": config.get("optimizer_total_config"),
        }
        rows.append(row)
    with open(INDEX_CSV_PATH, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(INDEX_JSONL_PATH, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_top_level_results() -> List[Dict[str, object]]:
    rows = []
    for name in sorted(os.listdir(RESULTS_DIR)):
        if name in {"manifests", "curated_datasets"}:
            continue
        path = os.path.join(RESULTS_DIR, name)
        role = classify_top_level_entry(path)
        if os.path.isdir(path):
            bytes_used, files = walk_size_and_count(path)
            rows.append(
                {
                    "name": name,
                    "kind": role,
                    "files": files,
                    "size_mb": round(bytes_used / (1024.0 * 1024.0), 2),
                }
            )
        else:
            rows.append(
                {
                    "name": name,
                    "kind": role,
                    "files": 1,
                    "size_mb": round(file_size(path) / (1024.0 * 1024.0), 2),
                }
            )
    return rows


def format_table(rows: Iterable[Dict[str, object]], columns: List[Tuple[str, str]]) -> List[str]:
    rows = list(rows)
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    out = [header, sep]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |")
    return out


def write_inventory(manifests: List[Dict[str, object]]) -> None:
    top_level_rows = summarize_top_level_results()
    precise_rows = summarize_precisefz()
    class_counter = Counter(str(manifest["classification"]) for manifest in manifests)
    total_counts = Counter()
    for manifest in manifests:
        for filename, value in manifest["record_counts"].items():
            if filename.endswith(".jsonl"):
                total_counts[filename] += int(value)

    key_rows = []
    for manifest in manifests:
        counts = manifest["record_counts"]
        if not any(
            counts.get(filename, 0)
            for filename in (
                "dataset.jsonl",
                "problem_instance.jsonl",
                "runs.jsonl",
                "run_summary.jsonl",
            )
        ):
            continue
        key_rows.append(
            {
                "path": manifest["path"],
                "class": manifest["classification"],
                "shards": manifest["shard_count"],
                "dataset": counts.get("dataset.jsonl", 0),
                "problem": counts.get("problem_instance.jsonl", 0),
                "runs": counts.get("runs.jsonl", 0),
                "summary": counts.get("run_summary.jsonl", 0),
                "checkpoints": counts.get("run_checkpoint.jsonl", 0),
            }
        )
    key_rows.sort(key=lambda row: (row["class"], row["path"]))

    lines: List[str] = []
    lines.append("# AICO-Intellig Dataset Inventory")
    lines.append("")
    lines.append(f"Generated at: `{utc_now()}`")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "This inventory is a non-destructive organization layer. It does not move or rewrite raw results. "
        "Instead, it records canonical file locations, aggregate record counts, and directory roles for the current repository state."
    )
    lines.append("")
    lines.append("## Canonical Interfaces")
    lines.append("")
    lines.append("- `problem_instance.jsonl`: canonical instance-level structural metadata.")
    lines.append("- `run_summary.jsonl`: canonical run-level summary table.")
    lines.append("- `iteration_event.jsonl`: canonical fine-grained trajectory events.")
    lines.append("- `run_checkpoint.jsonl`: materialized checkpoint cache derived from iteration events.")
    lines.append("- `dataset.jsonl`: training-oriented instance view with labels and candidate aggregates.")
    lines.append("- `runs.jsonl`: lightweight run view used by ranker-style pipelines.")
    lines.append("")
    lines.append("## Current Scale")
    lines.append("")
    lines.append(f"- Discovered data-bearing experiment directories: `{len(manifests)}`")
    lines.append(
        "- Aggregate JSONL record counts: "
        + ", ".join(
            f"`{filename}` = `{total_counts.get(filename, 0)}`"
            for filename in (
                "dataset.jsonl",
                "problem_instance.jsonl",
                "runs.jsonl",
                "run_summary.jsonl",
                "iteration_event.jsonl",
                "run_checkpoint.jsonl",
            )
        )
    )
    lines.append(
        "- Manifest classes: "
        + ", ".join(f"`{name}` = `{count}`" for name, count in sorted(class_counter.items()))
    )
    lines.append("")
    lines.append("## Results Root Overview")
    lines.append("")
    lines.extend(
        format_table(
            top_level_rows,
            [
                ("name", "Entry"),
                ("kind", "Role"),
                ("files", "Files"),
                ("size_mb", "Size (MB)"),
            ],
        )
    )
    lines.append("")
    lines.append("## Canonical Experiment Index")
    lines.append("")
    lines.extend(
        format_table(
            key_rows,
            [
                ("path", "Path"),
                ("class", "Class"),
                ("shards", "Shards"),
                ("dataset", "dataset"),
                ("problem", "problem"),
                ("runs", "runs"),
                ("summary", "run_summary"),
                ("checkpoints", "checkpoints"),
            ],
        )
    )
    lines.append("")
    if precise_rows:
        lines.append("## Industrial Runtime Assets")
        lines.append("")
        lines.append(
            "The `folderA/preciseFZ` tree is not part of the same JSONL dataset interface. "
            "It contains runtime assets, databanks, workspaces, and outputs for CORCA/preciseFZ integration."
        )
        lines.append("")
        lines.extend(
            format_table(
                precise_rows,
                [
                    ("name", "Entry"),
                    ("kind", "Kind"),
                    ("files", "Files"),
                    ("size_mb", "Size (MB)"),
                ],
            )
        )
        lines.append("")
    lines.append("## Recommended Read Order")
    lines.append("")
    lines.append("- For dataset auditing: start from `results/curated_datasets/index.csv`.")
    lines.append("- For an individual sweep: open the matching manifest under `results/manifests/`.")
    lines.append("- For instance-level features: prefer `problem_instance.jsonl` over `dataset.jsonl`.")
    lines.append("- For run-level statistics: prefer `run_summary.jsonl` over `runs.jsonl` whenever available.")
    lines.append("- For early-trajectory policy work: treat `iteration_event.jsonl` as source and `run_checkpoint.jsonl` as cache.")
    lines.append("")
    os.makedirs(DOCS_DIR, exist_ok=True)
    with open(INVENTORY_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


def build_manifest_summary(manifests: List[Dict[str, object]]) -> Dict[str, object]:
    total_counts = defaultdict(int)
    for manifest in manifests:
        for filename, value in manifest["record_counts"].items():
            if filename.endswith(".jsonl"):
                total_counts[filename] += int(value)
    return {
        "generated_at": utc_now(),
        "manifest_count": len(manifests),
        "classes": Counter(str(manifest["classification"]) for manifest in manifests),
        "aggregate_jsonl_counts": dict(sorted(total_counts.items())),
        "inventory_path": relpath(INVENTORY_PATH),
        "index_csv_path": relpath(INDEX_CSV_PATH),
        "index_jsonl_path": relpath(INDEX_JSONL_PATH),
    }


def main() -> None:
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    for name in os.listdir(MANIFEST_DIR):
        if name.endswith(".json"):
            os.remove(os.path.join(MANIFEST_DIR, name))
    experiment_dirs = discover_experiment_dirs(RESULTS_DIR)
    manifests = [build_manifest(path) for path in experiment_dirs]
    for manifest in manifests:
        path = os.path.join(MANIFEST_DIR, manifest_filename(str(manifest["path"])))
        write_json(path, manifest)
    write_index(manifests)
    write_inventory(manifests)
    write_json(MANIFEST_SUMMARY_PATH, build_manifest_summary(manifests))
    print(f"Wrote {len(manifests)} manifests")
    print(f"Wrote inventory to {relpath(INVENTORY_PATH)}")
    print(f"Wrote curated index to {relpath(INDEX_CSV_PATH)} and {relpath(INDEX_JSONL_PATH)}")


if __name__ == "__main__":
    main()
