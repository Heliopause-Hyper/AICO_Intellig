import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Set


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CURATED_DIR = os.path.join(REPO_ROOT, "results", "curated_datasets")
UNIVERSES_DIR = os.path.join(CURATED_DIR, "universes")
INDEX_PATH = os.path.join(CURATED_DIR, "index.csv")
EXPORT_SUMMARY_PATH = os.path.join(CURATED_DIR, "export_summary.json")
TABLES = {
    "instances": os.path.join(CURATED_DIR, "instances.jsonl"),
    "run_summaries": os.path.join(CURATED_DIR, "run_summaries.jsonl"),
    "runs_light": os.path.join(CURATED_DIR, "runs_light.jsonl"),
    "checkpoints": os.path.join(CURATED_DIR, "checkpoints.jsonl"),
}
UNIVERSES = ("paper", "training", "full_archive")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            yield json.loads(text)


def ensure_int(row: Dict[str, str], key: str) -> int:
    value = row.get(key, "")
    try:
        return int(float(value)) if value not in {"", None} else 0
    except Exception:
        return 0


def bad_experiments_from_summary(summary: dict, experiment_paths: Iterable[str]) -> Set[str]:
    experiments = set(experiment_paths)
    bad = set()
    for item in summary.get("bad_rows", {}).get("examples", []):
        file_path = item.get("file", "")
        for exp in experiments:
            if file_path.startswith(exp + "/") or file_path == exp:
                bad.add(exp)
                break
    return bad


def basename(path: str) -> str:
    return path.rstrip("/").split("/")[-1]


def starts_with_any(text: str, prefixes: Iterable[str]) -> bool:
    return any(text.startswith(prefix) for prefix in prefixes)


def contains_any(text: str, tokens: Iterable[str]) -> bool:
    return any(token in text for token in tokens)


def is_substantive(row: Dict[str, str]) -> bool:
    return any(
        ensure_int(row, key) > 0
        for key in ("runs_lines", "run_summary_lines", "run_checkpoint_lines")
    )


def include_training(row: Dict[str, str]) -> Dict[str, object]:
    path = row["path"]
    name = basename(path)
    reasons: List[str] = []
    if not is_substantive(row):
        reasons.append("No substantive run/checkpoint records.")
    if starts_with_any(name, ("test_",)):
        reasons.append("Test-only directory.")
    if name == "corca_test_v1":
        reasons.append("Small CORCA sanity-check directory.")
    if contains_any(name, ("smoke", "probe")):
        reasons.append("Smoke/probe dataset.")
    include = not reasons
    return {"include": include, "reasons": reasons or ["Substantive experiment or training view."]}


def include_paper(row: Dict[str, str], bad_experiments: Set[str]) -> Dict[str, object]:
    path = row["path"]
    name = basename(path)
    reasons: List[str] = []
    if ensure_int(row, "run_summary_lines") <= 0:
        reasons.append("No canonical run_summary records.")
    if starts_with_any(name, ("_", "test_")):
        reasons.append("Derived or test directory.")
    if name in {"corca_test_v1", "static_ranker_baseline_latest", "sns_v1_full18_i150_r2"}:
        reasons.append("Auxiliary baseline, sanity, or pilot directory.")
    if contains_any(name, ("smoke", "probe")):
        reasons.append("Smoke/probe dataset.")
    if path in bad_experiments:
        reasons.append("Contains malformed JSONL rows in current archive.")
    include = not reasons
    return {"include": include, "reasons": reasons or ["Stable experiment with canonical run summaries."]}


def build_assignments(rows: List[Dict[str, str]], bad_experiments: Set[str]) -> List[Dict[str, object]]:
    assignments = []
    for row in rows:
        path = row["path"]
        training = include_training(row)
        paper = include_paper(row, bad_experiments)
        assignments.append(
            {
                **row,
                "paper_include": int(paper["include"]),
                "paper_reason": "; ".join(paper["reasons"]),
                "training_include": int(training["include"]),
                "training_reason": "; ".join(training["reasons"]),
                "full_archive_include": 1,
                "full_archive_reason": "All curated exportable manifests are retained in the full archive universe.",
            }
        )
    return assignments


def write_assignments(assignments: List[Dict[str, object]]) -> str:
    os.makedirs(UNIVERSES_DIR, exist_ok=True)
    path = os.path.join(UNIVERSES_DIR, "assignments.csv")
    fieldnames = list(assignments[0].keys()) if assignments else []
    write_csv(path, assignments, fieldnames)
    return path


def selected_paths(assignments: List[Dict[str, object]], universe: str) -> Set[str]:
    key = f"{universe}_include"
    return {str(row["path"]) for row in assignments if int(row[key]) == 1}


def filter_index(assignments: List[Dict[str, object]], universe: str) -> List[Dict[str, object]]:
    key = f"{universe}_include"
    return [row for row in assignments if int(row[key]) == 1]


def write_universe_index(universe_dir: str, rows: List[Dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(os.path.join(universe_dir, "index.csv"), rows, fieldnames)


def write_experiment_list(universe_dir: str, paths: Set[str]) -> None:
    with open(os.path.join(universe_dir, "experiment_paths.txt"), "w", encoding="utf-8") as handle:
        for path in sorted(paths):
            handle.write(path + "\n")


def filter_table(input_path: str, output_path: str, allowed_paths: Set[str]) -> int:
    count = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for row in iter_jsonl(input_path):
            if row.get("_source_experiment") not in allowed_paths:
                continue
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_universe_readme(universe_dir: str, universe: str, counts: Dict[str, int], duplicate_base_tables: bool) -> None:
    lines = [
        f"# {universe} Universe",
        "",
        f"Generated at: `{utc_now()}`",
        "",
        "## Scope",
        "",
    ]
    if duplicate_base_tables:
        lines.append("This universe has its own filtered JSONL exports derived from `results/curated_datasets/`.")
    else:
        lines.append("This universe is defined by index and experiment membership only; it reuses the parent curated tables instead of duplicating them.")
    lines.extend(
        [
            "",
            "## Counts",
            "",
        ]
    )
    for key in ("experiments", "instances", "run_summaries", "runs_light", "checkpoints"):
        if key in counts:
            lines.append(f"- `{key}`: `{counts[key]}`")
    with open(os.path.join(universe_dir, "README.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


def write_universe_summary(universe_dir: str, universe: str, counts: Dict[str, int], duplicate_base_tables: bool) -> None:
    payload = {
        "generated_at": utc_now(),
        "universe": universe,
        "counts": counts,
        "duplicates_filtered_tables": duplicate_base_tables,
    }
    with open(os.path.join(universe_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def build_universe(universe: str, assignments: List[Dict[str, object]]) -> Dict[str, int]:
    universe_dir = os.path.join(UNIVERSES_DIR, universe)
    os.makedirs(universe_dir, exist_ok=True)
    rows = filter_index(assignments, universe)
    paths = selected_paths(assignments, universe)
    counts = {"experiments": len(paths)}
    write_universe_index(universe_dir, rows)
    write_experiment_list(universe_dir, paths)
    duplicate = universe in {"paper", "training"}
    if duplicate:
        counts["instances"] = filter_table(TABLES["instances"], os.path.join(universe_dir, "instances.jsonl"), paths)
        counts["run_summaries"] = filter_table(TABLES["run_summaries"], os.path.join(universe_dir, "run_summaries.jsonl"), paths)
        counts["runs_light"] = filter_table(TABLES["runs_light"], os.path.join(universe_dir, "runs_light.jsonl"), paths)
        counts["checkpoints"] = filter_table(TABLES["checkpoints"], os.path.join(universe_dir, "checkpoints.jsonl"), paths)
    write_universe_summary(universe_dir, universe, counts, duplicate)
    write_universe_readme(universe_dir, universe, counts, duplicate)
    return counts


def main() -> None:
    os.makedirs(UNIVERSES_DIR, exist_ok=True)
    export_summary = read_json(EXPORT_SUMMARY_PATH)
    allowed_classes = set((export_summary.get("included_classes") or {}).keys())
    rows = [
        row
        for row in load_csv(INDEX_PATH)
        if row.get("classification") in allowed_classes
    ]
    exp_paths = [row["path"] for row in rows]
    bad_experiments = bad_experiments_from_summary(export_summary, exp_paths)
    assignments = build_assignments(rows, bad_experiments)
    assignments_path = write_assignments(assignments)
    summary = {
        "generated_at": utc_now(),
        "assignments_path": os.path.relpath(assignments_path, REPO_ROOT).replace(os.sep, "/"),
        "bad_experiments": sorted(bad_experiments),
        "universes": {},
    }
    for universe in UNIVERSES:
        summary["universes"][universe] = build_universe(universe, assignments)
    with open(os.path.join(UNIVERSES_DIR, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote assignments for {len(assignments)} experiments")
    for universe in UNIVERSES:
        counts = summary["universes"][universe]
        print(f"- {universe}: {counts['experiments']} experiments")


if __name__ == "__main__":
    main()
