from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Set


REPO_ROOT = "/home/ycl/AICO-Intellig"
DEFAULT_INSTANCES = os.path.join(REPO_ROOT, "results", "curated_datasets", "instances.jsonl")
DEFAULT_RUN_SUMMARIES = os.path.join(REPO_ROOT, "results", "curated_datasets", "run_summaries.jsonl")
DEFAULT_MANIFESTS = os.path.join(REPO_ROOT, "results", "manifests")
DEFAULT_OUT = os.path.join(REPO_ROOT, "results", "complexity_study", "archive_event_level_audit.json")
DEFAULT_FAMILIES = ["sns_flow_family", "advection_v1", "blackscholes_v1", "heat_time_v1", "sns_v3"]


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_jsonl(path: str, bad_rows: List[Dict[str, Any]] | None = None) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                yield json.loads(text)
            except Exception as exc:
                if bad_rows is not None and len(bad_rows) < 50:
                    bad_rows.append(
                        {
                            "path": _relpath(path),
                            "line": line_number,
                            "error": str(exc),
                            "preview": text[:200],
                        }
                    )
                continue


def _typed_config(row: Dict[str, Any]) -> Dict[str, Any]:
    typed = row.get("typed_config_json")
    return dict(typed) if isinstance(typed, dict) else {}


def _canonical_family(row: Dict[str, Any]) -> str:
    typed = _typed_config(row)
    family = (
        typed.get("family_version")
        or row.get("problem_type")
        or typed.get("template_name")
        or row.get("template_name")
        or "unknown"
    )
    return str(family)


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _relpath(path: str) -> str:
    return os.path.relpath(path, REPO_ROOT).replace(os.sep, "/")


def _load_instance_index(instances_path: str, families: Set[str]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for row in _iter_jsonl(instances_path):
        family = _canonical_family(row)
        if family not in families:
            continue
        instance_id = str(row.get("instance_id") or "")
        if not instance_id:
            continue
        index[instance_id] = {
            "family": family,
            "decision_dim": _safe_int(row.get("decision_dim") or _typed_config(row).get("decision_dim")),
            "variable_name_json": list(row.get("variable_name_json") or _typed_config(row).get("variable_name_json") or []),
            "problem_type": row.get("problem_type"),
            "source_experiment": row.get("_source_experiment"),
        }
    return index


def _load_target_runs(
    run_summaries_path: str, instance_index: Dict[str, Dict[str, Any]]
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Set[str]]]:
    run_index: Dict[str, Dict[str, Any]] = {}
    experiment_to_runs: Dict[str, Set[str]] = defaultdict(set)
    for row in _iter_jsonl(run_summaries_path):
        instance_id = str(row.get("instance_id") or "")
        instance_meta = instance_index.get(instance_id)
        if instance_meta is None:
            continue
        run_id = str(row.get("run_id") or "")
        if not run_id:
            continue
        source_experiment = str(row.get("_source_experiment") or "")
        run_index[run_id] = {
            "run_id": run_id,
            "instance_id": instance_id,
            "family": instance_meta["family"],
            "decision_dim": instance_meta["decision_dim"],
            "variable_name_json": instance_meta["variable_name_json"],
            "num_evals": _safe_int(row.get("num_evals")),
            "failure_count": _safe_int(row.get("failure_count")),
            "timeout_count": _safe_int(row.get("timeout_count")),
            "final_feasible_flag": bool(row.get("final_feasible_flag")) if row.get("final_feasible_flag") is not None else None,
            "source_experiment": source_experiment,
            "source_file": row.get("_source_file"),
        }
        if source_experiment:
            experiment_to_runs[source_experiment].add(run_id)
    return run_index, experiment_to_runs


def _load_manifest_iteration_paths(manifests_dir: str, experiments: Set[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for name in sorted(os.listdir(manifests_dir)):
        if not name.endswith(".json") or name == "manifest_summary.json":
            continue
        path = os.path.join(manifests_dir, name)
        manifest = _read_json(path)
        experiment_path = str(manifest.get("path") or "")
        if experiment_path not in experiments:
            continue
        out[experiment_path] = [
            os.path.join(REPO_ROOT, rel.replace("/", os.sep))
            for rel in (manifest.get("canonical_paths", {}).get("iteration_event.jsonl") or [])
        ]
    return out


def _empty_run_stats() -> Dict[str, Any]:
    return {
        "event_count": 0,
        "unique_eval_indices": set(),
        "max_eval_index": 0,
        "monotone_eval_index": True,
        "first_eval_index": None,
        "last_eval_index": None,
        "x_json_present_count": 0,
        "x_dim_match_count": 0,
        "solver_status": Counter(),
        "feasible_true_count": 0,
        "new_best_count": 0,
        "missing_x_examples": [],
        "bad_dim_examples": [],
    }


def _scan_iteration_events(
    iteration_paths: Dict[str, List[str]], run_index: Dict[str, Dict[str, Any]], bad_rows: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    run_stats: Dict[str, Dict[str, Any]] = defaultdict(_empty_run_stats)
    for experiment_path, paths in iteration_paths.items():
        target_runs = {run_id for run_id, meta in run_index.items() if meta["source_experiment"] == experiment_path}
        if not target_runs:
            continue
        for path in paths:
            for row in _iter_jsonl(path, bad_rows=bad_rows):
                run_id = str(row.get("run_id") or "")
                if run_id not in target_runs:
                    continue
                stats = run_stats[run_id]
                stats["event_count"] += 1
                eval_index = _safe_int(row.get("eval_index"))
                if eval_index is not None:
                    stats["unique_eval_indices"].add(eval_index)
                    stats["max_eval_index"] = max(stats["max_eval_index"], eval_index)
                    if stats["first_eval_index"] is None:
                        stats["first_eval_index"] = eval_index
                    last_eval = stats["last_eval_index"]
                    if last_eval is not None and eval_index < last_eval:
                        stats["monotone_eval_index"] = False
                    stats["last_eval_index"] = eval_index
                x_json = row.get("x_json")
                expected_dim = run_index[run_id]["decision_dim"]
                if isinstance(x_json, list):
                    stats["x_json_present_count"] += 1
                    if expected_dim is not None and len(x_json) == expected_dim:
                        stats["x_dim_match_count"] += 1
                    elif len(stats["bad_dim_examples"]) < 3:
                        stats["bad_dim_examples"].append(
                            {
                                "eval_index": eval_index,
                                "observed_dim": len(x_json),
                                "expected_dim": expected_dim,
                            }
                        )
                elif len(stats["missing_x_examples"]) < 3:
                    stats["missing_x_examples"].append({"eval_index": eval_index, "x_json": x_json})
                solver_status = str(row.get("solver_status") or "")
                if solver_status:
                    stats["solver_status"][solver_status] += 1
                if row.get("feasible_flag") is True:
                    stats["feasible_true_count"] += 1
                if row.get("is_new_best") is True:
                    stats["new_best_count"] += 1
    return run_stats


def _family_summary(
    family: str,
    run_ids: List[str],
    run_index: Dict[str, Dict[str, Any]],
    run_stats: Dict[str, Dict[str, Any]],
    iteration_paths: Dict[str, List[str]],
) -> Dict[str, Any]:
    total_runs = len(run_ids)
    runs_with_events = 0
    event_total = 0
    x_present_total = 0
    x_match_total = 0
    monotone_count = 0
    contiguous_count = 0
    num_eval_match_count = 0
    max_eval_match_count = 0
    first_eval_is_one_count = 0
    solver_counter: Counter[str] = Counter()
    experiments: Set[str] = set()
    example_bad_runs: List[Dict[str, Any]] = []

    for run_id in run_ids:
        meta = run_index[run_id]
        stats = run_stats.get(run_id)
        experiments.add(meta["source_experiment"])
        if not stats or stats["event_count"] == 0:
            if len(example_bad_runs) < 5:
                example_bad_runs.append({"run_id": run_id, "issue": "missing_iteration_events"})
            continue

        runs_with_events += 1
        event_total += stats["event_count"]
        x_present_total += stats["x_json_present_count"]
        x_match_total += stats["x_dim_match_count"]
        solver_counter.update(stats["solver_status"])

        unique_evals = stats["unique_eval_indices"]
        num_evals = meta["num_evals"]
        max_eval = stats["max_eval_index"]
        contiguous = bool(unique_evals) and min(unique_evals) == 1 and len(unique_evals) == max_eval

        if stats["monotone_eval_index"]:
            monotone_count += 1
        if contiguous:
            contiguous_count += 1
        if stats["first_eval_index"] == 1:
            first_eval_is_one_count += 1
        if num_evals is not None and stats["event_count"] == num_evals:
            num_eval_match_count += 1
        if num_evals is not None and max_eval == num_evals:
            max_eval_match_count += 1

        if len(example_bad_runs) < 5:
            issues: List[str] = []
            if not stats["monotone_eval_index"]:
                issues.append("non_monotone_eval_index")
            if not contiguous:
                issues.append("non_contiguous_eval_index")
            if num_evals is not None and stats["event_count"] != num_evals:
                issues.append("event_count_mismatch")
            if num_evals is not None and max_eval != num_evals:
                issues.append("max_eval_mismatch")
            if stats["x_json_present_count"] != stats["event_count"]:
                issues.append("missing_x_json")
            if stats["x_dim_match_count"] != stats["x_json_present_count"]:
                issues.append("x_dim_mismatch")
            if issues:
                example_bad_runs.append(
                    {
                        "run_id": run_id,
                        "instance_id": meta["instance_id"],
                        "issues": issues,
                        "num_evals_run_summary": num_evals,
                        "event_count": stats["event_count"],
                        "max_eval_index": max_eval,
                        "first_eval_index": stats["first_eval_index"],
                        "last_eval_index": stats["last_eval_index"],
                        "missing_x_examples": stats["missing_x_examples"],
                        "bad_dim_examples": stats["bad_dim_examples"],
                    }
                )

    decision_dims = sorted({run_index[run_id]["decision_dim"] for run_id in run_ids if run_index[run_id]["decision_dim"] is not None})
    variable_signatures = sorted(
        {
            "|".join(str(x) for x in run_index[run_id]["variable_name_json"])
            for run_id in run_ids
            if run_index[run_id]["variable_name_json"]
        }
    )

    return {
        "family": family,
        "n_runs": total_runs,
        "runs_with_iteration_events": runs_with_events,
        "iteration_event_coverage": float(runs_with_events / max(1, total_runs)),
        "total_iteration_events": event_total,
        "x_json_presence_rate": float(x_present_total / max(1, event_total)),
        "x_json_dim_match_rate": float(x_match_total / max(1, x_present_total)),
        "run_level_monotone_eval_index_rate": float(monotone_count / max(1, runs_with_events)),
        "run_level_contiguous_eval_index_rate": float(contiguous_count / max(1, runs_with_events)),
        "run_level_first_eval_is_one_rate": float(first_eval_is_one_count / max(1, runs_with_events)),
        "run_summary_event_count_match_rate": float(num_eval_match_count / max(1, runs_with_events)),
        "run_summary_max_eval_match_rate": float(max_eval_match_count / max(1, runs_with_events)),
        "decision_dims_seen": decision_dims,
        "variable_signatures_seen": variable_signatures[:10],
        "source_experiments": sorted(experiments),
        "iteration_event_files": {
            experiment: [_relpath(path) for path in iteration_paths.get(experiment, [])]
            for experiment in sorted(experiments)
        },
        "solver_status_counts": dict(sorted(solver_counter.items())),
        "example_problem_runs": example_bad_runs,
        "supports_event_level_structure_audit": bool(
            runs_with_events == total_runs
            and event_total > 0
            and x_present_total == event_total
            and x_match_total == x_present_total
            and monotone_count == runs_with_events
            and contiguous_count == runs_with_events
            and num_eval_match_count == runs_with_events
            and max_eval_match_count == runs_with_events
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit event-level evaluation traces for real AICO families.")
    ap.add_argument("--instances", default=DEFAULT_INSTANCES)
    ap.add_argument("--run_summaries", default=DEFAULT_RUN_SUMMARIES)
    ap.add_argument("--manifests", default=DEFAULT_MANIFESTS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES)
    args = ap.parse_args()

    families = {str(x) for x in args.families}
    bad_rows: List[Dict[str, Any]] = []
    instance_index = _load_instance_index(os.path.abspath(args.instances), families)
    run_index, experiment_to_runs = _load_target_runs(os.path.abspath(args.run_summaries), instance_index)
    iteration_paths = _load_manifest_iteration_paths(os.path.abspath(args.manifests), set(experiment_to_runs))
    run_stats = _scan_iteration_events(iteration_paths, run_index, bad_rows)

    families_to_runs: Dict[str, List[str]] = defaultdict(list)
    for run_id, meta in run_index.items():
        families_to_runs[meta["family"]].append(run_id)

    summaries = [
        _family_summary(family, sorted(families_to_runs.get(family, [])), run_index, run_stats, iteration_paths)
        for family in sorted(families)
    ]

    out = {
        "instances_path": os.path.abspath(args.instances),
        "run_summaries_path": os.path.abspath(args.run_summaries),
        "manifests_path": os.path.abspath(args.manifests),
        "families": summaries,
        "n_target_instances": len(instance_index),
        "n_target_runs": len(run_index),
        "n_target_experiments": len(experiment_to_runs),
        "bad_iteration_rows_skipped": len(bad_rows),
        "bad_iteration_row_examples": bad_rows,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(os.path.abspath(args.out), "w", encoding="utf-8") as handle:
        json.dump(out, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("=" * 80)
    print("AICO EVENT-LEVEL TRACE AUDIT")
    print("=" * 80)
    for row in summaries:
        print(
            f"{row['family']:<20} "
            f"runs={row['n_runs']:<6d} "
            f"coverage={row['iteration_event_coverage']:.3f} "
            f"x={row['x_json_presence_rate']:.3f}/{row['x_json_dim_match_rate']:.3f} "
            f"mono={row['run_level_monotone_eval_index_rate']:.3f} "
            f"match={row['run_summary_event_count_match_rate']:.3f}"
        )
    print("")
    print(f"Wrote: {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
