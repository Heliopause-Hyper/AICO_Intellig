from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple


REPO_ROOT = "/home/ycl/AICO-Intellig"
DEFAULT_INSTANCES = os.path.join(REPO_ROOT, "results", "curated_datasets", "instances.jsonl")
DEFAULT_RUN_SUMMARIES = os.path.join(REPO_ROOT, "results", "curated_datasets", "run_summaries.jsonl")
DEFAULT_OUT = os.path.join(REPO_ROOT, "results", "complexity_study", "archive_family_audit.json")


def _read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            yield json.loads(text)


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


def _value_list(row: Dict[str, Any], key: str) -> List[Any]:
    typed = _typed_config(row)
    raw = typed.get(key)
    if isinstance(raw, list):
        return list(raw)
    raw = row.get(key)
    if isinstance(raw, list):
        return list(raw)
    return []


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _signature(names: List[Any]) -> str:
    clean = [str(x) for x in names if str(x)]
    return "|".join(clean) if clean else "<unknown>"


def _bucket_family_summary(instances_path: str, runs_path: str) -> Dict[str, Any]:
    family_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    instance_to_family: Dict[str, str] = {}

    for row in _read_jsonl(instances_path):
        family = _canonical_family(row)
        family_rows[family].append(row)
        iid = str(row.get("instance_id") or "")
        if iid:
            instance_to_family[iid] = family

    run_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(runs_path):
        iid = str(row.get("instance_id") or "")
        family = instance_to_family.get(iid)
        if family:
            run_rows[family].append(row)

    family_summaries: List[Dict[str, Any]] = []
    for family in sorted(family_rows):
        rows = family_rows[family]
        runs = run_rows.get(family, [])

        dims = [_safe_int(r.get("decision_dim") or _typed_config(r).get("decision_dim")) for r in rows]
        budgets = [_safe_int(r.get("budget_eval") or _typed_config(r).get("budget_eval")) for r in rows]
        walltimes = [_safe_float(r.get("budget_walltime_sec") or _typed_config(r).get("budget_walltime_sec")) for r in rows]
        num_constraints = [_safe_int(r.get("num_constraints") or _typed_config(r).get("num_constraints")) for r in rows]
        variable_signatures = [_signature(_value_list(r, "variable_name_json")) for r in rows]
        objective_types = [str(r.get("objective_type") or _typed_config(r).get("objective_type") or "") for r in rows]
        task_types = [str(r.get("task_type") or _typed_config(r).get("task_type") or "") for r in rows]
        pde_types = [str(r.get("pde_type") or _typed_config(r).get("pde_type") or "") for r in rows]

        signature_counter = Counter(variable_signatures)
        dominant_signature, dominant_count = signature_counter.most_common(1)[0]

        num_evals = [_safe_int(r.get("num_evals")) for r in runs if r.get("num_evals") is not None]
        feasible_flags = [_safe_int(bool(r.get("final_feasible_flag"))) for r in runs]
        failure_counts = [_safe_int(r.get("failure_count")) for r in runs]
        timeout_counts = [_safe_int(r.get("timeout_count")) for r in runs]

        dominant_share = dominant_count / max(1, len(rows))

        candidate_score = 0
        if len(rows) >= 100:
            candidate_score += 1
        if _median(dims) >= 3:
            candidate_score += 1
        if dominant_share >= 0.60:
            candidate_score += 1
        if _median(num_evals) >= 20:
            candidate_score += 1

        strict_ready = bool(
            len(rows) >= 100
            and _median(dims) >= 3
            and dominant_share >= 0.40
            and _median(num_evals) >= 20
        )
        manual_followup = bool(
            len(rows) >= 100
            and _median(num_evals) >= 20
            and (_median(dims) >= 3 or dominant_share >= 0.40)
        )

        family_summaries.append(
            {
                "family": family,
                "n_instances": len(rows),
                "n_runs": len(runs),
                "decision_dim": {
                    "min": min(dims) if dims else 0,
                    "median": _median(dims),
                    "max": max(dims) if dims else 0,
                    "unique": sorted(set(dims)),
                },
                "budget_eval_median": _median(budgets),
                "budget_walltime_sec_median": _median(walltimes),
                "num_constraints_median": _median(num_constraints),
                "num_evals": {
                    "median": _median(num_evals),
                    "max": max(num_evals) if num_evals else 0,
                },
                "dominant_variable_signature": dominant_signature,
                "dominant_variable_signature_share": dominant_share,
                "unique_variable_signatures": len(signature_counter),
                "top_variable_signatures": [
                    {"signature": sig, "count": count}
                    for sig, count in signature_counter.most_common(5)
                ],
                "objective_types": dict(Counter(x for x in objective_types if x)),
                "task_types": dict(Counter(x for x in task_types if x)),
                "pde_types": dict(Counter(x for x in pde_types if x)),
                "run_feasible_rate": sum(feasible_flags) / max(1, len(feasible_flags)),
                "run_failure_count_mean": (sum(failure_counts) / len(failure_counts)) if failure_counts else 0.0,
                "run_timeout_count_mean": (sum(timeout_counts) / len(timeout_counts)) if timeout_counts else 0.0,
                "candidate_score": candidate_score,
                "recommended_for_shared_structure": strict_ready,
                "manual_followup_candidate": manual_followup,
            }
        )

    ranked = sorted(
        family_summaries,
        key=lambda row: (
            row["candidate_score"],
            row["n_instances"],
            row["decision_dim"]["median"],
            row["n_runs"],
        ),
        reverse=True,
    )

    return {
        "instances_path": instances_path,
        "run_summaries_path": runs_path,
        "n_families": len(ranked),
        "recommended_families": [row["family"] for row in ranked if row["recommended_for_shared_structure"]],
        "manual_followup_families": [row["family"] for row in ranked if row["manual_followup_candidate"]],
        "families": ranked,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit AICO family structure for complexity study readiness.")
    ap.add_argument("--instances", default=DEFAULT_INSTANCES)
    ap.add_argument("--run_summaries", default=DEFAULT_RUN_SUMMARIES)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    summary = _bucket_family_summary(
        instances_path=os.path.abspath(args.instances),
        runs_path=os.path.abspath(args.run_summaries),
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(os.path.abspath(args.out), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("=" * 80)
    print("AICO FAMILY AUDIT FOR COMPLEXITY STUDY")
    print("=" * 80)
    for row in summary["families"][:10]:
        print(
            f"{row['family']:<30} "
            f"inst={row['n_instances']:<5d} "
            f"runs={row['n_runs']:<6d} "
            f"dim_med={row['decision_dim']['median']:<4.1f} "
            f"sig_share={row['dominant_variable_signature_share']:.2f} "
            f"score={row['candidate_score']}"
        )
    print("")
    print("Recommended families:")
    for family in summary["recommended_families"]:
        print(f"  - {family}")
    print("")
    print("Manual follow-up families:")
    for family in summary["manual_followup_families"][:10]:
        print(f"  - {family}")
    print("")
    print(f"Wrote: {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
