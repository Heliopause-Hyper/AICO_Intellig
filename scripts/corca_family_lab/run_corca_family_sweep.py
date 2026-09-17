from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List


REPO_ROOT = "/home/ycl/AICO-Intellig"
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "results", "corca_family_lab")
DEFAULT_PROTOCOL = os.path.join(DEFAULT_OUT_DIR, "corca_family_protocol_v1.json")


def _import_pilot_runner():
    import sys

    lab_dir = os.path.join(REPO_ROOT, "scripts", "corca_family_lab")
    if lab_dir not in sys.path:
        sys.path.append(lab_dir)
    from run_corca_family_pilot import run_problem

    return run_problem


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _problem_types_for_family(family: Dict[str, object]) -> List[str]:
    raw = family.get("problem_type")
    if isinstance(raw, list):
        return [str(x) for x in raw]
    return [str(raw)]


def _records_digest(payload: Dict[str, object]) -> Dict[str, object]:
    records = list(payload.get("records") or [])
    scores = [float(row.get("score", 1e9)) for row in records]
    successes = [bool(row.get("success")) for row in records]
    best = min(records, key=lambda row: float(row.get("score", 1e9))) if records else {}
    return {
        "case_count": len(records),
        "success_count": int(sum(successes)),
        "success_rate": (sum(successes) / len(successes)) if successes else 0.0,
        "best_case_id": best.get("case_id"),
        "best_score": float(best.get("score", 1e9)) if best else 1e9,
        "mean_score": (sum(scores) / len(scores)) if scores else 1e9,
    }


def run_family_sweep(protocol_path: str, out_dir: str, family_id: str, max_cases: int) -> str:
    protocol = _load_json(protocol_path)
    run_problem = _import_pilot_runner()

    family_rows = [row for row in list(protocol.get("families") or []) if row.get("family_id") == family_id]
    if not family_rows:
        raise ValueError(f"Family not found in protocol: {family_id}")
    family = family_rows[0]

    summaries = []
    for problem_type in _problem_types_for_family(family):
        path = run_problem(problem_type=problem_type, max_cases=max_cases, out_dir=out_dir)
        payload = _load_json(path)
        summaries.append(
            {
                "problem_type": problem_type,
                "summary_path": path,
                "digest": _records_digest(payload),
            }
        )

    best_score = min(float(row["digest"]["best_score"]) for row in summaries) if summaries else 1e9
    payload = {
        "family_id": family_id,
        "protocol_path": os.path.abspath(protocol_path),
        "max_cases_per_problem": int(max_cases),
        "problem_summaries": summaries,
        "best_score_across_family": float(best_score),
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{family_id}_sweep_summary.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Run isolated CORCA family sweeps from the lab protocol.")
    ap.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    ap.add_argument("--family_id", default="all", choices=["all", "corca_state_family_v1", "corca_transient_family_v1"])
    ap.add_argument("--max_cases", type=int, default=3)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    protocol = _load_json(args.protocol)
    family_ids = [str(row.get("family_id")) for row in list(protocol.get("families") or [])]
    if args.family_id != "all":
        family_ids = [args.family_id]

    out_paths = []
    for family_id in family_ids:
        out_paths.append(
            run_family_sweep(
                protocol_path=os.path.abspath(args.protocol),
                out_dir=os.path.abspath(args.out_dir),
                family_id=family_id,
                max_cases=args.max_cases,
            )
        )

    for path in out_paths:
        print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
