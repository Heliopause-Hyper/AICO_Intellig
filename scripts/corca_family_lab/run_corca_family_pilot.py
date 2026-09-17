from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List


REPO_ROOT = "/home/ycl/AICO-Intellig"
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "results", "corca_family_lab")


def _import_simulators():
    import sys

    src_dir = os.path.join(REPO_ROOT, "src")
    if src_dir not in sys.path:
        sys.path.append(src_dir)
    from simulator_interface import CORCAEvolSimulator, CORCAStateSimulator, CORCAXenonSimulator

    return CORCAStateSimulator, CORCAEvolSimulator, CORCAXenonSimulator


def _import_lab_helpers():
    import sys

    lab_dir = os.path.join(REPO_ROOT, "scripts", "corca_family_lab")
    if lab_dir not in sys.path:
        sys.path.append(lab_dir)
    from objectives import score_family
    from parse_corca_outputs import parse_burnup_output, parse_xenon_output
    from state_family_cases import state_family_candidates
    from transient_family_cases import burnup_family_candidates, xenon_family_candidates

    return (
        score_family,
        parse_burnup_output,
        parse_xenon_output,
        state_family_candidates,
        burnup_family_candidates,
        xenon_family_candidates,
    )


def _runtime_paths() -> Dict[str, str]:
    lpd = os.path.join(REPO_ROOT, "folderA", "preciseFZ")
    return {
        "lpd": lpd,
        "exec_state": os.path.join(lpd, "apply", "exec_state"),
        "exec_pre_burn": os.path.join(lpd, "apply", "exec_pre_burn"),
        "exec_pre_xenon": os.path.join(lpd, "apply", "exec_pre_xenon"),
        "hdf5": os.path.join(lpd, "databank", "COMRES_last", "define_rod_01_001.hdf5"),
        "burnup_out": os.path.join(lpd, "result", "SimuOutBurnup.out"),
        "xenon_out": os.path.join(lpd, "result", "SimuOutXenon.out"),
    }


def _candidate_grid(problem_type: str) -> List[Dict[str, object]]:
    if problem_type == "corca_state":
        _, _, _, state_family_candidates, _, _ = _import_lab_helpers()
        return list(state_family_candidates())
    if problem_type == "corca_evol":
        _, _, _, _, burnup_family_candidates, _ = _import_lab_helpers()
        return list(burnup_family_candidates())
    if problem_type == "corca_xenon":
        _, _, _, _, _, xenon_family_candidates = _import_lab_helpers()
        return list(xenon_family_candidates())
    raise ValueError(f"Unsupported problem_type: {problem_type}")


def _merge_lab_metrics(
    problem_type: str,
    result: Dict[str, object],
    parse_burnup_output,
    parse_xenon_output,
    runtime: Dict[str, str],
) -> Dict[str, object]:
    merged = dict(result)
    if problem_type == "corca_evol" and os.path.exists(runtime["burnup_out"]):
        extra = parse_burnup_output(runtime["burnup_out"])
        merged.update({k: v for k, v in extra.items() if k not in {"kind", "source_path"}})
    if problem_type == "corca_xenon" and os.path.exists(runtime["xenon_out"]):
        extra = parse_xenon_output(runtime["xenon_out"])
        merged.update({k: v for k, v in extra.items() if k not in {"kind", "source_path"}})
    return merged


def _build_simulator(problem_type: str, runtime: Dict[str, str]):
    CORCAStateSimulator, CORCAEvolSimulator, CORCAXenonSimulator = _import_simulators()
    if problem_type == "corca_state":
        return CORCAStateSimulator(
            exec_state_path=runtime["exec_state"],
            hdf5_file_path=runtime["hdf5"],
            template_dir=os.path.join(REPO_ROOT, "templates", "corcasim"),
        )
    if problem_type == "corca_evol":
        return CORCAEvolSimulator(
            exec_evol_path=runtime["exec_pre_burn"],
            hdf5_file_path=runtime["hdf5"],
            template_dir=os.path.join(REPO_ROOT, "templates", "corcasim"),
        )
    if problem_type == "corca_xenon":
        return CORCAXenonSimulator(
            exec_xenon_path=runtime["exec_pre_xenon"],
            hdf5_file_path=runtime["hdf5"],
            template_dir=os.path.join(REPO_ROOT, "templates", "corcasim"),
        )
    raise ValueError(f"Unsupported problem_type: {problem_type}")


def _template_name(problem_type: str) -> str:
    return {
        "corca_state": "default",
        "corca_evol": "burnup",
        "corca_xenon": "xenon",
    }[problem_type]


def _required_metric_names(problem_type: str) -> List[str]:
    if problem_type == "corca_state":
        return ["keff", "FQ", "FDH", "AO"]
    if problem_type == "corca_evol":
        return ["keff", "FQ", "burnup_deep"]
    if problem_type == "corca_xenon":
        return ["keff", "AO", "DI", "boron_concentration"]
    raise ValueError(f"Unsupported problem_type: {problem_type}")


def _missing_required_metrics(problem_type: str, metrics: Dict[str, object]) -> List[str]:
    missing = []
    for name in _required_metric_names(problem_type):
        if metrics.get(name) is None:
            missing.append(name)
    return missing


def run_problem(problem_type: str, max_cases: int, out_dir: str) -> str:
    score_family, parse_burnup_output, parse_xenon_output, _, _, _ = _import_lab_helpers()
    runtime = _runtime_paths()
    simulator = _build_simulator(problem_type, runtime)

    cases = _candidate_grid(problem_type)[: max(1, int(max_cases))]
    records: List[Dict[str, object]] = []

    for idx, params in enumerate(cases):
        case_label = str(params.get("_case_label", f"{problem_type}_case_{idx:02d}"))
        sim_params = {k: v for k, v in params.items() if not str(k).startswith("_")}
        result = simulator.run_simulation(_template_name(problem_type), dict(sim_params))
        merged = _merge_lab_metrics(problem_type, result, parse_burnup_output, parse_xenon_output, runtime)
        missing_metrics = _missing_required_metrics(problem_type, merged) if merged.get("success") else []
        success = bool(merged.get("success")) and not missing_metrics
        if success:
            score, pieces = score_family(problem_type, merged)
        else:
            pieces = {}
            if not merged.get("success"):
                pieces["execution_failure"] = 1e9
            if missing_metrics:
                pieces["missing_required_metrics"] = float(len(missing_metrics))
            score = 1e9
        records.append(
            {
                "case_id": f"{problem_type}_case_{idx:02d}",
                "case_label": case_label,
                "problem_type": problem_type,
                "params": sim_params,
                "success": success,
                "score": float(score),
                "score_breakdown": pieces,
                "missing_metrics": missing_metrics,
                "metrics": merged,
            }
        )

    best = min(records, key=lambda row: float(row["score"]))
    payload = {
        "problem_type": problem_type,
        "case_count": len(records),
        "best_case_id": best["case_id"],
        "best_score": best["score"],
        "records": records,
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{problem_type}_pilot_summary.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return out_path


def problem_types(arg: str) -> Iterable[str]:
    if arg == "all":
        return ("corca_state", "corca_evol", "corca_xenon")
    return (arg,)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run isolated CORCA family pilot cases.")
    ap.add_argument("--problem_type", default="all", choices=["all", "corca_state", "corca_evol", "corca_xenon"])
    ap.add_argument("--max_cases", type=int, default=2)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    out_paths = []
    for problem_type in problem_types(args.problem_type):
        out_paths.append(run_problem(problem_type, max_cases=args.max_cases, out_dir=out_dir))

    for path in out_paths:
        print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
