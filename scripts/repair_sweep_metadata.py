import argparse
import glob
import hashlib
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _write_jsonl_line(path: str, row: Dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        return
    with open(path, "a", encoding="utf-8", buffering=1) as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _stable_int(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def _shard_keep(problem_type: str, opt_params: List[str], objective_expression: str, shard_index: int, shard_count: int) -> bool:
    if shard_count <= 1:
        return True
    key = json.dumps(
        {"problem_type": str(problem_type), "opt_params": list(opt_params), "objective_expression": str(objective_expression)},
        ensure_ascii=False,
        sort_keys=True,
    )
    return (_stable_int(key) % int(shard_count)) == int(shard_index)


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            return float(v)
    except Exception:
        return None
    return None


def _infer_pde_type(problem_type: str, template_name: str) -> str:
    s = f"{str(problem_type or '').lower()}::{str(template_name or '').lower()}"
    if "advection" in s:
        return "hyperbolic"
    if "blackscholes" in s or "heat_time" in s:
        return "parabolic"
    if "sns" in s or "thmf" in s or "thermal_fins" in s or "heat_v" in s or "iaea" in s or "neutron_diffusion" in s:
        return "elliptic"
    return "unknown"


def _var_range_stats(lows: List[float], highs: List[float]) -> Dict[str, float]:
    widths = []
    for lo, hi in zip(lows, highs):
        try:
            widths.append(float(hi) - float(lo))
        except Exception:
            continue
    if not widths:
        return {"avg_var_range": 0.0, "max_var_range": 0.0, "min_var_range": 0.0}
    return {
        "avg_var_range": float(sum(widths) / float(len(widths))),
        "max_var_range": float(max(widths)),
        "min_var_range": float(min(widths)),
    }


def _physics_domain_from_template(template_name: str) -> str:
    t = str(template_name or "").lower()
    if t == "sns":
        return "fluid"
    if t == "thmf":
        return "heat"
    if t == "iaea":
        return "neutronics"
    if "advection" in t or "blackscholes" in t or "heat_time" in t:
        return "fluid"
    return "unknown"


def _build_pi_row(base_row: Dict[str, Any], max_iterations: int, time_limit_s: float) -> Dict[str, Any]:
    instance_id = str(base_row.get("instance_id") or "")
    problem_type = str(base_row.get("problem_type") or "")
    template_name = str(base_row.get("template_name") or "")
    opt_params = list(base_row.get("opt_params") or [])
    param_ranges = dict(base_row.get("param_ranges") or {})
    typed_cfg = base_row.get("typed_config_json")
    typed_cfg = typed_cfg if isinstance(typed_cfg, dict) else {}

    if typed_cfg.get("instance_id") != instance_id:
        typed_cfg = {**typed_cfg, "instance_id": instance_id}

    pde_type = str(typed_cfg.get("pde_type") or "").strip()
    if not pde_type or pde_type == "unknown":
        pde_type = _infer_pde_type(problem_type, template_name)
        typed_cfg = {**typed_cfg, "pde_type": pde_type}

    physics_domain = typed_cfg.get("physics_domain") or _physics_domain_from_template(template_name)
    task_type = typed_cfg.get("task_type") or "calibration"
    objective_type = typed_cfg.get("objective_type") or "target_matching"
    decision_dim = int(typed_cfg.get("decision_dim") or len(opt_params))
    num_constraints = int(typed_cfg.get("num_constraints") or 0)

    lows = []
    highs = []
    for p in opt_params:
        pr = param_ranges.get(p)
        if isinstance(pr, (list, tuple)) and len(pr) == 2:
            lows.append(float(pr[0]))
            highs.append(float(pr[1]))
        else:
            lows.append(0.0)
            highs.append(0.0)

    vr = _var_range_stats(lows, highs)
    budget_eval = int(typed_cfg.get("budget_eval") or max_iterations)
    budget_wall = float(typed_cfg.get("budget_walltime_sec") or time_limit_s)

    pi_row = {
        "instance_id": instance_id,
        "physics_domain": physics_domain,
        "pde_type": pde_type,
        "task_type": task_type,
        "objective_type": objective_type,
        "decision_dim": decision_dim,
        "variable_name_json": typed_cfg.get("variable_name_json") or list(opt_params),
        "var_lower_bound_json": typed_cfg.get("var_lower_bound_json") or lows,
        "var_upper_bound_json": typed_cfg.get("var_upper_bound_json") or highs,
        "num_constraints": num_constraints,
        "constraint_name_json": typed_cfg.get("constraint_name_json") or [],
        "constraint_type_json": typed_cfg.get("constraint_type_json") or [],
        "constraint_threshold_json": typed_cfg.get("constraint_threshold_json") or [],
        "budget_eval": budget_eval,
        "budget_walltime_sec": budget_wall,
        "typed_config_json": typed_cfg,
        **vr,
        "constraint_density": (float(num_constraints) / float(decision_dim)) if decision_dim > 0 else 0.0,
        "budget_per_dim": (float(budget_eval) / float(decision_dim)) if decision_dim > 0 else 0.0,
        "problem_type": problem_type,
        "template_name": template_name,
        "status": base_row.get("status"),
        "smoke": base_row.get("smoke"),
    }
    return pi_row


def _build_ds_row(
    base_row: Dict[str, Any],
    budget: Dict[str, Any],
    runs_for_instance: List[Dict[str, Any]],
) -> Dict[str, Any]:
    instance_id = str(base_row.get("instance_id") or "")
    problem_type = str(base_row.get("problem_type") or "")
    template_name = str(base_row.get("template_name") or "")
    opt_params = list(base_row.get("opt_params") or [])
    param_ranges = dict(base_row.get("param_ranges") or {})
    objective_expression = str(base_row.get("objective_expression") or "")
    typed_cfg = base_row.get("typed_config_json")
    typed_cfg = typed_cfg if isinstance(typed_cfg, dict) else {}

    by_algo: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in runs_for_instance:
        opt = str(r.get("optimizer") or "")
        method = str(r.get("method") or "")
        if not opt or not method:
            continue
        by_algo.setdefault((opt, method), []).append(r)

    candidate_results = []
    for (opt, method), rows in sorted(by_algo.items()):
        seed_runs = []
        for r in rows:
            seed_runs.append(
                {
                    "instance_id": instance_id,
                    "run_id": r.get("run_id"),
                    "seed": int(r.get("seed") or 0),
                    "problem_type": problem_type,
                    "optimizer": opt,
                    "method": method,
                    "success": bool(r.get("success", False)),
                    "best_value": r.get("best_value"),
                    "iterations": r.get("iterations", 0),
                    "execution_time": r.get("execution_time", 0.0),
                }
            )
        successes = [r for r in seed_runs if r.get("success")]
        if successes:
            m_best = []
            for r in successes:
                v = _safe_float(r.get("best_value"))
                if v is not None:
                    m_best.append(v)
            mean_best = (sum(m_best) / float(len(m_best))) if m_best else float("inf")
        else:
            mean_best = float("inf")
        m_time = []
        for r in seed_runs:
            v = _safe_float(r.get("execution_time"))
            if v is not None:
                m_time.append(v)
        mean_time = (sum(m_time) / float(len(m_time))) if m_time else 0.0

        candidate_results.append(
            {
                "instance_id": instance_id,
                "problem_type": problem_type,
                "optimizer": opt,
                "method": method,
                "seeds": list(budget.get("seeds") or []),
                "runs": seed_runs,
                "success_rate": (len(successes) / float(len(seed_runs))) if seed_runs else 0.0,
                "mean_best_value": mean_best,
                "mean_execution_time": mean_time,
            }
        )

    def _key_fn(x: Dict[str, Any]) -> float:
        v = _safe_float(x.get("mean_best_value"))
        return float(v) if v is not None else float("inf")

    best = min(candidate_results, key=_key_fn) if candidate_results else None
    label = f"{best['optimizer']}-{best['method']}" if best and _key_fn(best) != float("inf") else None

    return {
        "instance_id": instance_id,
        "problem_type": problem_type,
        "template_name": template_name,
        "opt_params": opt_params,
        "param_ranges": param_ranges,
        "objective_expression": objective_expression,
        "budget": dict(budget),
        "candidates": candidate_results,
        "label": label,
        "status": base_row.get("status"),
        "smoke": base_row.get("smoke"),
        "typed_config_json": typed_cfg,
    }


def _load_base_dataset(ds_path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in _read_jsonl(ds_path):
        iid = str(r.get("instance_id") or "")
        if iid:
            out[iid] = r
    return out


def _load_runs_by_instance(runs_path: str) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for r in _read_jsonl(runs_path):
        iid = str(r.get("instance_id") or "")
        if not iid:
            continue
        out.setdefault(iid, []).append(r)
    return out


def _audit_one_shard(shard_dir: str, base_map: Dict[str, Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    shard_count = int(cfg.get("shard_count") or 1)
    shard_index = int(cfg.get("shard_index") or 0)
    assigned = set()
    for iid, r in base_map.items():
        if _shard_keep(r.get("problem_type"), r.get("opt_params") or [], r.get("objective_expression") or "", shard_index, shard_count):
            assigned.add(iid)

    ds_path = os.path.join(shard_dir, "dataset.jsonl")
    pi_path = os.path.join(shard_dir, "problem_instance.jsonl")
    rs_path = os.path.join(shard_dir, "run_summary.jsonl")
    runs_path = os.path.join(shard_dir, "runs.jsonl")

    ds_ids = {str(r.get("instance_id") or "") for r in _read_jsonl(ds_path) if r.get("instance_id")}
    pi_ids = {str(r.get("instance_id") or "") for r in _read_jsonl(pi_path) if r.get("instance_id")}
    rs_ids = {str(r.get("instance_id") or "") for r in _read_jsonl(rs_path) if r.get("instance_id")}
    runs_by_inst = _load_runs_by_instance(runs_path)

    return {
        "assigned": assigned,
        "ds_ids": ds_ids,
        "pi_ids": pi_ids,
        "rs_ids": rs_ids,
        "runs_by_inst": runs_by_inst,
        "paths": {"dataset": ds_path, "problem_instance": pi_path, "run_summary": rs_path, "runs": runs_path},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--repair", choices=["problem_instance", "dataset", "both"], default="both")
    args = ap.parse_args()

    sweep_dir = os.path.abspath(str(args.sweep_dir))
    dry_run = not bool(args.apply)

    cfg_paths = sorted(glob.glob(os.path.join(sweep_dir, "shard_*", "full_sweep_config.json")))
    if not cfg_paths:
        raise SystemExit(f"no shard config found under: {sweep_dir}")

    total_added_pi = 0
    total_added_ds = 0

    for cfg_path in cfg_paths:
        shard_dir = os.path.dirname(cfg_path)
        cfg = json.load(open(cfg_path, "r", encoding="utf-8"))
        base_ds = str(cfg.get("instances_from_dataset_path") or "").strip()
        if not base_ds or not os.path.exists(base_ds):
            raise SystemExit(f"instances_from_dataset_path missing: {cfg_path}")

        base_map = _load_base_dataset(base_ds)
        budget_cfg = cfg.get("budget") or {}
        budget = {
            "max_iterations": int(budget_cfg.get("max_iterations", 0) or 0),
            "time_limit_s": float(budget_cfg.get("time_limit_s", 0.0) or 0.0),
            "seeds": [int(s) for s in (cfg.get("seeds") or [])],
        }

        shard_index = int(cfg.get("shard_index") or 0)
        shard_name = os.path.basename(shard_dir)
        shard_report = _audit_one_shard(shard_dir, base_map, {"shard_count": cfg.get("shard_count"), "shard_index": shard_index})

        rs_ids = set(shard_report["rs_ids"])
        need_pi = sorted(list(rs_ids - set(shard_report["pi_ids"])))
        need_ds = sorted(list(rs_ids - set(shard_report["ds_ids"])))

        if need_pi:
            print(f"{shard_name} pi_missing={len(need_pi)}")
        if need_ds:
            print(f"{shard_name} dataset_missing={len(need_ds)}")

        if args.repair in {"problem_instance", "both"}:
            for iid in need_pi:
                base_row = base_map.get(iid)
                if not base_row:
                    continue
                pi_row = _build_pi_row(base_row, max_iterations=int(budget["max_iterations"]), time_limit_s=float(budget["time_limit_s"]))
                _write_jsonl_line(shard_report["paths"]["problem_instance"], pi_row, dry_run=dry_run)
                total_added_pi += 1

        if args.repair in {"dataset", "both"}:
            runs_by_inst = shard_report["runs_by_inst"]
            for iid in need_ds:
                base_row = base_map.get(iid)
                if not base_row:
                    continue
                ds_row = _build_ds_row(base_row, budget=budget, runs_for_instance=list(runs_by_inst.get(iid) or []))
                _write_jsonl_line(shard_report["paths"]["dataset"], ds_row, dry_run=dry_run)
                total_added_ds += 1

    print(json.dumps({"dry_run": bool(dry_run), "added_problem_instance": total_added_pi, "added_dataset": total_added_ds}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

