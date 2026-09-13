import argparse
import json
import os
import hashlib
import random
import time
import uuid
from typing import Any, Dict, List, Tuple


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    

def _json_default(o):
    try:
        import numpy as np

        if isinstance(o, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(o)
        if isinstance(o, (np.float_, np.float16, np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
    except Exception:
        pass
    return str(o)


def _write_jsonl(f, row: Dict[str, Any]) -> None:
    f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


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


def _checkpoint_ratios() -> List[int]:
    return [1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 55, 60, 70, 80, 85, 90, 100]


def _run_checkpoints(run_id: str, instance_id: str, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    n = int(len(history))
    if n <= 0:
        return []
    best_loss = float("inf")
    best_feas = float("inf")
    first_feasible_idx = None
    total_eval_time = 0.0
    failures = 0
    best_last_update = 0
    trace = []
    for i, ev in enumerate(history, start=1):
        loss = ev.get("loss_value")
        try:
            loss_f = float(loss)
        except Exception:
            loss_f = float("inf")
        feas = bool(ev.get("feasible_flag", False))
        if feas and first_feasible_idx is None:
            first_feasible_idx = i
        if not feas:
            failures += 1
        try:
            total_eval_time += float(ev.get("eval_time_sec", 0.0) or 0.0)
        except Exception:
            pass
        improved = False
        if loss_f < best_loss:
            best_loss = loss_f
            best_last_update = i
            improved = True
        if feas and loss_f < best_feas:
            best_feas = loss_f
        trace.append({"i": i, "loss": loss_f, "feas": feas, "improved": improved})

    final_best = best_loss if best_loss != float("inf") else None
    regrets = []
    if final_best is not None:
        for t in trace:
            regrets.append(max(0.0, float(t["loss"]) - float(final_best)))
    auc_regret_prefix = []
    acc = 0.0
    for r in regrets:
        acc += float(r)
        auc_regret_prefix.append(acc)

    out: List[Dict[str, Any]] = []
    cid = 0
    for ratio in _checkpoint_ratios():
        idx = int((float(ratio) / 100.0) * float(n))
        idx = max(1, min(n, idx))
        ev = history[idx - 1]
        loss = ev.get("loss_value")
        try:
            loss_f = float(loss)
        except Exception:
            loss_f = float("inf")
        feas = bool(ev.get("feasible_flag", False))
        solver_status = str(ev.get("solver_status") or "ok")
        best_so_far = min(float(t["loss"]) for t in trace[:idx]) if idx > 0 else float("inf")
        best_feas_so_far = min(float(t["loss"]) for t in trace[:idx] if t["feas"]) if any(t["feas"] for t in trace[:idx]) else float("inf")
        if best_feas_so_far == float("inf"):
            best_feas_so_far = None
        window = 5
        w0 = max(1, idx - window)
        seg = trace[w0 - 1 : idx]
        recent_best = min(float(t["loss"]) for t in seg) if seg else best_so_far
        recent_improvement = 0.0
        try:
            recent_improvement = max(0.0, float(seg[0]["loss"]) - float(recent_best))
        except Exception:
            recent_improvement = 0.0
        stagnation = 0
        for t in reversed(seg):
            if t["improved"]:
                break
            stagnation += 1
        feas_ratio = (sum(1 for t in trace[:idx] if t["feas"]) / float(idx)) if idx > 0 else 0.0
        fail_rate = (sum(1 for t in trace[max(0, idx - window) : idx] if not t["feas"]) / float(min(window, idx))) if idx > 0 else 0.0
        norm_impr = 0.0
        try:
            first = float(trace[0]["loss"])
            denom = max(1e-9, abs(first))
            norm_impr = max(0.0, (first - float(best_so_far)) / denom) if best_so_far != float("inf") else 0.0
        except Exception:
            norm_impr = 0.0
        out.append(
            {
                "checkpoint_id": f"{run_id}_{cid}",
                "run_id": run_id,
                "instance_id": instance_id,
                "checkpoint_type": "ratio",
                "checkpoint_ratio": int(ratio),
                "eval_index": int(idx),
                "objective_raw": float(loss_f) if loss_f != float("inf") else None,
                "constraint_violation_total": float(ev.get("constraint_violation_total", 0.0) or 0.0),
                "feasible_flag": bool(feas),
                "best_loss": float(best_so_far) if best_so_far != float("inf") else None,
                "best_feasible_loss": best_feas_so_far,
                "solver_status": solver_status,
                "recent_improvement_rate": float(recent_improvement),
                "stagnation_length": int(stagnation),
                "feasible_ratio": float(feas_ratio),
                "first_feasible_eval_index": int(first_feasible_idx) if first_feasible_idx is not None else None,
                "constraint_violation_slope": None,
                "avg_eval_time": float(total_eval_time / float(max(1, idx))),
                "recent_failure_rate": float(fail_rate),
                "normalized_improvement": float(norm_impr),
                "auc_regret_prefix": float(auc_regret_prefix[idx - 1]) if auc_regret_prefix and (idx - 1) < len(auc_regret_prefix) else None,
                "sampling_radius": None,
                "local_lipschitz_estimate": None,
                "incumbent_age": int(idx - best_last_update) if best_last_update else int(idx),
            }
        )
        cid += 1

    out.append(
        {
            "checkpoint_id": f"{run_id}_{cid}",
            "run_id": run_id,
            "instance_id": instance_id,
            "checkpoint_type": "final",
            "checkpoint_ratio": 100,
            "eval_index": int(n),
            "objective_raw": float(trace[-1]["loss"]) if trace else None,
            "constraint_violation_total": float(history[-1].get("constraint_violation_total", 0.0) or 0.0),
            "feasible_flag": bool(history[-1].get("feasible_flag", False)),
            "best_loss": float(best_loss) if best_loss != float("inf") else None,
            "best_feasible_loss": float(best_feas) if best_feas != float("inf") else None,
            "solver_status": str(history[-1].get("solver_status") or "ok"),
            "recent_improvement_rate": None,
            "stagnation_length": None,
            "feasible_ratio": (sum(1 for t in trace if t["feas"]) / float(n)) if n > 0 else 0.0,
            "first_feasible_eval_index": int(first_feasible_idx) if first_feasible_idx is not None else None,
            "constraint_violation_slope": None,
            "avg_eval_time": float(total_eval_time / float(max(1, n))),
            "recent_failure_rate": None,
            "normalized_improvement": None,
            "auc_regret_prefix": float(auc_regret_prefix[-1]) if auc_regret_prefix else None,
            "sampling_radius": None,
            "local_lipschitz_estimate": None,
            "incumbent_age": 0,
        }
    )
    return out


def _try_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _algo_available(optimizer: str) -> bool:
    if optimizer == "SciPy":
        return _try_import("scipy")
    if optimizer == "Optuna":
        return _try_import("optuna")
    if optimizer == "Hyperopt":
        return _try_import("hyperopt")
    if optimizer == "Nevergrad":
        try:
            import numpy as np
            np_dict = np.__dict__
            if "float_" not in np_dict:
                np.float_ = np.float64
            if "float" not in np_dict:
                np.float = float
            if "int" not in np_dict:
                np.int = int
            if "bool" not in np_dict:
                np.bool = bool
            if "object" not in np_dict:
                np.object = object
            if "str" not in np_dict:
                np.str = str
            if "complex" not in np_dict:
                np.complex = complex
            if "long" not in np_dict:
                np.long = int
            __import__("nevergrad")
            return True
        except Exception:
            return False
    return False


def _make_instance_id(problem_type: str) -> str:
    return f"inst_{problem_type}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _make_stable_instance_id(
    problem_type: str,
    opt_params: List[str],
    objective_expression: str,
    budget: Dict[str, Any],
    seeds: List[int],
) -> str:
    payload = {
        "problem_type": problem_type,
        "opt_params": list(opt_params),
        "objective_expression": str(objective_expression),
        "budget": {
            "max_iterations": int(budget.get("max_iterations", 0)),
            "time_limit_s": float(budget.get("time_limit_s", 0.0)),
        },
        "seeds": [int(s) for s in seeds],
    }
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:16]
    return f"inst_{problem_type}_{h}"


def _make_stable_instance_id_v2(payload: Dict[str, Any]) -> str:
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:16]
    problem_type = str(payload.get("problem_type") or "unknown")
    return f"inst_{problem_type}_{h}"


def _select_param_ranges(problem_config: dict, opt_params: List[str]) -> Dict[str, Tuple[float, float]]:
    ranges = problem_config.get("parameters", {}) or {}
    out = {}
    for p in opt_params:
        if p not in ranges:
            raise KeyError(f"参数不存在: {p}")
        lo, hi = ranges[p]
        out[p] = (float(lo), float(hi))
    return out


def _set_seed(seed: int):
    random.seed(int(seed))
    try:
        import numpy as np
        np.random.seed(int(seed))
    except Exception:
        pass


def _stable_int(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def _shard_keep(problem_type: str, opt_params: List[str], objective_expression: str, shard_index: int, shard_count: int) -> bool:
    if shard_count <= 1:
        return True
    key = json.dumps(
        {
            "problem_type": str(problem_type),
            "opt_params": list(opt_params),
            "objective_expression": str(objective_expression),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return (_stable_int(key) % int(shard_count)) == int(shard_index)


def _candidate_shard_keep(optimizer: str, method: str, shard_index: int, shard_count: int) -> bool:
    if shard_count <= 1:
        return True
    key = json.dumps(
        {
            "optimizer": str(optimizer),
            "method": str(method),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return (_stable_int(key) % int(shard_count)) == int(shard_index)


def _objective_expression_for_metric(metric: str, targets: Dict[str, float]) -> str:
    m = str(metric)
    if m == "keff":
        tgt = float(targets.get("keff", 1.0))
        return f"abs(result.get('keff', 1e10) - {tgt})"
    if m == "pressuredrop":
        pd_tgt = targets.get("pressuredrop_target")
        if pd_tgt is not None:
            return f"abs(result.get('pressuredrop', 1e10) - {float(pd_tgt)})"
        return "result.get('pressuredrop', 1e10)"
    if m in {"tempuniformity"}:
        return "abs(result.get('tempuniformity', 1e10))"
    if m in {"efficiency", "totalheatflux"}:
        return f"-result.get('{m}', -1e10)"
    if m in {"u_mean"}:
        tgt = targets.get("u_mean_target")
        if tgt is not None:
            return f"abs(result.get('u_mean', 1e10) - {float(tgt)})"
    return f"result.get('{m}', 1e10)"


def _sns_flow_family_generate_instances(cfg: dict, budget: Dict[str, Any], seeds: List[int]) -> List[Dict[str, Any]]:
    count = int(cfg.get("sns_family_count", cfg.get("auto_max_total_instances", 200)) or 0)
    base_seed = int(cfg.get("sns_family_seed", cfg.get("auto_seed", 0)) or 0)
    u_lower_lo, u_lower_hi = cfg.get("sns_family_u_lower_range", [3.0, 8.0])
    u_upper_lo, u_upper_hi = cfg.get("sns_family_u_upper_range", [12.0, 20.0])
    min_width = float(cfg.get("sns_family_min_u_width", 6.0))
    max_width = cfg.get("sns_family_max_u_width", None)
    if max_width is not None:
        max_width = float(max_width)

    clean_ratio = float(cfg.get("sns_family_clean_ratio", 0.75))
    bucket_ratios = cfg.get("sns_family_target_bucket_ratios") or {}
    try:
        r_mid = float(bucket_ratios.get("mid", 0.0))
        r_edge = float(bucket_ratios.get("edge", 0.0))
        r_out = float(bucket_ratios.get("outside", 0.0))
    except Exception:
        r_mid, r_edge, r_out = 0.0, 0.0, 0.0
    if r_mid <= 0.0 and r_edge <= 0.0 and r_out <= 0.0:
        r_out = max(0.0, min(1.0, 1.0 - float(clean_ratio)))
        r_mid = max(0.0, min(1.0, float(clean_ratio) * 0.7))
        r_edge = max(0.0, min(1.0, float(clean_ratio) * 0.3))
    s = max(1e-12, r_mid + r_edge + r_out)
    r_mid, r_edge, r_out = r_mid / s, r_edge / s, r_out / s

    mid_margin_frac = float(cfg.get("sns_family_mid_margin_frac", 0.2))
    edge_frac = float(cfg.get("sns_family_edge_frac", 0.15))
    outside_margin = float(cfg.get("sns_family_outside_margin", cfg.get("sns_family_hard_target_margin", 80.0)))
    approx_slope = float(cfg.get("sns_family_dp_slope", 107.5))
    approx_intercept = float(cfg.get("sns_family_dp_intercept", -177.5))
    problem_type = str(cfg.get("sns_family_problem_type", "sns_flow_family"))

    rng = random.Random(_stable_int(f"sns_flow_family:{base_seed}"))
    instances: List[Dict[str, Any]] = []
    for _ in range(max(0, count)):
        u_lo = float(rng.uniform(float(u_lower_lo), float(u_lower_hi)))
        u_hi = float(rng.uniform(float(u_upper_lo), float(u_upper_hi)))
        if u_hi - u_lo < min_width:
            u_hi = u_lo + min_width
        if max_width is not None and (u_hi - u_lo) > max_width:
            u_hi = u_lo + max_width
        u_lo = max(1e-6, u_lo)
        u_hi = max(u_lo + 1e-6, u_hi)

        dp_lo_est = approx_slope * u_lo + approx_intercept
        dp_hi_est = approx_slope * u_hi + approx_intercept
        if dp_hi_est < dp_lo_est:
            dp_lo_est, dp_hi_est = dp_hi_est, dp_lo_est
        span = float(max(1e-9, dp_hi_est - dp_lo_est))

        r = rng.random()
        if r < r_mid:
            lo = float(dp_lo_est + span * float(mid_margin_frac))
            hi = float(dp_hi_est - span * float(mid_margin_frac))
            if hi <= lo:
                lo, hi = float(dp_lo_est), float(dp_hi_est)
            target_dp = float(rng.uniform(lo, hi))
            bucket = "mid"
        elif r < (r_mid + r_edge):
            if rng.random() < 0.5:
                lo = float(dp_lo_est)
                hi = float(dp_lo_est + span * float(edge_frac))
            else:
                lo = float(dp_hi_est - span * float(edge_frac))
                hi = float(dp_hi_est)
            if hi <= lo:
                lo, hi = float(dp_lo_est), float(dp_hi_est)
            target_dp = float(rng.uniform(lo, hi))
            bucket = "edge"
        else:
            if rng.random() < 0.5:
                target_dp = float(dp_lo_est - abs(outside_margin) - abs(rng.gauss(0.0, abs(outside_margin) * 0.25)))
            else:
                target_dp = float(dp_hi_est + abs(outside_margin) + abs(rng.gauss(0.0, abs(outside_margin) * 0.25)))
            bucket = "outside"

        opt_params = ["uMax"]
        param_ranges = {"uMax": (float(u_lo), float(u_hi))}
        objective_expression = f"abs(result.get('pressuredrop', 1e10) - {target_dp})"

        typed_config = {
            "instance_id": None,
            "physics_domain": "fluid",
            "task_type": "calibration",
            "objective_type": "target_matching",
            "decision_dim": 1,
            "variable_name_json": opt_params,
            "var_lower_bound_json": [float(u_lo)],
            "var_upper_bound_json": [float(u_hi)],
            "num_constraints": 0,
            "constraint_name_json": [],
            "constraint_type_json": [],
            "constraint_threshold_json": [],
            "budget_eval": int(budget.get("max_iterations", 0)),
            "budget_walltime_sec": float(budget.get("time_limit_s", 0.0)),
            "template_name": "sns",
            "targets": {"pressuredrop_target": float(target_dp)},
            "generation_bucket": bucket,
        }

        payload = {
            "problem_type": problem_type,
            "opt_params": opt_params,
            "objective_expression": objective_expression,
            "param_ranges": param_ranges,
            "budget": {
                "max_iterations": int(budget.get("max_iterations", 0)),
                "time_limit_s": float(budget.get("time_limit_s", 0.0)),
            },
            "seeds": [int(s) for s in seeds],
            "typed_config_json": typed_config,
        }
        instance_id = _make_stable_instance_id_v2(payload)
        typed_config["instance_id"] = instance_id
        instances.append(
            {
                "instance_id": instance_id,
                "problem_type": problem_type,
                "opt_params": opt_params,
                "param_ranges": param_ranges,
                "objective_expression": objective_expression,
                "typed_config_json": typed_config,
            }
        )
    return instances


def _sns_objective_expr(metric: str, objective_type: str, target_value: Any = None) -> str:
    metric = str(metric)
    objective_type = str(objective_type)
    if objective_type == "target_matching":
        if target_value is None:
            return f"abs(result.get('{metric}', 1e10))"
        return f"abs(result.get('{metric}', 1e10) - {float(target_value)})"
    if objective_type == "max":
        return f"-result.get('{metric}', -1e10)"
    return f"result.get('{metric}', 1e10)"


def _sns_metric_estimate_ranges(
    u_lo: float,
    u_hi: float,
    mu_lo: float,
    mu_hi: float,
    r_lo: float,
    r_hi: float,
    approx_slope: float,
    approx_intercept: float,
) -> Dict[str, Tuple[float, float]]:
    mu_mid = 0.5 * (float(mu_lo) + float(mu_hi))
    r_mid = 0.5 * (float(r_lo) + float(r_hi))
    mu_factor = max(0.5, min(2.5, 0.1 / max(1e-6, mu_mid)))
    r_factor = max(0.7, min(1.5, r_mid / 0.2))

    def _dp_est(u: float) -> float:
        return max(1.0, float((float(approx_slope) * float(u) + float(approx_intercept)) * mu_factor * r_factor))

    dp_lo = _dp_est(u_lo)
    dp_hi = _dp_est(u_hi)
    if dp_hi < dp_lo:
        dp_lo, dp_hi = dp_hi, dp_lo

    avg_lo = max(1.0, 0.48 * dp_lo)
    avg_hi = max(avg_lo + 1e-6, 0.48 * dp_hi)
    pmax_lo = max(avg_lo + 1.0, 1.28 * dp_lo)
    pmax_hi = max(pmax_lo + 1e-6, 1.28 * dp_hi)

    vel_scale = max(0.85, min(1.35, mu_factor ** 0.08))
    geom_scale = max(0.9, min(1.25, r_factor ** 0.35))
    vel_lo = max(0.1, 1.35 * float(u_lo) * vel_scale * geom_scale)
    vel_hi = max(vel_lo + 1e-6, 1.35 * float(u_hi) * vel_scale * geom_scale)
    if vel_hi < vel_lo:
        vel_lo, vel_hi = vel_hi, vel_lo

    return {
        "pressuredrop": (float(dp_lo), float(dp_hi)),
        "avgpressure": (float(avg_lo), float(avg_hi)),
        "pmax": (float(pmax_lo), float(pmax_hi)),
        "velmagmax": (float(vel_lo), float(vel_hi)),
    }


def _sns_sample_target_value(
    rng: random.Random,
    metric_lo_est: float,
    metric_hi_est: float,
    r_mid: float,
    r_edge: float,
    _r_out: float,
    mid_margin_frac: float,
    edge_frac: float,
    outside_margin: float,
) -> Tuple[float, str]:
    if metric_hi_est < metric_lo_est:
        metric_lo_est, metric_hi_est = metric_hi_est, metric_lo_est
    metric_lo_est = max(0.1, float(metric_lo_est))
    metric_hi_est = max(metric_lo_est + 1e-6, float(metric_hi_est))
    span = float(max(1e-9, metric_hi_est - metric_lo_est))
    outside_pad = max(1.0, min(abs(float(outside_margin)), max(5.0, 0.25 * max(abs(metric_hi_est), abs(metric_lo_est), span))))
    r = rng.random()
    if r < r_mid:
        lo = float(metric_lo_est + span * float(mid_margin_frac))
        hi = float(metric_hi_est - span * float(mid_margin_frac))
        if hi <= lo:
            lo, hi = float(metric_lo_est), float(metric_hi_est)
        return float(rng.uniform(lo, hi)), "mid"
    if r < (r_mid + r_edge):
        if rng.random() < 0.5:
            lo = float(metric_lo_est)
            hi = float(metric_lo_est + span * float(edge_frac))
        else:
            lo = float(metric_hi_est - span * float(edge_frac))
            hi = float(metric_hi_est)
        if hi <= lo:
            lo, hi = float(metric_lo_est), float(metric_hi_est)
        return float(rng.uniform(lo, hi)), "edge"
    if rng.random() < 0.5:
        return max(0.1, float(metric_lo_est - outside_pad - abs(rng.gauss(0.0, outside_pad * 0.25)))), "outside"
    return max(0.1, float(metric_hi_est + outside_pad + abs(rng.gauss(0.0, outside_pad * 0.25)))), "outside"


def _sns_build_constraints(rng: random.Random, est_ranges: Dict[str, Tuple[float, float]], task_constraints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for spec in task_constraints or []:
        if not isinstance(spec, dict):
            continue
        metric = str(spec.get("metric") or "").strip()
        ctype = str(spec.get("type") or "<=").strip()
        if not metric:
            continue
        value_range = spec.get("value_range")
        if isinstance(value_range, (list, tuple)) and len(value_range) >= 2:
            vr_lo = float(value_range[0])
            vr_hi = float(value_range[1])
            value = float(rng.uniform(min(vr_lo, vr_hi), max(vr_lo, vr_hi)))
        else:
            if metric not in est_ranges:
                continue
            lo_est, hi_est = est_ranges[metric]
            anchor = str(spec.get("anchor") or "mid").strip().lower()
            if anchor == "lo":
                base = float(lo_est)
            elif anchor == "hi":
                base = float(hi_est)
            else:
                base = 0.5 * (float(lo_est) + float(hi_est))

            ratio_range = spec.get("ratio_range", [1.0, 1.0])
            if isinstance(ratio_range, (list, tuple)) and len(ratio_range) >= 2:
                rr_lo = float(ratio_range[0])
                rr_hi = float(ratio_range[1])
            else:
                rr_lo = rr_hi = float(ratio_range if ratio_range is not None else 1.0)
            scale = float(rng.uniform(min(rr_lo, rr_hi), max(rr_lo, rr_hi)))
            value = float(base * scale + float(spec.get("offset", 0.0) or 0.0))
        min_value = spec.get("min_value")
        max_value = spec.get("max_value")
        if min_value is not None:
            value = max(float(min_value), value)
        if max_value is not None:
            value = min(float(max_value), value)
        out.append(
            {
                "metric": metric,
                "type": ctype,
                "value": float(value),
                "tol": float(spec.get("tol", 0.0) or 0.0),
                "penalty_weight": float(spec.get("penalty_weight", 1000.0) or 1000.0),
            }
        )
    return out


def _sns_constraint_fields(constraints: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[float]]:
    names: List[str] = []
    types: List[str] = []
    thresholds: List[float] = []
    for c in constraints or []:
        try:
            names.append(str(c.get("metric") or ""))
            types.append(str(c.get("type") or ""))
            thresholds.append(float(c.get("value")))
        except Exception:
            continue
    return names, types, thresholds


def _instance_constraint_report(result: Dict[str, Any], constraints: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_violation = 0.0
    violations: List[Dict[str, Any]] = []
    safe_globals = {
        "result": result,
        "abs": abs,
        "min": min,
        "max": max,
        "pow": pow,
        "sqrt": lambda x: x**0.5,
        "float": float,
        "__builtins__": {},
    }
    for c in constraints or []:
        if not isinstance(c, dict):
            continue
        metric = str(c.get("metric") or "").strip()
        ctype = str(c.get("type") or "").strip().lower()
        if not metric or not ctype:
            continue
        try:
            mv = result.get(metric)
            if mv is None:
                mv = eval(metric, safe_globals)
            actual = float(mv)
            target = float(c.get("value"))
            tol = float(c.get("tol", 0.0) or 0.0)
        except Exception:
            continue
        vio = 0.0
        if ctype in {"<=", "le"}:
            vio = max(0.0, actual - target - tol)
        elif ctype in {">=", "ge"}:
            vio = max(0.0, target - actual - tol)
        elif ctype in {"==", "eq"}:
            vio = max(0.0, abs(actual - target) - tol)
        elif ctype == "<":
            vio = max(0.0, actual - target)
        elif ctype == ">":
            vio = max(0.0, target - actual)
        if vio > 0.0:
            total_violation += float(vio)
            violations.append({"metric": metric, "type": ctype, "target": float(target), "actual": float(actual), "violation": float(vio)})
    return {"total_violation": float(total_violation), "violations": violations}


def _sample_subrange(rng: random.Random, base_lo: float, base_hi: float, min_frac: float = 0.4, max_frac: float = 1.0) -> Tuple[float, float]:
    lo0 = float(min(base_lo, base_hi))
    hi0 = float(max(base_lo, base_hi))
    span = max(1e-9, hi0 - lo0)
    frac_lo = max(1e-6, float(min(min_frac, max_frac)))
    frac_hi = max(frac_lo, float(max(min_frac, max_frac)))
    width = span * float(rng.uniform(frac_lo, frac_hi))
    width = min(width, span)
    center_lo = lo0 + 0.5 * width
    center_hi = hi0 - 0.5 * width
    if center_hi <= center_lo:
        return float(lo0), float(hi0)
    center = float(rng.uniform(center_lo, center_hi))
    return float(center - 0.5 * width), float(center + 0.5 * width)


def _choose_opt_params(rng: random.Random, pool: List[str], dim_choices: Any) -> List[str]:
    pool2 = [str(p) for p in pool if str(p)]
    if not pool2:
        return []
    if isinstance(dim_choices, (list, tuple)):
        dims = [max(1, int(d)) for d in dim_choices]
        dim = int(rng.choice(dims))
    else:
        dim = max(1, int(dim_choices or 1))
    dim = min(dim, len(pool2))
    return list(rng.sample(pool2, dim))


def _extract_target_metric_value(typed_cfg: Dict[str, Any]) -> Tuple[Any, Any]:
    try:
        metric = str(typed_cfg.get("primary_metric") or "").strip() or None
        targets = typed_cfg.get("targets") or {}
        if metric and f"{metric}_target" in targets:
            return metric, targets.get(f"{metric}_target")
        for k, v in targets.items():
            if str(k).endswith("_target"):
                return str(k)[: -len("_target")], v
            return str(k), v
    except Exception:
        pass
    return None, None


def _template_smoke_test(simulator, template_name: str, param_ranges: Dict[str, Tuple[float, float]], probe_endpoints: bool = False) -> Dict[str, Any]:
    def _probe(params: Dict[str, float]) -> Dict[str, Any]:
        t0 = time.time()
        res = simulator.run_simulation(template_name, params)
        dt = float(time.time() - t0)
        numeric = {}
        if isinstance(res, dict):
            for k, v in res.items():
                try:
                    numeric[str(k)] = float(v)
                except Exception:
                    continue
        return {
            "success": bool(res.get("success", False)) if isinstance(res, dict) else False,
            "execution_time": dt,
            "error": res.get("error") if isinstance(res, dict) else None,
            "numeric": numeric,
        }

    mid_params = {k: float(0.5 * (float(v[0]) + float(v[1]))) for k, v in param_ranges.items()}
    mid = _probe(mid_params)
    out = {
        "mid_params": mid_params,
        "success": bool(mid.get("success", False)),
        "execution_time_mid": mid.get("execution_time"),
        "error_mid": mid.get("error"),
    }
    for k, v in (mid.get("numeric") or {}).items():
        out[f"{k}_mid"] = float(v)

    if probe_endpoints:
        lo_params = {k: float(v[0]) for k, v in param_ranges.items()}
        hi_params = {k: float(v[1]) for k, v in param_ranges.items()}
        lo = _probe(lo_params)
        hi = _probe(hi_params)
        out["success_lo"] = bool(lo.get("success", False))
        out["success_hi"] = bool(hi.get("success", False))
        out["execution_time_lo"] = lo.get("execution_time")
        out["execution_time_hi"] = hi.get("execution_time")
        for k in sorted(set((lo.get("numeric") or {}).keys()) | set((hi.get("numeric") or {}).keys()) | set((mid.get("numeric") or {}).keys())):
            out[f"{k}_lo"] = (lo.get("numeric") or {}).get(k)
            out[f"{k}_hi"] = (hi.get("numeric") or {}).get(k)
    return out


def _heat_metric_estimate_ranges(param_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
    def _mean(names: List[str], idx: int, default: float) -> float:
        vals = []
        for n in names:
            if n in param_ranges:
                vals.append(float(param_ranges[n][idx]))
        if vals:
            return float(sum(vals) / float(len(vals)))
        return float(default)

    k_names = [f"k{i}" for i in range(1, 6)]
    k_lo = _mean(k_names, 0, 0.5)
    k_hi = _mean(k_names, 1, 0.8)
    bi_lo = float(param_ranges.get("Bi", (0.1, 0.1))[0])
    bi_hi = float(param_ranges.get("Bi", (0.5, 0.5))[1])

    def _eff(kv: float, bi: float) -> float:
        return max(0.01, min(0.95, 0.05 + 0.30 * (kv / (kv + 0.35)) * (bi / (bi + 0.20))))

    def _flux(eff: float, bi: float) -> float:
        return max(1e-4, eff * (0.06 + 0.06 * bi))

    def _mean_temp(kv: float, bi: float) -> float:
        return max(0.02, min(1.0, 0.10 + 0.55 / (1.0 + 1.2 * kv + 1.6 * bi)))

    def _max_temp(kv: float, bi: float) -> float:
        return max(0.10, min(1.2, 0.25 + 0.90 / (1.0 + 0.6 * kv + 1.0 * bi)))

    def _uniformity(kv: float, bi: float) -> float:
        return max(0.002, min(0.10, 0.07 / (1.0 + 1.1 * kv + 0.6 * bi)))

    eff_lo = _eff(k_lo, bi_lo)
    eff_hi = _eff(k_hi, bi_hi)
    flux_lo = _flux(eff_lo, bi_lo)
    flux_hi = _flux(eff_hi, bi_hi)
    mt_lo = _mean_temp(k_hi, bi_hi)
    mt_hi = _mean_temp(k_lo, bi_lo)
    mx_lo = _max_temp(k_hi, bi_hi)
    mx_hi = _max_temp(k_lo, bi_lo)
    tu_lo = _uniformity(k_hi, bi_hi)
    tu_hi = _uniformity(k_lo, bi_lo)
    return {
        "efficiency": (min(eff_lo, eff_hi), max(eff_lo, eff_hi)),
        "totalheatflux": (min(flux_lo, flux_hi), max(flux_lo, flux_hi)),
        "meantemp": (min(mt_lo, mt_hi), max(mt_lo, mt_hi)),
        "maxtemp": (min(mx_lo, mx_hi), max(mx_lo, mx_hi)),
        "abs(result.get('tempuniformity', 1e10))": (min(tu_lo, tu_hi), max(tu_lo, tu_hi)),
    }


def _heat_family_v1_generate_instances(cfg: dict, budget: Dict[str, Any], seeds: List[int]) -> List[Dict[str, Any]]:
    count = int(cfg.get("heat_family_count", cfg.get("auto_max_total_instances", 200)) or 0)
    base_seed = int(cfg.get("heat_family_seed", cfg.get("auto_seed", 0)) or 0)
    problem_type = str(cfg.get("heat_family_problem_type", "thermal_fins_family_v1"))
    width_frac = cfg.get("heat_family_width_frac_range") or [0.45, 1.0]
    wf_lo = float(width_frac[0])
    wf_hi = float(width_frac[1])
    param_space = cfg.get("heat_family_param_space") or {
        "k1": [0.1, 1.0], "k2": [0.1, 1.0], "k3": [0.1, 1.0], "k4": [0.1, 1.0], "k5": [0.1, 1.0], "Bi": [0.01, 1.0]
    }
    task_catalog = cfg.get("heat_family_task_catalog") or [
        {"name": "max_eff_2d", "objective_type": "max", "task_type": "design_optimization", "metric": "efficiency", "param_pool": list(param_space.keys()), "dim_choices": [2], "weight": 0.18},
        {"name": "max_eff_4d_safe", "objective_type": "max", "task_type": "design_optimization", "metric": "efficiency", "param_pool": list(param_space.keys()), "dim_choices": [4], "constraints": [{"metric": "maxtemp", "type": "<=", "anchor": "mid", "ratio_range": [0.95, 1.05], "penalty_weight": 1200.0}], "weight": 0.16},
        {"name": "max_heatflux_3d", "objective_type": "max", "task_type": "design_optimization", "metric": "totalheatflux", "param_pool": list(param_space.keys()), "dim_choices": [3], "weight": 0.17},
        {"name": "min_meantemp_3d", "objective_type": "min", "task_type": "design_optimization", "metric": "meantemp", "param_pool": list(param_space.keys()), "dim_choices": [3], "weight": 0.17},
        {"name": "min_maxtemp_3d", "objective_type": "min", "task_type": "design_optimization", "metric": "maxtemp", "param_pool": list(param_space.keys()), "dim_choices": [3], "weight": 0.16},
        {"name": "min_uniformity_4d", "objective_type": "min", "task_type": "design_optimization", "metric": "tempuniformity", "param_pool": list(param_space.keys()), "dim_choices": [4], "weight": 0.16},
    ]
    weights = [max(1e-9, float(t.get("weight", 1.0))) for t in task_catalog]
    rng = random.Random(_stable_int(f"heat_family_v1:{base_seed}"))
    instances: List[Dict[str, Any]] = []
    for _ in range(max(0, count)):
        task = rng.choices(task_catalog, weights=weights, k=1)[0]
        opt_params = _choose_opt_params(rng, list(task.get("param_pool") or list(param_space.keys())), task.get("dim_choices") or [3])
        param_ranges = {}
        for p in opt_params:
            lo0, hi0 = param_space[p]
            param_ranges[p] = _sample_subrange(rng, float(lo0), float(hi0), wf_lo, wf_hi)
        metric = str(task.get("metric", "efficiency"))
        objective_type = str(task.get("objective_type", "max"))
        objective_metric = "abs(result.get('tempuniformity', 1e10))" if metric == "tempuniformity" and objective_type != "max" else metric
        objective_expression = _sns_objective_expr(objective_metric, objective_type)
        est_ranges = _heat_metric_estimate_ranges(param_ranges)
        constraints = _sns_build_constraints(rng, est_ranges, task.get("constraints") or [])
        c_names, c_types, c_thresholds = _sns_constraint_fields(constraints)
        typed_config = {
            "instance_id": None,
            "physics_domain": "heat",
            "task_type": str(task.get("task_type", "design_optimization")),
            "objective_type": objective_type,
            "decision_dim": len(opt_params),
            "variable_name_json": list(opt_params),
            "var_lower_bound_json": [float(param_ranges[p][0]) for p in opt_params],
            "var_upper_bound_json": [float(param_ranges[p][1]) for p in opt_params],
            "num_constraints": len(constraints),
            "constraint_name_json": c_names,
            "constraint_type_json": c_types,
            "constraint_threshold_json": c_thresholds,
            "constraints": constraints,
            "budget_eval": int(budget.get("max_iterations", 0)),
            "budget_walltime_sec": float(budget.get("time_limit_s", 0.0)),
            "template_name": "thmf",
            "targets": {},
            "generation_bucket": str(task.get("name", "mixed")),
            "family_version": "heat_v1",
            "task_name": str(task.get("name", "mixed")),
            "primary_metric": metric,
            "metric_estimates": {k: {"lo": float(v[0]), "hi": float(v[1])} for k, v in est_ranges.items()},
        }
        payload = {
            "problem_type": problem_type,
            "opt_params": list(opt_params),
            "objective_expression": objective_expression,
            "param_ranges": param_ranges,
            "budget": {"max_iterations": int(budget.get("max_iterations", 0)), "time_limit_s": float(budget.get("time_limit_s", 0.0))},
            "seeds": [int(s) for s in seeds],
            "typed_config_json": typed_config,
        }
        instance_id = _make_stable_instance_id_v2(payload)
        typed_config["instance_id"] = instance_id
        instances.append({"instance_id": instance_id, "problem_type": problem_type, "opt_params": list(opt_params), "param_ranges": param_ranges, "objective_expression": objective_expression, "typed_config_json": typed_config})
    return instances


def _iaea_family_v1_generate_instances(cfg: dict, budget: Dict[str, Any], seeds: List[int]) -> List[Dict[str, Any]]:
    count = int(cfg.get("iaea_family_count", cfg.get("auto_max_total_instances", 200)) or 0)
    base_seed = int(cfg.get("iaea_family_seed", cfg.get("auto_seed", 0)) or 0)
    problem_type = str(cfg.get("iaea_family_problem_type", "neutron_diffusion_family_v1"))
    width_frac = cfg.get("iaea_family_width_frac_range") or [0.35, 1.0]
    wf_lo = float(width_frac[0])
    wf_hi = float(width_frac[1])
    param_space = cfg.get("iaea_family_param_space") or {
        "D11": [1.35, 1.65], "D12": [1.35, 1.65], "D13": [1.35, 1.65], "D14": [1.8, 2.2],
        "D21": [0.35, 0.45], "D22": [0.35, 0.45], "D23": [0.35, 0.45], "D24": [0.25, 0.35],
        "Sigmaa11": [0.005, 0.015], "Sigmaa12": [0.005, 0.015], "Sigmaa13": [0.005, 0.015], "Sigmaa14": [0.0, 0.005],
        "Sigmaa21": [0.06, 0.10], "Sigmaa22": [0.07, 0.10], "Sigmaa23": [0.10, 0.16], "Sigmaa24": [0.005, 0.015],
        "musigmaf11": [0.0, 0.01], "musigmaf12": [0.0, 0.01], "musigmaf13": [0.0, 0.01], "musigmaf14": [0.0, 0.01],
        "musigmaf21": [0.12, 0.15], "musigmaf22": [0.12, 0.15], "musigmaf23": [0.12, 0.15], "musigmaf24": [0.0, 0.01],
        "chi11": [0.9, 1.1], "chi12": [0.9, 1.1], "chi13": [0.9, 1.1], "chi14": [0.9, 1.1],
        "chi21": [0.0, 0.1], "chi22": [0.0, 0.1], "chi23": [0.0, 0.1], "chi24": [0.0, 0.1],
        "Sigmas121": [0.015, 0.025], "Sigmas122": [0.015, 0.025], "Sigmas123": [0.015, 0.025], "Sigmas124": [0.03, 0.05],
    }
    core_pool = ["Sigmaa21", "Sigmaa22", "Sigmaa23", "musigmaf21", "musigmaf22", "musigmaf23", "D21", "D22", "D23"]
    shape_pool = ["D11", "D12", "D13", "Sigmaa11", "Sigmaa12", "Sigmaa13", "Sigmas121", "Sigmas122", "Sigmas123"]
    refl_pool = ["D14", "D24", "Sigmaa14", "Sigmaa24", "Sigmas124"]
    task_catalog = cfg.get("iaea_family_task_catalog") or [
        {"name": "calib_keff_core_2d", "objective_type": "target_matching", "task_type": "calibration", "metric": "keff", "param_pool": core_pool, "dim_choices": [2], "target_range": [0.995, 1.045], "weight": 0.24},
        {"name": "calib_keff_mix_4d", "objective_type": "target_matching", "task_type": "calibration", "metric": "keff", "param_pool": core_pool + shape_pool, "dim_choices": [4], "target_range": [0.99, 1.05], "weight": 0.22},
        {"name": "min_fq_keffband_4d", "objective_type": "min", "task_type": "design_optimization", "metric": "FQ", "param_pool": shape_pool + refl_pool, "dim_choices": [4], "constraints": [{"metric": "keff", "type": ">=", "value_range": [0.998, 1.020], "penalty_weight": 1500.0}, {"metric": "keff", "type": "<=", "value_range": [1.030, 1.055], "penalty_weight": 1500.0}], "weight": 0.22},
        {"name": "max_keff_safe_4d", "objective_type": "max", "task_type": "design_optimization", "metric": "keff", "param_pool": core_pool + refl_pool, "dim_choices": [4], "constraints": [{"metric": "FQ", "type": "<=", "value_range": [2.35, 2.95], "penalty_weight": 1200.0}], "weight": 0.18},
        {"name": "min_fq_core_6d", "objective_type": "min", "task_type": "design_optimization", "metric": "FQ", "param_pool": core_pool + shape_pool, "dim_choices": [6], "constraints": [{"metric": "keff", "type": ">=", "value_range": [0.99, 1.02], "penalty_weight": 1500.0}], "weight": 0.14},
    ]
    bucket_ratios = cfg.get("iaea_family_target_bucket_ratios") or {"mid": 0.55, "edge": 0.25, "outside": 0.20}
    r_mid = float(bucket_ratios.get("mid", 0.55))
    r_edge = float(bucket_ratios.get("edge", 0.25))
    r_out = float(bucket_ratios.get("outside", 0.20))
    s = max(1e-12, r_mid + r_edge + r_out)
    r_mid, r_edge, r_out = r_mid / s, r_edge / s, r_out / s
    weights = [max(1e-9, float(t.get("weight", 1.0))) for t in task_catalog]
    rng = random.Random(_stable_int(f"iaea_family_v1:{base_seed}"))
    instances: List[Dict[str, Any]] = []
    for _ in range(max(0, count)):
        task = rng.choices(task_catalog, weights=weights, k=1)[0]
        opt_params = _choose_opt_params(rng, list(task.get("param_pool") or list(param_space.keys())), task.get("dim_choices") or [4])
        param_ranges = {}
        for p in opt_params:
            lo0, hi0 = param_space[p]
            param_ranges[p] = _sample_subrange(rng, float(lo0), float(hi0), wf_lo, wf_hi)
        metric = str(task.get("metric", "keff"))
        objective_type = str(task.get("objective_type", "target_matching"))
        targets: Dict[str, float] = {}
        bucket = str(task.get("name", "mixed"))
        if objective_type == "target_matching":
            tgt_lo, tgt_hi = task.get("target_range") or [0.99, 1.05]
            target_value, bucket = _sns_sample_target_value(rng, float(tgt_lo), float(tgt_hi), r_mid, r_edge, r_out, 0.20, 0.15, 0.02)
            targets = {metric: float(target_value)}
            objective_expression = _sns_objective_expr(metric, objective_type, target_value)
        else:
            objective_expression = _sns_objective_expr(metric, objective_type)
        constraints = _sns_build_constraints(rng, {}, task.get("constraints") or [])
        c_names, c_types, c_thresholds = _sns_constraint_fields(constraints)
        typed_config = {
            "instance_id": None,
            "physics_domain": "neutronics",
            "task_type": str(task.get("task_type", "design_optimization")),
            "objective_type": objective_type,
            "decision_dim": len(opt_params),
            "variable_name_json": list(opt_params),
            "var_lower_bound_json": [float(param_ranges[p][0]) for p in opt_params],
            "var_upper_bound_json": [float(param_ranges[p][1]) for p in opt_params],
            "num_constraints": len(constraints),
            "constraint_name_json": c_names,
            "constraint_type_json": c_types,
            "constraint_threshold_json": c_thresholds,
            "constraints": constraints,
            "budget_eval": int(budget.get("max_iterations", 0)),
            "budget_walltime_sec": float(budget.get("time_limit_s", 0.0)),
            "template_name": "iaea",
            "targets": targets,
            "generation_bucket": bucket,
            "family_version": "iaea_v1",
            "task_name": str(task.get("name", "mixed")),
            "primary_metric": metric,
        }
        payload = {
            "problem_type": problem_type,
            "opt_params": list(opt_params),
            "objective_expression": objective_expression,
            "param_ranges": param_ranges,
            "budget": {"max_iterations": int(budget.get("max_iterations", 0)), "time_limit_s": float(budget.get("time_limit_s", 0.0))},
            "seeds": [int(s) for s in seeds],
            "typed_config_json": typed_config,
        }
        instance_id = _make_stable_instance_id_v2(payload)
        typed_config["instance_id"] = instance_id
        instances.append({"instance_id": instance_id, "problem_type": problem_type, "opt_params": list(opt_params), "param_ranges": param_ranges, "objective_expression": objective_expression, "typed_config_json": typed_config})
    return instances


def _heat_time_family_v1_generate_instances(cfg: dict, budget: Dict[str, Any], seeds: List[int]) -> List[Dict[str, Any]]:
    count = int(cfg.get("heat_time_family_count", cfg.get("auto_max_total_instances", 200)) or 0)
    base_seed = int(cfg.get("heat_time_family_seed", cfg.get("auto_seed", 0)) or 0)
    problem_type = str(cfg.get("heat_time_family_problem_type", "ex_heat_time_family_v1"))
    width_frac = cfg.get("heat_time_family_width_frac_range") or [0.35, 1.0]
    wf_lo = float(width_frac[0])
    wf_hi = float(width_frac[1])
    param_space = cfg.get("heat_time_family_param_space") or {
        "k_hi": [0.6, 3.0],
        "k_lo": [0.05, 1.5],
        "kf": [0.4, 3.0],
        "ue": [5.0, 40.0],
        "T": [1.0, 12.0],
        "dt": [0.02, 0.5],
    }
    task_catalog = cfg.get("heat_time_family_task_catalog") or [
        {"name": "calib_u_mean_3d", "objective_type": "target_matching", "task_type": "calibration", "metric": "u_mean", "param_pool": list(param_space.keys()), "dim_choices": [3], "target_range": [26.0, 40.0], "weight": 0.55},
        {"name": "calib_u_max_3d", "objective_type": "target_matching", "task_type": "calibration", "metric": "u_max", "param_pool": list(param_space.keys()), "dim_choices": [3], "target_range": [28.0, 48.0], "weight": 0.25},
        {"name": "min_cpu_2d", "objective_type": "min", "task_type": "design_optimization", "metric": "cpu", "param_pool": list(param_space.keys()), "dim_choices": [2], "weight": 0.20},
    ]
    weights = [max(1e-9, float(t.get("weight", 1.0))) for t in task_catalog]
    bucket_ratios = cfg.get("heat_time_family_target_bucket_ratios") or {"mid": 0.60, "edge": 0.25, "outside": 0.15}
    r_mid = float(bucket_ratios.get("mid", 0.60))
    r_edge = float(bucket_ratios.get("edge", 0.25))
    r_out = float(bucket_ratios.get("outside", 0.15))
    s = max(1e-12, r_mid + r_edge + r_out)
    r_mid, r_edge, r_out = r_mid / s, r_edge / s, r_out / s
    rng = random.Random(_stable_int(f"heat_time_family_v1:{base_seed}"))
    instances: List[Dict[str, Any]] = []
    for _ in range(max(0, count)):
        task = rng.choices(task_catalog, weights=weights, k=1)[0]
        opt_params = _choose_opt_params(rng, list(task.get("param_pool") or list(param_space.keys())), task.get("dim_choices") or [3])
        param_ranges = {}
        for p in opt_params:
            lo0, hi0 = param_space[p]
            param_ranges[p] = _sample_subrange(rng, float(lo0), float(hi0), wf_lo, wf_hi)
        metric = str(task.get("metric", "u_mean"))
        objective_type = str(task.get("objective_type", "target_matching"))
        targets: Dict[str, float] = {}
        bucket = str(task.get("name", "mixed"))
        if objective_type == "target_matching":
            tgt_lo, tgt_hi = task.get("target_range") or [0.0, 1.0]
            target_value, bucket = _sns_sample_target_value(rng, float(tgt_lo), float(tgt_hi), r_mid, r_edge, r_out, 0.20, 0.15, 0.02 * max(1.0, abs(float(tgt_hi) - float(tgt_lo))))
            targets = {f"{metric}_target": float(target_value)}
            if metric == "u_mean":
                targets = {"u_mean_target": float(target_value)}
            objective_expression = _objective_expression_for_metric(metric, targets)
        else:
            objective_expression = _objective_expression_for_metric(metric, {})
        typed_config = {
            "instance_id": None,
            "physics_domain": "heat",
            "task_type": str(task.get("task_type", "calibration")),
            "objective_type": objective_type,
            "decision_dim": len(opt_params),
            "variable_name_json": list(opt_params),
            "var_lower_bound_json": [float(param_ranges[p][0]) for p in opt_params],
            "var_upper_bound_json": [float(param_ranges[p][1]) for p in opt_params],
            "num_constraints": 0,
            "constraint_name_json": [],
            "constraint_type_json": [],
            "constraint_threshold_json": [],
            "constraints": [],
            "budget_eval": int(budget.get("max_iterations", 0)),
            "budget_walltime_sec": float(budget.get("time_limit_s", 0.0)),
            "template_name": "ex_heat_time",
            "targets": targets,
            "generation_bucket": bucket,
            "family_version": "heat_time_v1",
            "task_name": str(task.get("name", "mixed")),
            "primary_metric": metric,
        }
        payload = {
            "problem_type": problem_type,
            "opt_params": list(opt_params),
            "objective_expression": objective_expression,
            "param_ranges": param_ranges,
            "budget": {"max_iterations": int(budget.get("max_iterations", 0)), "time_limit_s": float(budget.get("time_limit_s", 0.0))},
            "seeds": [int(s) for s in seeds],
            "typed_config_json": typed_config,
        }
        instance_id = _make_stable_instance_id_v2(payload)
        typed_config["instance_id"] = instance_id
        instances.append({"instance_id": instance_id, "problem_type": problem_type, "opt_params": list(opt_params), "param_ranges": param_ranges, "objective_expression": objective_expression, "typed_config_json": typed_config})
    return instances


def _blackscholes2d_family_v1_generate_instances(cfg: dict, budget: Dict[str, Any], seeds: List[int]) -> List[Dict[str, Any]]:
    count = int(cfg.get("blackscholes_family_count", cfg.get("auto_max_total_instances", 200)) or 0)
    base_seed = int(cfg.get("blackscholes_family_seed", cfg.get("auto_seed", 0)) or 0)
    problem_type = str(cfg.get("blackscholes_family_problem_type", "ex_blackscholes2d_family_v1"))
    width_frac = cfg.get("blackscholes_family_width_frac_range") or [0.35, 1.0]
    wf_lo = float(width_frac[0])
    wf_hi = float(width_frac[1])
    param_space = cfg.get("blackscholes_family_param_space") or {
        "sigma1": [0.05, 0.80],
        "sigma2": [0.05, 0.80],
        "rho": [-0.75, 0.75],
        "r": [0.0, 0.20],
        "K": [10.0, 80.0],
        "dt": [0.002, 0.05],
        "T": [0.2, 2.0],
    }
    task_catalog = cfg.get("blackscholes_family_task_catalog") or [
        {"name": "calib_u_mean_3d", "objective_type": "target_matching", "task_type": "calibration", "metric": "u_mean", "param_pool": list(param_space.keys()), "dim_choices": [3], "target_range": [1.0, 12.0], "weight": 0.55},
        {"name": "calib_u_min_3d", "objective_type": "target_matching", "task_type": "calibration", "metric": "u_min", "param_pool": list(param_space.keys()), "dim_choices": [3], "target_range": [0.0, 6.0], "weight": 0.20},
        {"name": "min_u_mean_4d", "objective_type": "min", "task_type": "design_optimization", "metric": "u_mean", "param_pool": list(param_space.keys()), "dim_choices": [4], "weight": 0.25},
    ]
    weights = [max(1e-9, float(t.get("weight", 1.0))) for t in task_catalog]
    bucket_ratios = cfg.get("blackscholes_family_target_bucket_ratios") or {"mid": 0.60, "edge": 0.25, "outside": 0.15}
    r_mid = float(bucket_ratios.get("mid", 0.60))
    r_edge = float(bucket_ratios.get("edge", 0.25))
    r_out = float(bucket_ratios.get("outside", 0.15))
    s = max(1e-12, r_mid + r_edge + r_out)
    r_mid, r_edge, r_out = r_mid / s, r_edge / s, r_out / s
    rng = random.Random(_stable_int(f"blackscholes_family_v1:{base_seed}"))
    instances: List[Dict[str, Any]] = []
    for _ in range(max(0, count)):
        task = rng.choices(task_catalog, weights=weights, k=1)[0]
        opt_params = _choose_opt_params(rng, list(task.get("param_pool") or list(param_space.keys())), task.get("dim_choices") or [3])
        param_ranges = {}
        for p in opt_params:
            lo0, hi0 = param_space[p]
            param_ranges[p] = _sample_subrange(rng, float(lo0), float(hi0), wf_lo, wf_hi)
        metric = str(task.get("metric", "u_mean"))
        objective_type = str(task.get("objective_type", "target_matching"))
        targets: Dict[str, float] = {}
        bucket = str(task.get("name", "mixed"))
        if objective_type == "target_matching":
            tgt_lo, tgt_hi = task.get("target_range") or [0.0, 1.0]
            target_value, bucket = _sns_sample_target_value(rng, float(tgt_lo), float(tgt_hi), r_mid, r_edge, r_out, 0.20, 0.15, 0.02 * max(1.0, abs(float(tgt_hi) - float(tgt_lo))))
            targets = {f"{metric}_target": float(target_value)}
            if metric == "u_mean":
                targets = {"u_mean_target": float(target_value)}
            objective_expression = _objective_expression_for_metric(metric, targets)
        else:
            objective_expression = _objective_expression_for_metric(metric, {})
        typed_config = {
            "instance_id": None,
            "physics_domain": "finance",
            "task_type": str(task.get("task_type", "calibration")),
            "objective_type": objective_type,
            "decision_dim": len(opt_params),
            "variable_name_json": list(opt_params),
            "var_lower_bound_json": [float(param_ranges[p][0]) for p in opt_params],
            "var_upper_bound_json": [float(param_ranges[p][1]) for p in opt_params],
            "num_constraints": 0,
            "constraint_name_json": [],
            "constraint_type_json": [],
            "constraint_threshold_json": [],
            "constraints": [],
            "budget_eval": int(budget.get("max_iterations", 0)),
            "budget_walltime_sec": float(budget.get("time_limit_s", 0.0)),
            "template_name": "ex_blackscholes2d",
            "targets": targets,
            "generation_bucket": bucket,
            "family_version": "blackscholes_v1",
            "task_name": str(task.get("name", "mixed")),
            "primary_metric": metric,
        }
        payload = {
            "problem_type": problem_type,
            "opt_params": list(opt_params),
            "objective_expression": objective_expression,
            "param_ranges": param_ranges,
            "budget": {"max_iterations": int(budget.get("max_iterations", 0)), "time_limit_s": float(budget.get("time_limit_s", 0.0))},
            "seeds": [int(s) for s in seeds],
            "typed_config_json": typed_config,
        }
        instance_id = _make_stable_instance_id_v2(payload)
        typed_config["instance_id"] = instance_id
        instances.append({"instance_id": instance_id, "problem_type": problem_type, "opt_params": list(opt_params), "param_ranges": param_ranges, "objective_expression": objective_expression, "typed_config_json": typed_config})
    return instances


def _advection2d_family_v1_generate_instances(cfg: dict, budget: Dict[str, Any], seeds: List[int]) -> List[Dict[str, Any]]:
    count = int(cfg.get("advection_family_count", cfg.get("auto_max_total_instances", 200)) or 0)
    base_seed = int(cfg.get("advection_family_seed", cfg.get("auto_seed", 0)) or 0)
    problem_type = str(cfg.get("advection_family_problem_type", "ex_advection2d_family_v1"))
    width_frac = cfg.get("advection_family_width_frac_range") or [0.35, 1.0]
    wf_lo = float(width_frac[0])
    wf_hi = float(width_frac[1])
    param_space = cfg.get("advection_family_param_space") or {
        "a": [-3.0, 3.0],
        "b": [-3.0, 3.0],
        "dt": [0.002, 0.08],
        "T": [0.2, 3.0],
        "sigma": [0.2, 2.0],
        "amp": [0.2, 3.0],
    }
    task_catalog = cfg.get("advection_family_task_catalog") or [
        {"name": "calib_u_mean_3d", "objective_type": "target_matching", "task_type": "calibration", "metric": "u_mean", "param_pool": list(param_space.keys()), "dim_choices": [3], "target_range": [0.02, 0.20], "weight": 0.55},
        {"name": "calib_u_max_3d", "objective_type": "target_matching", "task_type": "calibration", "metric": "u_max", "param_pool": list(param_space.keys()), "dim_choices": [3], "target_range": [0.05, 1.20], "weight": 0.25},
        {"name": "min_cpu_2d", "objective_type": "min", "task_type": "design_optimization", "metric": "cpu", "param_pool": list(param_space.keys()), "dim_choices": [2], "weight": 0.20},
    ]
    weights = [max(1e-9, float(t.get("weight", 1.0))) for t in task_catalog]
    bucket_ratios = cfg.get("advection_family_target_bucket_ratios") or {"mid": 0.60, "edge": 0.25, "outside": 0.15}
    r_mid = float(bucket_ratios.get("mid", 0.60))
    r_edge = float(bucket_ratios.get("edge", 0.25))
    r_out = float(bucket_ratios.get("outside", 0.15))
    s = max(1e-12, r_mid + r_edge + r_out)
    r_mid, r_edge, r_out = r_mid / s, r_edge / s, r_out / s
    rng = random.Random(_stable_int(f"advection_family_v1:{base_seed}"))
    instances: List[Dict[str, Any]] = []
    for _ in range(max(0, count)):
        task = rng.choices(task_catalog, weights=weights, k=1)[0]
        opt_params = _choose_opt_params(rng, list(task.get("param_pool") or list(param_space.keys())), task.get("dim_choices") or [3])
        param_ranges = {}
        for p in opt_params:
            lo0, hi0 = param_space[p]
            param_ranges[p] = _sample_subrange(rng, float(lo0), float(hi0), wf_lo, wf_hi)
        metric = str(task.get("metric", "u_mean"))
        objective_type = str(task.get("objective_type", "target_matching"))
        targets: Dict[str, float] = {}
        bucket = str(task.get("name", "mixed"))
        if objective_type == "target_matching":
            tgt_lo, tgt_hi = task.get("target_range") or [0.0, 1.0]
            target_value, bucket = _sns_sample_target_value(rng, float(tgt_lo), float(tgt_hi), r_mid, r_edge, r_out, 0.20, 0.15, 0.05 * max(1.0, abs(float(tgt_hi) - float(tgt_lo))))
            targets = {f"{metric}_target": float(target_value)}
            if metric == "u_mean":
                targets = {"u_mean_target": float(target_value)}
            objective_expression = _objective_expression_for_metric(metric, targets)
        else:
            objective_expression = _objective_expression_for_metric(metric, {})
        typed_config = {
            "instance_id": None,
            "physics_domain": "fluid",
            "task_type": str(task.get("task_type", "calibration")),
            "objective_type": objective_type,
            "decision_dim": len(opt_params),
            "variable_name_json": list(opt_params),
            "var_lower_bound_json": [float(param_ranges[p][0]) for p in opt_params],
            "var_upper_bound_json": [float(param_ranges[p][1]) for p in opt_params],
            "num_constraints": 0,
            "constraint_name_json": [],
            "constraint_type_json": [],
            "constraint_threshold_json": [],
            "constraints": [],
            "budget_eval": int(budget.get("max_iterations", 0)),
            "budget_walltime_sec": float(budget.get("time_limit_s", 0.0)),
            "template_name": "ex_advection2d",
            "targets": targets,
            "generation_bucket": bucket,
            "family_version": "advection_v1",
            "task_name": str(task.get("name", "mixed")),
            "primary_metric": metric,
        }
        payload = {
            "problem_type": problem_type,
            "opt_params": list(opt_params),
            "objective_expression": objective_expression,
            "param_ranges": param_ranges,
            "budget": {"max_iterations": int(budget.get("max_iterations", 0)), "time_limit_s": float(budget.get("time_limit_s", 0.0))},
            "seeds": [int(s) for s in seeds],
            "typed_config_json": typed_config,
        }
        instance_id = _make_stable_instance_id_v2(payload)
        typed_config["instance_id"] = instance_id
        instances.append({"instance_id": instance_id, "problem_type": problem_type, "opt_params": list(opt_params), "param_ranges": param_ranges, "objective_expression": objective_expression, "typed_config_json": typed_config})
    return instances


def _sns_flow_family_v2_generate_instances(cfg: dict, budget: Dict[str, Any], seeds: List[int]) -> List[Dict[str, Any]]:
    count = int(cfg.get("sns_family_count", cfg.get("auto_max_total_instances", 200)) or 0)
    base_seed = int(cfg.get("sns_family_seed", cfg.get("auto_seed", 0)) or 0)
    problem_type = str(cfg.get("sns_family_problem_type", "sns_flow_family_v2"))

    u_lower_lo, u_lower_hi = cfg.get("sns_family_u_lower_range", [3.0, 8.0])
    u_upper_lo, u_upper_hi = cfg.get("sns_family_u_upper_range", [12.0, 20.0])
    min_u_width = float(cfg.get("sns_family_min_u_width", 6.0))
    max_u_width = cfg.get("sns_family_max_u_width", None)
    max_u_width = float(max_u_width) if max_u_width is not None else None

    mu_lower_lo, mu_lower_hi = cfg.get("sns_family_mu_lower_range", [0.03, 0.10])
    mu_upper_lo, mu_upper_hi = cfg.get("sns_family_mu_upper_range", [0.12, 0.30])
    min_mu_width = float(cfg.get("sns_family_min_mu_width", 0.04))
    max_mu_width = cfg.get("sns_family_max_mu_width", None)
    max_mu_width = float(max_mu_width) if max_mu_width is not None else None

    r_lower_lo, r_lower_hi = cfg.get("sns_family_r_lower_range", [0.10, 0.16])
    r_upper_lo, r_upper_hi = cfg.get("sns_family_r_upper_range", [0.18, 0.30])
    min_r_width = float(cfg.get("sns_family_min_r_width", 0.04))
    max_r_width = cfg.get("sns_family_max_r_width", None)
    max_r_width = float(max_r_width) if max_r_width is not None else None

    bucket_ratios = cfg.get("sns_family_target_bucket_ratios") or {}
    r_mid = float(bucket_ratios.get("mid", 0.6))
    r_edge = float(bucket_ratios.get("edge", 0.25))
    r_out = float(bucket_ratios.get("outside", 0.15))
    s = max(1e-12, r_mid + r_edge + r_out)
    r_mid, r_edge, r_out = r_mid / s, r_edge / s, r_out / s
    mid_margin_frac = float(cfg.get("sns_family_mid_margin_frac", 0.2))
    edge_frac = float(cfg.get("sns_family_edge_frac", 0.15))
    outside_margin = float(cfg.get("sns_family_outside_margin", 120.0))
    approx_slope = float(cfg.get("sns_family_dp_slope", 107.5))
    approx_intercept = float(cfg.get("sns_family_dp_intercept", -177.5))

    task_catalog = cfg.get("sns_family_task_catalog") or [
        {"name": "calib_dp_1d", "objective_type": "target_matching", "task_type": "calibration", "metric": "pressuredrop", "opt_params": ["uMax"], "weight": 0.22},
        {"name": "calib_dp_2d", "objective_type": "target_matching", "task_type": "calibration", "metric": "pressuredrop", "opt_params": ["uMax", "Mu"], "weight": 0.18},
        {"name": "min_dp_geom", "objective_type": "min", "task_type": "design_optimization", "metric": "pressuredrop", "opt_params": ["uMax", "Mu", "R"], "weight": 0.20},
        {"name": "min_velmax", "objective_type": "min", "task_type": "design_optimization", "metric": "velmagmax", "opt_params": ["uMax", "Mu", "R"], "weight": 0.16},
        {"name": "min_pmax", "objective_type": "min", "task_type": "design_optimization", "metric": "pmax", "opt_params": ["uMax", "R"], "weight": 0.14},
        {"name": "max_dp", "objective_type": "max", "task_type": "design_optimization", "metric": "pressuredrop", "opt_params": ["uMax", "Mu", "R"], "weight": 0.10},
    ]
    weights = [max(1e-9, float(t.get("weight", 1.0))) for t in task_catalog]

    def _sample_range(rng: random.Random, lo_lo: float, lo_hi: float, hi_lo: float, hi_hi: float, min_width: float, max_width: Any = None) -> Tuple[float, float]:
        lo = float(rng.uniform(float(lo_lo), float(lo_hi)))
        hi = float(rng.uniform(float(hi_lo), float(hi_hi)))
        if hi - lo < min_width:
            hi = lo + float(min_width)
        if max_width is not None and (hi - lo) > float(max_width):
            hi = lo + float(max_width)
        hi = max(lo + 1e-9, hi)
        return float(lo), float(hi)

    def _sample_target_dp(rng: random.Random, dp_lo_est: float, dp_hi_est: float) -> Tuple[float, str]:
        if dp_hi_est < dp_lo_est:
            dp_lo_est, dp_hi_est = dp_hi_est, dp_lo_est
        dp_lo_est = max(1.0, float(dp_lo_est))
        dp_hi_est = max(dp_lo_est + 1e-6, float(dp_hi_est))
        span = float(max(1e-9, dp_hi_est - dp_lo_est))
        r = rng.random()
        if r < r_mid:
            lo = float(dp_lo_est + span * float(mid_margin_frac))
            hi = float(dp_hi_est - span * float(mid_margin_frac))
            if hi <= lo:
                lo, hi = float(dp_lo_est), float(dp_hi_est)
            return float(rng.uniform(lo, hi)), "mid"
        if r < (r_mid + r_edge):
            if rng.random() < 0.5:
                lo = float(dp_lo_est)
                hi = float(dp_lo_est + span * float(edge_frac))
            else:
                lo = float(dp_hi_est - span * float(edge_frac))
                hi = float(dp_hi_est)
            if hi <= lo:
                lo, hi = float(dp_lo_est), float(dp_hi_est)
            return float(rng.uniform(lo, hi)), "edge"
        if rng.random() < 0.5:
            return max(1.0, float(dp_lo_est - abs(outside_margin) - abs(rng.gauss(0.0, abs(outside_margin) * 0.25)))), "outside"
        return max(1.0, float(dp_hi_est + abs(outside_margin) + abs(rng.gauss(0.0, abs(outside_margin) * 0.25)))), "outside"

    rng = random.Random(_stable_int(f"sns_flow_family_v2:{base_seed}"))
    instances: List[Dict[str, Any]] = []
    for _ in range(max(0, count)):
        task = rng.choices(task_catalog, weights=weights, k=1)[0]
        opt_params = list(task.get("opt_params") or ["uMax"])

        u_lo, u_hi = _sample_range(rng, float(u_lower_lo), float(u_lower_hi), float(u_upper_lo), float(u_upper_hi), min_u_width, max_u_width)
        mu_lo, mu_hi = _sample_range(rng, float(mu_lower_lo), float(mu_lower_hi), float(mu_upper_lo), float(mu_upper_hi), min_mu_width, max_mu_width)
        r_lo, r_hi = _sample_range(rng, float(r_lower_lo), float(r_lower_hi), float(r_upper_lo), float(r_upper_hi), min_r_width, max_r_width)

        param_ranges: Dict[str, Tuple[float, float]] = {}
        if "uMax" in opt_params:
            param_ranges["uMax"] = (u_lo, u_hi)
        if "Mu" in opt_params:
            param_ranges["Mu"] = (mu_lo, mu_hi)
        if "R" in opt_params:
            param_ranges["R"] = (r_lo, r_hi)

        metric = str(task.get("metric", "pressuredrop"))
        objective_type = str(task.get("objective_type", "min"))
        task_type = str(task.get("task_type", "design_optimization"))
        targets: Dict[str, float] = {}
        bucket = str(task.get("name", "mixed"))

        if objective_type == "target_matching" and metric == "pressuredrop":
            mu_mid = 0.5 * (mu_lo + mu_hi)
            r_mid = 0.5 * (r_lo + r_hi)
            mu_factor = max(0.5, min(2.5, 0.1 / max(1e-6, mu_mid)))
            r_factor = max(0.7, min(1.5, r_mid / 0.2))
            dp_lo_est = (approx_slope * u_lo + approx_intercept) * mu_factor * r_factor
            dp_hi_est = (approx_slope * u_hi + approx_intercept) * mu_factor * r_factor
            target_dp, bucket = _sample_target_dp(rng, float(dp_lo_est), float(dp_hi_est))
            targets = {"pressuredrop_target": float(target_dp)}
            objective_expression = _sns_objective_expr(metric, objective_type, target_dp)
        else:
            objective_expression = _sns_objective_expr(metric, objective_type)

        var_lows = [float(param_ranges[p][0]) for p in opt_params]
        var_highs = [float(param_ranges[p][1]) for p in opt_params]
        typed_config = {
            "instance_id": None,
            "physics_domain": "fluid",
            "task_type": task_type,
            "objective_type": objective_type,
            "decision_dim": len(opt_params),
            "variable_name_json": opt_params,
            "var_lower_bound_json": var_lows,
            "var_upper_bound_json": var_highs,
            "num_constraints": 0,
            "constraint_name_json": [],
            "constraint_type_json": [],
            "constraint_threshold_json": [],
            "budget_eval": int(budget.get("max_iterations", 0)),
            "budget_walltime_sec": float(budget.get("time_limit_s", 0.0)),
            "template_name": "sns",
            "targets": targets,
            "generation_bucket": bucket,
            "family_version": "sns_v2",
            "task_name": str(task.get("name", "mixed")),
            "primary_metric": metric,
        }

        payload = {
            "problem_type": problem_type,
            "opt_params": opt_params,
            "objective_expression": objective_expression,
            "param_ranges": param_ranges,
            "budget": {
                "max_iterations": int(budget.get("max_iterations", 0)),
                "time_limit_s": float(budget.get("time_limit_s", 0.0)),
            },
            "seeds": [int(s) for s in seeds],
            "typed_config_json": typed_config,
        }
        instance_id = _make_stable_instance_id_v2(payload)
        typed_config["instance_id"] = instance_id
        instances.append(
            {
                "instance_id": instance_id,
                "problem_type": problem_type,
                "opt_params": opt_params,
                "param_ranges": param_ranges,
                "objective_expression": objective_expression,
                "typed_config_json": typed_config,
            }
        )
    return instances


def _sns_flow_family_v3_generate_instances(cfg: dict, budget: Dict[str, Any], seeds: List[int]) -> List[Dict[str, Any]]:
    count = int(cfg.get("sns_family_count", cfg.get("auto_max_total_instances", 200)) or 0)
    base_seed = int(cfg.get("sns_family_seed", cfg.get("auto_seed", 0)) or 0)
    problem_type = str(cfg.get("sns_family_problem_type", "sns_flow_family_v3"))

    u_lower_lo, u_lower_hi = cfg.get("sns_family_u_lower_range", [3.0, 8.0])
    u_upper_lo, u_upper_hi = cfg.get("sns_family_u_upper_range", [12.0, 20.0])
    min_u_width = float(cfg.get("sns_family_min_u_width", 6.0))
    max_u_width = cfg.get("sns_family_max_u_width", None)
    max_u_width = float(max_u_width) if max_u_width is not None else None

    mu_lower_lo, mu_lower_hi = cfg.get("sns_family_mu_lower_range", [0.03, 0.10])
    mu_upper_lo, mu_upper_hi = cfg.get("sns_family_mu_upper_range", [0.12, 0.30])
    min_mu_width = float(cfg.get("sns_family_min_mu_width", 0.04))
    max_mu_width = cfg.get("sns_family_max_mu_width", None)
    max_mu_width = float(max_mu_width) if max_mu_width is not None else None

    r_lower_lo, r_lower_hi = cfg.get("sns_family_r_lower_range", [0.10, 0.16])
    r_upper_lo, r_upper_hi = cfg.get("sns_family_r_upper_range", [0.18, 0.30])
    min_r_width = float(cfg.get("sns_family_min_r_width", 0.04))
    max_r_width = cfg.get("sns_family_max_r_width", None)
    max_r_width = float(max_r_width) if max_r_width is not None else None

    bucket_ratios = cfg.get("sns_family_target_bucket_ratios") or {}
    r_mid = float(bucket_ratios.get("mid", 0.5))
    r_edge = float(bucket_ratios.get("edge", 0.3))
    r_out = float(bucket_ratios.get("outside", 0.2))
    s = max(1e-12, r_mid + r_edge + r_out)
    r_mid, r_edge, r_out = r_mid / s, r_edge / s, r_out / s
    mid_margin_frac = float(cfg.get("sns_family_mid_margin_frac", 0.2))
    edge_frac = float(cfg.get("sns_family_edge_frac", 0.15))
    outside_margin = float(cfg.get("sns_family_outside_margin", 120.0))
    approx_slope = float(cfg.get("sns_family_dp_slope", 107.5))
    approx_intercept = float(cfg.get("sns_family_dp_intercept", -177.5))

    task_catalog = cfg.get("sns_family_task_catalog") or [
        {"name": "calib_dp_1d", "objective_type": "target_matching", "task_type": "calibration", "metric": "pressuredrop", "opt_params": ["uMax"], "weight": 0.15},
        {"name": "calib_dp_2d", "objective_type": "target_matching", "task_type": "calibration", "metric": "pressuredrop", "opt_params": ["uMax", "Mu"], "weight": 0.12},
        {"name": "calib_avgp_2d", "objective_type": "target_matching", "task_type": "calibration", "metric": "avgpressure", "opt_params": ["uMax", "Mu"], "weight": 0.10},
        {"name": "min_dp_geom", "objective_type": "min", "task_type": "design_optimization", "metric": "pressuredrop", "opt_params": ["uMax", "Mu", "R"], "weight": 0.11},
        {"name": "min_avgpressure", "objective_type": "min", "task_type": "design_optimization", "metric": "avgpressure", "opt_params": ["uMax", "Mu", "R"], "weight": 0.10},
        {"name": "min_velmax", "objective_type": "min", "task_type": "design_optimization", "metric": "velmagmax", "opt_params": ["uMax", "Mu", "R"], "weight": 0.10},
        {
            "name": "min_pmax_with_dp_floor",
            "objective_type": "min",
            "task_type": "design_optimization",
            "metric": "pmax",
            "opt_params": ["uMax", "R"],
            "constraints": [{"metric": "pressuredrop", "type": ">=", "anchor": "mid", "ratio_range": [0.80, 1.00], "penalty_weight": 1500.0}],
            "weight": 0.12,
        },
        {
            "name": "max_dp_safe",
            "objective_type": "max",
            "task_type": "design_optimization",
            "metric": "pressuredrop",
            "opt_params": ["uMax", "Mu", "R"],
            "constraints": [
                {"metric": "pmax", "type": "<=", "anchor": "mid", "ratio_range": [1.02, 1.18], "penalty_weight": 1200.0},
                {"metric": "velmagmax", "type": "<=", "anchor": "mid", "ratio_range": [1.00, 1.12], "penalty_weight": 1200.0},
            ],
            "weight": 0.10,
        },
        {
            "name": "max_avgpressure_safe",
            "objective_type": "max",
            "task_type": "design_optimization",
            "metric": "avgpressure",
            "opt_params": ["uMax", "Mu", "R"],
            "constraints": [{"metric": "pmax", "type": "<=", "anchor": "mid", "ratio_range": [1.02, 1.15], "penalty_weight": 1200.0}],
            "weight": 0.10,
        },
    ]
    weights = [max(1e-9, float(t.get("weight", 1.0))) for t in task_catalog]

    def _sample_range(rng: random.Random, lo_lo: float, lo_hi: float, hi_lo: float, hi_hi: float, min_width: float, max_width: Any = None) -> Tuple[float, float]:
        lo = float(rng.uniform(float(lo_lo), float(lo_hi)))
        hi = float(rng.uniform(float(hi_lo), float(hi_hi)))
        if hi - lo < min_width:
            hi = lo + float(min_width)
        if max_width is not None and (hi - lo) > float(max_width):
            hi = lo + float(max_width)
        hi = max(lo + 1e-9, hi)
        return float(lo), float(hi)

    rng = random.Random(_stable_int(f"sns_flow_family_v3:{base_seed}"))
    instances: List[Dict[str, Any]] = []
    for _ in range(max(0, count)):
        task = rng.choices(task_catalog, weights=weights, k=1)[0]
        opt_params = list(task.get("opt_params") or ["uMax"])

        u_lo, u_hi = _sample_range(rng, float(u_lower_lo), float(u_lower_hi), float(u_upper_lo), float(u_upper_hi), min_u_width, max_u_width)
        mu_lo, mu_hi = _sample_range(rng, float(mu_lower_lo), float(mu_lower_hi), float(mu_upper_lo), float(mu_upper_hi), min_mu_width, max_mu_width)
        r_lo, r_hi = _sample_range(rng, float(r_lower_lo), float(r_lower_hi), float(r_upper_lo), float(r_upper_hi), min_r_width, max_r_width)

        param_ranges: Dict[str, Tuple[float, float]] = {}
        if "uMax" in opt_params:
            param_ranges["uMax"] = (u_lo, u_hi)
        if "Mu" in opt_params:
            param_ranges["Mu"] = (mu_lo, mu_hi)
        if "R" in opt_params:
            param_ranges["R"] = (r_lo, r_hi)

        metric = str(task.get("metric", "pressuredrop"))
        objective_type = str(task.get("objective_type", "min"))
        task_type = str(task.get("task_type", "design_optimization"))
        est_ranges = _sns_metric_estimate_ranges(u_lo, u_hi, mu_lo, mu_hi, r_lo, r_hi, approx_slope, approx_intercept)
        targets: Dict[str, float] = {}
        bucket = str(task.get("name", "mixed"))

        if objective_type == "target_matching":
            metric_lo_est, metric_hi_est = est_ranges.get(metric, est_ranges["pressuredrop"])
            target_value, bucket = _sns_sample_target_value(
                rng,
                float(metric_lo_est),
                float(metric_hi_est),
                float(r_mid),
                float(r_edge),
                float(r_out),
                float(mid_margin_frac),
                float(edge_frac),
                float(outside_margin),
            )
            targets = {f"{metric}_target": float(target_value)}
            objective_expression = _sns_objective_expr(metric, objective_type, target_value)
        else:
            objective_expression = _sns_objective_expr(metric, objective_type)

        constraints = _sns_build_constraints(rng, est_ranges, task.get("constraints") or [])
        c_names, c_types, c_thresholds = _sns_constraint_fields(constraints)
        metric_estimates = {k: {"lo": float(v[0]), "hi": float(v[1])} for k, v in est_ranges.items()}

        var_lows = [float(param_ranges[p][0]) for p in opt_params]
        var_highs = [float(param_ranges[p][1]) for p in opt_params]
        typed_config = {
            "instance_id": None,
            "physics_domain": "fluid",
            "task_type": task_type,
            "objective_type": objective_type,
            "decision_dim": len(opt_params),
            "variable_name_json": opt_params,
            "var_lower_bound_json": var_lows,
            "var_upper_bound_json": var_highs,
            "num_constraints": len(constraints),
            "constraint_name_json": c_names,
            "constraint_type_json": c_types,
            "constraint_threshold_json": c_thresholds,
            "constraints": constraints,
            "budget_eval": int(budget.get("max_iterations", 0)),
            "budget_walltime_sec": float(budget.get("time_limit_s", 0.0)),
            "template_name": "sns",
            "targets": targets,
            "generation_bucket": bucket,
            "family_version": "sns_v3",
            "task_name": str(task.get("name", "mixed")),
            "primary_metric": metric,
            "metric_estimates": metric_estimates,
        }

        payload = {
            "problem_type": problem_type,
            "opt_params": opt_params,
            "objective_expression": objective_expression,
            "param_ranges": param_ranges,
            "budget": {
                "max_iterations": int(budget.get("max_iterations", 0)),
                "time_limit_s": float(budget.get("time_limit_s", 0.0)),
            },
            "seeds": [int(s) for s in seeds],
            "typed_config_json": typed_config,
        }
        instance_id = _make_stable_instance_id_v2(payload)
        typed_config["instance_id"] = instance_id
        instances.append(
            {
                "instance_id": instance_id,
                "problem_type": problem_type,
                "opt_params": opt_params,
                "param_ranges": param_ranges,
                "objective_expression": objective_expression,
                "typed_config_json": typed_config,
            }
        )
    return instances


def _load_instances_from_dataset(dataset_path: str, sample_clean: int, sample_hard: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(_stable_int(f"dataset_sample:{dataset_path}:{seed}"))
    clean: List[Dict[str, Any]] = []
    hard: List[Dict[str, Any]] = []
    other: List[Dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            status = str(row.get("status") or "").strip().lower()
            inst = {
                "instance_id": row.get("instance_id"),
                "problem_type": row.get("problem_type"),
                "opt_params": row.get("opt_params") or [],
                "param_ranges": row.get("param_ranges") or {},
                "objective_expression": row.get("objective_expression") or "",
                "typed_config_json": row.get("typed_config_json"),
            }
            if not inst.get("instance_id") or not inst.get("problem_type") or not inst.get("opt_params") or not inst.get("objective_expression"):
                continue
            if status == "valid_clean":
                clean.append(inst)
            elif status == "valid_hard":
                hard.append(inst)
            else:
                other.append(inst)

    rng.shuffle(clean)
    rng.shuffle(hard)
    rng.shuffle(other)

    if int(sample_clean) <= 0 and int(sample_hard) <= 0:
        out: List[Dict[str, Any]] = []
        out.extend(clean)
        out.extend(hard)
        out.extend(other)
        return out
    out = []
    out.extend(clean[: max(0, int(sample_clean))])
    out.extend(hard[: max(0, int(sample_hard))])
    if len(out) < (max(0, int(sample_clean)) + max(0, int(sample_hard))):
        out.extend(other[: max(0, (max(0, int(sample_clean)) + max(0, int(sample_hard))) - len(out))])
    return out


def _build_all_candidates() -> List[Dict[str, str]]:
    import sys
    sys.path.append(_repo_root())
    from src.problem_config import get_optimizer_config

    cfg = get_optimizer_config()
    out: List[Dict[str, str]] = []
    for optimizer, meta in cfg.items():
        for method in meta.get("methods", []) or []:
            out.append({"optimizer": optimizer, "method": method})
    return out


def _auto_build_instances(problem_types: Dict[str, dict], cfg: dict) -> List[Dict[str, Any]]:
    include = cfg.get("auto_include_problem_types")
    
    auto_mode = str(cfg.get("auto_mode") or "").strip().lower()
    if auto_mode == "manual":
        problem_types_to_run = cfg.get("auto_problem_types", [])
        if problem_types_to_run:
            include = problem_types_to_run
            
    exclude = set(cfg.get("auto_exclude_problem_types") or [])
    per_problem = int(cfg.get("auto_instances_per_problem", 6))
    sizes = cfg.get("auto_opt_param_sizes") or [1, 2, 3]
    sizes = [int(s) for s in sizes if int(s) > 0]
    targets = cfg.get("auto_targets") or {"keff": 1.0}
    base_seed = int(cfg.get("auto_seed", 0))
    include_corca = bool(cfg.get("auto_include_corca", False))
    shard_count = int(cfg.get("shard_count", 1) or 1)
    shard_index = int(cfg.get("shard_index", 0) or 0)

    instances: List[Dict[str, Any]] = []
    seen = set()

    for problem_type in sorted(problem_types.keys()):
        if include and problem_type not in include:
            continue
        if problem_type in exclude:
            continue
        pcfg = problem_types[problem_type] or {}
        if pcfg.get("template_name") == "colorbar" or problem_type == "colorbar":
            continue
        sim_type = pcfg.get("simulator_type")
        if sim_type in {"corca_state", "corca_evol", "corca_xenon"} and not include_corca:
            continue
        if sim_type not in {"freefem", "corca_state", "python", "corca_evol", "corca_xenon"}:
            continue

        params = list((pcfg.get("parameters") or {}).keys())
        if problem_type == "corca_state":
            params = ["Pp:", "Prk:", "Tin:"]
        if not params:
            continue
        objectives = list(pcfg.get("objectives") or [])
        if not objectives:
            objectives = ["cpu"]

        local_rng = random.Random(_stable_int(f"{problem_type}:{base_seed}"))
        local_rng.shuffle(params)
        local_rng.shuffle(objectives)

        for i in range(per_problem):
            sz = sizes[i % len(sizes)]
            sz = min(sz, len(params))
            if sz <= 0:
                continue
            start = (i * sz) % len(params)
            opt_params = params[start : start + sz]
            if len(opt_params) < sz:
                opt_params = local_rng.sample(params, sz)
            metric = objectives[i % len(objectives)]
            expr = _objective_expression_for_metric(metric, targets)
            key = (problem_type, tuple(opt_params), expr)
            if key in seen:
                continue
            seen.add(key)
            if not _shard_keep(problem_type, opt_params, expr, shard_index, shard_count):
                continue
            instances.append(
                {
                    "problem_type": problem_type,
                    "opt_params": opt_params,
                    "objective_expression": expr,
                }
            )

    max_total = cfg.get("auto_max_total_instances")
    if max_total is not None:
        instances = instances[: int(max_total)]
    return instances


def _run_one(
    simulator,
    algo_lib,
    problem_type: str,
    template_name: str,
    param_ranges: Dict[str, Tuple[float, float]],
    opt_params: List[str],
    optimizer: str,
    method: str,
    objective_expression: str,
    seed: int,
    constraints: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    iteration_history: List[Dict[str, Any]] = []
    walltime_acc = 0.0
    analysis_config = {
        "problem_type": problem_type,
        "objective_type": "expression",
        "primary_objective": "expression",
        "objective_expression": objective_expression,
        "opt_params": opt_params,
        "target_values": {},
        "constraints": list(constraints or []),
        "user_input": f"[training] {problem_type} -> {optimizer}-{method} seed={seed}",
    }

    def objective_function(params_list: List[float]) -> float:
        nonlocal walltime_acc
        params_dict = {name: value for name, value in zip(opt_params, params_list)}
        
        t0 = time.time()
        if problem_type == "corca_state":
            bore_l, bore_r = 0, 2000
            bore_steps = 10
            found = False
            last_result = {"success": False}
            for _ in range(bore_steps):
                if bore_l > bore_r:
                    break
                bore_mid = (bore_l + bore_r) // 2
                params_dict["bore_ppm"] = bore_mid
                params_dict["Prk:"] = int(params_dict.get("Prk:", 50))
                result = simulator.run_simulation(template_name, params_dict)
                last_result = result
                if not isinstance(result, dict) or not result.get("success"):
                    break
                k = float(result.get("keff", 1.0))
                if k < 1.033:
                    bore_r = bore_mid - 1
                elif k > 1.035:
                    bore_l = bore_mid + 1
                else:
                    found = True
                    break
            result = last_result
            
            k = float(result.get("keff", 1.0)) if result.get("success") else None
            if k is None:
                loss_value = 1e9
                total_violation = 1.0
                feasible = False
            else:
                fq = float(result.get("FQ", 1e9))
                fq_gap = abs(fq - 3.5)
                if found and fq <= 3.5:
                    loss_value = fq
                    total_violation = 0.0
                    feasible = True
                else:
                    keff_penalty = 1000.0 * min(abs(k - 1.033), abs(k - 1.035), abs(k - 1.034))
                    fq_penalty = max(0.0, fq - 3.5)
                    loss_value = 1e6 + keff_penalty + 100.0 * fq_penalty + fq_gap
                    total_violation = fq_penalty + (keff_penalty / 1000.0)
                    feasible = False
            success = True
            solver_status = "ok"
            eval_time = time.time() - t0
            try:
                walltime_acc += float(eval_time)
            except Exception:
                pass
        else:
            result = simulator.run_simulation(template_name, params_dict)
            eval_time = time.time() - t0
            try:
                walltime_acc += float(eval_time)
            except Exception:
                pass
            obj = algo_lib.calculate_objective(result, analysis_config, problem_type)
            try:
                loss_value = float(obj)
            except Exception:
                loss_value = float("inf")
            success = bool(result.get("success", False))
            c_report = _instance_constraint_report(result, constraints or [])
            total_violation = float(c_report.get("total_violation", 0.0) or 0.0)
            feasible = bool(success and (loss_value == loss_value) and (loss_value != float("inf")) and total_violation <= 1e-12)
            solver_status = "ok" if success else "crash"
            if not (loss_value == loss_value):
                solver_status = "nan"
        iteration_history.append(
            {
                "params": params_dict,
                "x_list": [float(v) for v in params_list],
                "objective_raw": float(loss_value) if loss_value != float("inf") else None,
                "loss_value": float(loss_value) if loss_value != float("inf") else None,
                "score_value": float(loss_value) if loss_value != float("inf") else None,
                "constraint_violation_total": float(total_violation),
                "feasible_flag": bool(feasible),
                "eval_time_sec": float(eval_time),
                "walltime_from_start_sec": float(walltime_acc),
                "solver_status": solver_status,
                "result": result,
            }
        )
        return float(loss_value)

    _set_seed(seed)
    if hasattr(algo_lib, "random_seed"):
        try:
            algo_lib.random_seed = int(seed)
        except Exception:
            pass

    start = time.time()
    if optimizer == "SciPy":
        algo_res = algo_lib.run_scipy(objective_function, param_ranges, method)
    elif optimizer == "Optuna":
        algo_res = algo_lib.run_optuna(objective_function, param_ranges, method)
    elif optimizer == "Hyperopt":
        algo_res = algo_lib.run_hyperopt(objective_function, param_ranges, method)
    elif optimizer == "Nevergrad":
        algo_res = algo_lib.run_nevergrad(objective_function, param_ranges, method)
    else:
        algo_res = {"success": False, "message": f"未知优化器: {optimizer}"}

    exec_time = time.time() - start
    result_data = {
        "optimizer": optimizer,
        "method": method,
        "seed": int(seed),
        "execution_time": float(exec_time),
        "iteration_history": iteration_history,
        "analysis_config": analysis_config,
        **algo_res,
    }
    return result_data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(_repo_root(), "training", "instances_small.json"))
    parser.add_argument("--out_dir", default=os.path.join(_repo_root(), "results", "training"))
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    budget = cfg.get("budget", {}) or {}
    max_iterations = int(budget.get("max_iterations", 15))
    time_limit_s = float(budget.get("time_limit_s", 60))
    resume = bool(cfg.get("resume", False))
    stable_ids = bool(cfg.get("stable_instance_ids", False))
    shard_count = int(cfg.get("shard_count", 1) or 1)
    shard_index = int(cfg.get("shard_index", 0) or 0)

    seeds = cfg.get("seeds")
    if seeds is None:
        repeats = int(cfg.get("repeats", 1))
        seeds = list(range(repeats))
    seeds = [int(s) for s in (seeds or [0])]

    candidates = cfg.get("candidates", []) or []
    if cfg.get("auto") and not candidates:
        candidates = _build_all_candidates()
    candidates = [c for c in candidates if _algo_available(c.get("optimizer"))]
    cand_shard_count = int(cfg.get("candidate_shard_count", 1) or 1)
    cand_shard_index = int(cfg.get("candidate_shard_index", 0) or 0)
    if cand_shard_count > 1:
        candidates = [
            c
            for c in candidates
            if _candidate_shard_keep(
                str(c.get("optimizer") or ""),
                str(c.get("method") or ""),
                cand_shard_index,
                cand_shard_count,
            )
        ]
    generate_only = bool(cfg.get("generate_only", False))
    if not generate_only and not candidates:
        raise RuntimeError("没有可用候选算法（SciPy/Optuna/Hyperopt/Nevergrad 均不可用）")

    instances = cfg.get("instances", []) or []
    if not instances and cfg.get("instances_from_dataset_path"):
        ds_path = str(cfg.get("instances_from_dataset_path")).strip()
        sample_seed = int(cfg.get("instances_sample_seed", 0))
        sample_clean = int(cfg.get("instances_sample_clean", 0))
        sample_hard = int(cfg.get("instances_sample_hard", 0))
        instances = _load_instances_from_dataset(ds_path, sample_clean=sample_clean, sample_hard=sample_hard, seed=sample_seed)
    if cfg.get("auto") and not instances:
        auto_mode = str(cfg.get("auto_mode") or "").strip().lower()
        if auto_mode in {"sns_flow_family", "sns_family"}:
            instances = _sns_flow_family_generate_instances(cfg, budget, seeds)
        elif auto_mode in {"sns_flow_family_v2", "sns_family_v2", "sns_diverse_v2"}:
            instances = _sns_flow_family_v2_generate_instances(cfg, budget, seeds)
        elif auto_mode in {"sns_flow_family_v3", "sns_family_v3", "sns_diverse_v3"}:
            instances = _sns_flow_family_v3_generate_instances(cfg, budget, seeds)
        elif auto_mode in {"thermal_fins_family_v1", "heat_family_v1", "thmf_family_v1"}:
            instances = _heat_family_v1_generate_instances(cfg, budget, seeds)
        elif auto_mode in {"neutron_diffusion_family_v1", "iaea_family_v1"}:
            instances = _iaea_family_v1_generate_instances(cfg, budget, seeds)
        elif auto_mode in {"ex_heat_time_family_v1", "heat_time_family_v1"}:
            instances = _heat_time_family_v1_generate_instances(cfg, budget, seeds)
        elif auto_mode in {"ex_blackscholes2d_family_v1", "blackscholes_family_v1"}:
            instances = _blackscholes2d_family_v1_generate_instances(cfg, budget, seeds)
        elif auto_mode in {"ex_advection2d_family_v1", "advection_family_v1"}:
            instances = _advection2d_family_v1_generate_instances(cfg, budget, seeds)
        else:
            import sys
            sys.path.append(_repo_root())
            from src.problem_config import ProblemConfig
            pc_auto = ProblemConfig()
            instances = _auto_build_instances(pc_auto.problem_types, cfg)
    if not instances:
        raise RuntimeError("配置中未提供 instances")

    os.makedirs(args.out_dir, exist_ok=True)
    try:
        with open(os.path.join(args.out_dir, "config_used.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    dataset_path = os.path.join(args.out_dir, "dataset.jsonl")
    runs_path = os.path.join(args.out_dir, "runs.jsonl")
    invalid_path = os.path.join(args.out_dir, "invalid_instances.jsonl")
    stats_path = os.path.join(args.out_dir, "stats.json")
    problem_instance_path = os.path.join(args.out_dir, "problem_instance.jsonl")
    optimizer_run_path = os.path.join(args.out_dir, "optimizer_run.jsonl")
    iteration_event_path = os.path.join(args.out_dir, "iteration_event.jsonl")
    run_checkpoint_path = os.path.join(args.out_dir, "run_checkpoint.jsonl")
    run_summary_path = os.path.join(args.out_dir, "run_summary.jsonl")

    dataset_done = set()
    existing_runs = {}
    if resume:
        if os.path.exists(dataset_path):
            with open(dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        iid = row.get("instance_id")
                        if iid:
                            dataset_done.add(str(iid))
                    except Exception:
                        continue

        if os.path.exists(runs_path):
            with open(runs_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        key = (
                            str(row.get("instance_id")),
                            str(row.get("optimizer")),
                            str(row.get("method")),
                            int(row.get("seed")),
                        )
                        existing_runs[key] = row
                    except Exception:
                        continue

    import sys
    sys.path.append(_repo_root())
    from src.problem_config import ProblemConfig
    from src.algorithm_library import AlgorithmLibrary
    from src.simulator_interface import SimulatorFactory
    from src.optimization_history import OptimizationHistory
    from src.freefem_simulator import get_template_pde_type

    pc = ProblemConfig()
    algo_lib = AlgorithmLibrary()
    algo_lib.max_iterations = max_iterations
    algo_lib.time_limit = time_limit_s

    simulator_cache: Dict[str, Any] = {}
    history = OptimizationHistory(
        db_path=os.path.join(args.out_dir, "optimization_history.db"),
        runs_dir=os.path.join(args.out_dir, "runs"),
    )

    ds_mode = "a" if resume else "w"
    run_mode = "a" if resume else "w"
    inv_mode = "a" if resume else "w"
    table_mode = "a" if resume else "w"
    validate_instances = bool(cfg.get("validate_instances", False))
    probe_endpoints = bool(cfg.get("sns_family_probe_endpoints", False))
    with (
        open(dataset_path, ds_mode, encoding="utf-8") as ds_f,
        open(runs_path, run_mode, encoding="utf-8") as run_f,
        open(invalid_path, inv_mode, encoding="utf-8") as inv_f,
        open(problem_instance_path, table_mode, encoding="utf-8") as pi_f,
        open(optimizer_run_path, table_mode, encoding="utf-8") as or_f,
        open(iteration_event_path, table_mode, encoding="utf-8") as ie_f,
        open(run_checkpoint_path, table_mode, encoding="utf-8") as rc_f,
        open(run_summary_path, table_mode, encoding="utf-8") as rs_f,
    ):
        stats = {"total": 0, "invalid_generation": 0, "valid_clean": 0, "valid_hard": 0, "other": 0}
        for inst in instances:
            problem_type = str(inst.get("problem_type", "")).strip()
            opt_params = inst.get("opt_params", []) or []
            objective_expression = str(inst.get("objective_expression", "")).strip()
            if not problem_type or not opt_params or not objective_expression:
                continue
            if not _shard_keep(problem_type, opt_params, objective_expression, shard_index, shard_count):
                continue

            problem_config = pc.get_problem_config(problem_type)
            if not problem_config:
                continue
            template_name = problem_config.get("template_name")
            if not template_name:
                continue

            param_ranges = inst.get("param_ranges")
            if param_ranges:
                pr_ok = True
                for p in opt_params:
                    if p not in param_ranges:
                        pr_ok = False
                        break
                if not pr_ok:
                    param_ranges = None
            if not param_ranges:
                param_ranges = _select_param_ranges(problem_config, opt_params)

            if stable_ids:
                if inst.get("instance_id"):
                    instance_id = str(inst.get("instance_id"))
                elif inst.get("param_ranges") or inst.get("typed_config_json"):
                    payload = {
                        "problem_type": problem_type,
                        "opt_params": list(opt_params),
                        "objective_expression": str(objective_expression),
                        "param_ranges": param_ranges,
                        "budget": {
                            "max_iterations": int(budget.get("max_iterations", 0)),
                            "time_limit_s": float(budget.get("time_limit_s", 0.0)),
                        },
                        "seeds": [int(s) for s in seeds],
                        "typed_config_json": inst.get("typed_config_json"),
                    }
                    instance_id = _make_stable_instance_id_v2(payload)
                else:
                    instance_id = _make_stable_instance_id(problem_type, opt_params, objective_expression, budget, seeds)
            else:
                instance_id = str(inst.get("instance_id") or _make_instance_id(problem_type))

            if resume and instance_id in dataset_done:
                continue
            sim_type = problem_config.get("simulator_type", "freefem")
            if sim_type not in simulator_cache:
                try:
                    simulator_cache[sim_type] = SimulatorFactory.create_simulator(sim_type)
                except Exception:
                    continue
            simulator = simulator_cache[sim_type]

            inst_status = None
            smoke = None
            if validate_instances and template_name in {"sns", "thmf", "iaea"}:
                smoke = _template_smoke_test(simulator, template_name, param_ranges, probe_endpoints=probe_endpoints)
                if not smoke.get("success", False):
                    inst_status = "invalid_generation"
                else:
                    if template_name == "sns":
                        min_width = float(cfg.get("sns_family_min_u_width", 6.0))
                    elif template_name == "thmf":
                        min_width = float(cfg.get("heat_family_min_width", 0.05))
                    else:
                        min_width = float(cfg.get("iaea_family_min_width", 1e-4))
                    widths = [float(v[1]) - float(v[0]) for v in param_ranges.values()]
                    min_inst_width = min(widths) if widths else min_width

                    tcfg = inst.get("typed_config_json")
                    target_metric, target_value = _extract_target_metric_value(tcfg if isinstance(tcfg, dict) else {})
                    smoke["target_metric"] = target_metric
                    smoke["target_value"] = float(target_value) if target_value is not None else None

                    decision_dim = int((tcfg or {}).get("decision_dim") or len(opt_params))
                    num_constraints = int((tcfg or {}).get("num_constraints") or 0)
                    if target_value is None:
                        hard_flag = bool(num_constraints > 0 or min_inst_width < min_width)
                        if template_name == "sns":
                            hard_flag = hard_flag or ("R" in param_ranges) or (decision_dim >= 3)
                        elif template_name == "thmf":
                            hard_flag = hard_flag or (decision_dim >= 4)
                        elif template_name == "iaea":
                            hard_flag = hard_flag or (decision_dim >= 5)
                        inst_status = "valid_hard" if hard_flag else "valid_clean"
                    else:
                        metric_name = str(target_metric or "unknown")
                        metric_lo = smoke.get(f"{metric_name}_lo")
                        metric_hi = smoke.get(f"{metric_name}_hi")
                        if metric_lo is not None and metric_hi is not None and float(metric_hi) < float(metric_lo):
                            metric_lo, metric_hi = metric_hi, metric_lo
                        in_range = False
                        if metric_lo is not None and metric_hi is not None:
                            in_range = (float(target_value) >= float(metric_lo)) and (float(target_value) <= float(metric_hi))
                        mid_val = smoke.get(f"{metric_name}_mid")
                        gap_mid = None
                        if mid_val is not None:
                            try:
                                gap_mid = abs(float(mid_val) - float(target_value))
                            except Exception:
                                gap_mid = None
                        smoke["gap_mid"] = gap_mid
                        inst_status = "valid_clean" if (in_range and min_inst_width >= min_width and num_constraints <= 0) else "valid_hard"

            stats["total"] += 1
            if inst_status in stats:
                stats[inst_status] += 1
            else:
                stats["other"] += 1

            typed_cfg = inst.get("typed_config_json") if isinstance(inst.get("typed_config_json"), dict) else {}
            if typed_cfg.get("instance_id") != instance_id:
                typed_cfg = {**typed_cfg, "instance_id": instance_id}
            if not typed_cfg.get("pde_type"):
                typed_cfg = {**typed_cfg, "pde_type": get_template_pde_type(str(template_name))}
            physics_domain = typed_cfg.get("physics_domain") or ("fluid" if template_name == "sns" else ("heat" if template_name == "thmf" else ("neutronics" if template_name == "iaea" else "unknown")))
            task_type = typed_cfg.get("task_type") or "calibration"
            objective_type = typed_cfg.get("objective_type") or "target_matching"
            lows = [float(param_ranges[p][0]) for p in opt_params]
            highs = [float(param_ranges[p][1]) for p in opt_params]
            vr = _var_range_stats(lows, highs)
            num_constraints = int(typed_cfg.get("num_constraints") or 0)
            decision_dim = int(typed_cfg.get("decision_dim") or len(opt_params))
            pi_row = {
                "instance_id": instance_id,
                "physics_domain": physics_domain,
                "pde_type": typed_cfg.get("pde_type"),
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
                "budget_eval": int(typed_cfg.get("budget_eval") or max_iterations),
                "budget_walltime_sec": float(typed_cfg.get("budget_walltime_sec") or time_limit_s),
                "typed_config_json": typed_cfg,
                **vr,
                "constraint_density": (float(num_constraints) / float(decision_dim)) if decision_dim > 0 else 0.0,
                "budget_per_dim": (float(int(typed_cfg.get("budget_eval") or max_iterations)) / float(decision_dim)) if decision_dim > 0 else 0.0,
                "problem_type": problem_type,
                "template_name": template_name,
                "status": inst_status,
                "smoke": smoke,
            }
            _write_jsonl(pi_f, pi_row)

            if inst_status == "invalid_generation":
                _write_jsonl(
                    inv_f,
                    {
                        "instance_id": instance_id,
                        "problem_type": problem_type,
                        "template_name": template_name,
                        "opt_params": opt_params,
                        "param_ranges": param_ranges,
                        "objective_expression": objective_expression,
                        "typed_config_json": typed_cfg,
                        "smoke": smoke,
                    },
                )
                _write_jsonl(
                    ds_f,
                    {
                        "instance_id": instance_id,
                        "problem_type": problem_type,
                        "template_name": template_name,
                        "opt_params": opt_params,
                        "param_ranges": param_ranges,
                        "objective_expression": objective_expression,
                        "budget": {"max_iterations": max_iterations, "time_limit_s": time_limit_s, "seeds": seeds},
                        "candidates": [],
                        "label": None,
                        "status": inst_status,
                        "smoke": smoke,
                        "typed_config_json": typed_cfg,
                    },
                )
                if resume:
                    dataset_done.add(instance_id)
                continue

            if generate_only:
                _write_jsonl(
                    ds_f,
                    {
                        "instance_id": instance_id,
                        "problem_type": problem_type,
                        "template_name": template_name,
                        "opt_params": opt_params,
                        "param_ranges": param_ranges,
                        "objective_expression": objective_expression,
                        "budget": {"max_iterations": max_iterations, "time_limit_s": time_limit_s, "seeds": seeds},
                        "candidates": [],
                        "label": None,
                        "status": inst_status,
                        "smoke": smoke,
                        "typed_config_json": typed_cfg,
                    },
                )
                if resume:
                    dataset_done.add(instance_id)
                continue

            candidate_results = []
            run_summaries_for_instance: List[Dict[str, Any]] = []
            for cand in candidates:
                optimizer = cand.get("optimizer")
                method = cand.get("method")
                if not optimizer or not method:
                    continue
                seed_runs = []
                for seed in seeds:
                    existing_key = (instance_id, str(optimizer), str(method), int(seed))
                    if resume and existing_key in existing_runs:
                        seed_runs.append(existing_runs[existing_key])
                        continue
                    typed_cfg_run = inst.get("typed_config_json") if isinstance(inst.get("typed_config_json"), dict) else {}
                    res = _run_one(
                        simulator=simulator,
                        algo_lib=algo_lib,
                        problem_type=problem_type,
                        template_name=template_name,
                        param_ranges=param_ranges,
                        opt_params=opt_params,
                        optimizer=optimizer,
                        method=method,
                        objective_expression=objective_expression,
                        seed=seed,
                        constraints=typed_cfg_run.get("constraints") or [],
                    )
                    run_id = history.add_optimization_record(res)
                    backend_lib = str(optimizer)
                    optimizer_name = str(method)
                    algo_cfg = {"method": method}
                    assigned_eval = int(max_iterations)
                    assigned_time = float(time_limit_s)
                    run_status = "success" if bool(res.get("success", False)) else "fail"
                    termination_reason = "budget" if run_status == "success" else "crash"
                    _write_jsonl(
                        or_f,
                        {
                            "run_id": run_id,
                            "instance_id": instance_id,
                            "seed": int(seed),
                            "optimizer_name": optimizer_name,
                            "backend_lib": backend_lib,
                            "algorithm_config_json": algo_cfg,
                            "assigned_budget_eval": assigned_eval,
                            "assigned_budget_walltime": assigned_time,
                            "run_status": run_status,
                            "termination_reason": termination_reason,
                        },
                    )

                    it_hist = res.get("iteration_history", []) or []
                    best_loss = float("inf")
                    best_feas = float("inf")
                    best_idx = None
                    best_feas_idx = None
                    for i, ev in enumerate(it_hist, start=1):
                        loss = ev.get("loss_value")
                        try:
                            loss_f = float(loss)
                        except Exception:
                            loss_f = float("inf")
                        feas = bool(ev.get("feasible_flag", False))
                        if loss_f < best_loss:
                            best_loss = loss_f
                            best_idx = i
                        if feas and loss_f < best_feas:
                            best_feas = loss_f
                            best_feas_idx = i
                        _write_jsonl(
                            ie_f,
                            {
                                "eval_id": f"{run_id}_{i}",
                                "run_id": run_id,
                                "eval_index": int(i),
                                "x_json": ev.get("x_list"),
                                "objective_raw": ev.get("objective_raw"),
                                "loss_value": ev.get("loss_value"),
                                "score_value": ev.get("score_value"),
                                "constraint_violation_total": ev.get("constraint_violation_total", 0.0),
                                "feasible_flag": bool(feas),
                                "eval_time_sec": ev.get("eval_time_sec"),
                                "walltime_from_start_sec": ev.get("walltime_from_start_sec"),
                                "solver_status": ev.get("solver_status"),
                                "is_new_best": bool(best_idx == i),
                                "is_new_best_feasible": bool(best_feas_idx == i),
                            },
                        )

                    for cp in _run_checkpoints(run_id, instance_id, it_hist):
                        _write_jsonl(rc_f, cp)

                    n_evals = int(len(it_hist))
                    final_feasible = any(bool(ev.get("feasible_flag", False)) for ev in it_hist)
                    evals_to_first_feasible = None
                    walltime_to_first_feasible = None
                    if final_feasible:
                        for ev_i, ev in enumerate(it_hist, start=1):
                            if bool(ev.get("feasible_flag", False)):
                                evals_to_first_feasible = ev_i
                                walltime_to_first_feasible = ev.get("walltime_from_start_sec")
                                break
                    walltime_to_best = None
                    if best_idx is not None and 1 <= int(best_idx) <= n_evals:
                        walltime_to_best = it_hist[int(best_idx) - 1].get("walltime_from_start_sec")
                    anytime_auc = None
                    cps = _run_checkpoints(run_id, instance_id, it_hist)
                    if cps:
                        anytime_auc = cps[-1].get("auc_regret_prefix")

                    failure_count = sum(1 for ev in it_hist if not bool(ev.get("feasible_flag", False)))
                    nan_count = sum(1 for ev in it_hist if str(ev.get("solver_status")) == "nan")
                    timeout_count = sum(1 for ev in it_hist if str(ev.get("solver_status")) == "timeout")
                    min_violation = min(float(ev.get("constraint_violation_total", 0.0) or 0.0) for ev in it_hist) if it_hist else 0.0
                    final_violation = float(it_hist[-1].get("constraint_violation_total", 0.0) or 0.0) if it_hist else 0.0
                    rs_row = {
                        "run_id": run_id,
                        "instance_id": instance_id,
                        "optimizer_name": optimizer_name,
                        "best_objective": float(best_loss) if best_loss != float("inf") else None,
                        "best_feasible_objective": float(best_feas) if best_feas != float("inf") else None,
                        "best_loss": float(best_loss) if best_loss != float("inf") else None,
                        "final_feasible_flag": bool(final_feasible),
                        "evals_to_best": int(best_idx) if best_idx is not None else None,
                        "walltime_to_best": walltime_to_best,
                        "evals_to_first_feasible": int(evals_to_first_feasible) if evals_to_first_feasible is not None else None,
                        "walltime_to_first_feasible": walltime_to_first_feasible,
                        "num_evals": int(n_evals),
                        "anytime_auc": anytime_auc,
                        "final_constraint_violation": final_violation,
                        "min_constraint_violation": min_violation,
                        "rank_within_instance": None,
                        "failure_count": int(failure_count),
                        "timeout_count": int(timeout_count),
                        "nan_count": int(nan_count),
                        "backend_lib": backend_lib,
                        "method": method,
                        "seed": int(seed),
                    }
                    run_summaries_for_instance.append(rs_row)
                    res_record = {
                        "instance_id": instance_id,
                        "run_id": run_id,
                        "seed": int(seed),
                        "problem_type": problem_type,
                        "optimizer": optimizer,
                        "method": method,
                        "success": bool(res.get("success", False)),
                        "best_value": res.get("best_value"),
                        "iterations": res.get("iterations", 0),
                        "execution_time": res.get("execution_time", 0.0),
                    }
                    seed_runs.append(res_record)
                    _write_jsonl(run_f, res_record)
                    if resume:
                        existing_runs[existing_key] = res_record

                successes = [r for r in seed_runs if r.get("success")]
                if successes:
                    try:
                        mean_best = sum(float(r.get("best_value")) for r in successes) / float(len(successes))
                    except Exception:
                        mean_best = float("inf")
                else:
                    mean_best = float("inf")
                try:
                    mean_time = sum(float(r.get("execution_time", 0.0)) for r in seed_runs) / float(len(seed_runs))
                except Exception:
                    mean_time = 0.0
                cand_summary = {
                    "instance_id": instance_id,
                    "problem_type": problem_type,
                    "optimizer": optimizer,
                    "method": method,
                    "seeds": seeds,
                    "runs": seed_runs,
                    "success_rate": (len(successes) / float(len(seed_runs))) if seed_runs else 0.0,
                    "mean_best_value": mean_best,
                    "mean_execution_time": mean_time,
                }
                candidate_results.append(cand_summary)

            rank_base = []
            for r in run_summaries_for_instance:
                if r.get("best_feasible_objective") is not None:
                    rank_base.append((float(r["best_feasible_objective"]), r))
            rank_base.sort(key=lambda x: x[0])
            rank = 1
            for _, r in rank_base:
                r["rank_within_instance"] = int(rank)
                rank += 1
            for r in run_summaries_for_instance:
                if r.get("rank_within_instance") is None:
                    r["rank_within_instance"] = int(rank)
                    rank += 1
                _write_jsonl(rs_f, r)

            def key_fn(x):
                try:
                    return float(x.get("mean_best_value", float("inf")))
                except Exception:
                    return float("inf")

            best = min(candidate_results, key=key_fn) if candidate_results else None
            label = f"{best['optimizer']}-{best['method']}" if best and key_fn(best) != float("inf") else None

            ds_row = {
                "instance_id": instance_id,
                "problem_type": problem_type,
                "template_name": template_name,
                "opt_params": opt_params,
                "param_ranges": param_ranges,
                "objective_expression": objective_expression,
                "budget": {"max_iterations": max_iterations, "time_limit_s": time_limit_s, "seeds": seeds},
                "candidates": candidate_results,
                "label": label,
                "status": inst_status,
                "smoke": smoke,
                "typed_config_json": typed_cfg,
            }
            _write_jsonl(ds_f, ds_row)
            if resume:
                dataset_done.add(instance_id)

    try:
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print("stats", stats)
        print("saved", stats_path)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
