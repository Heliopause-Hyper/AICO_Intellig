import argparse
import json
import os
import re
import glob
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

FEATURE_ABS_CAP = 1e6
EXCLUDED_PROBLEM_TYPES = {"corca_state", "corca_evol", "corca_xenon"}


class CatBoostDictRegressor:
    def __init__(self, random_state: int = 0):
        self.random_state = int(random_state)
        self.model = None
        self.columns: List[str] = []
        self.cat_features: List[str] = []

    @staticmethod
    def _is_categorical_series(series: Any) -> bool:
        import pandas as pd
        from pandas.api.types import is_object_dtype, is_string_dtype

        return bool(
            is_object_dtype(series.dtype)
            or is_string_dtype(series.dtype)
            or isinstance(series.dtype, pd.CategoricalDtype)
        )

    def _to_frame(self, rows: List[Dict[str, Any]]):
        import pandas as pd

        df = pd.DataFrame.from_records(rows)
        if self.columns:
            df = df.reindex(columns=self.columns)
        for col in df.columns:
            series = df[col]
            if col in self.cat_features or self._is_categorical_series(series):
                df[col] = series.fillna("").astype(str)
            else:
                df[col] = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
        return df

    def fit(self, X: List[Dict[str, Any]], y: List[float]):
        from catboost import CatBoostRegressor

        df = self._to_frame(X)
        self.columns = list(df.columns)
        self.cat_features = [c for c in self.columns if self._is_categorical_series(df[c])]
        self.model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=self.random_state,
            depth=8,
            learning_rate=0.05,
            iterations=600,
            l2_leaf_reg=5.0,
            verbose=False,
        )
        self.model.fit(df, y, cat_features=self.cat_features, verbose=False)
        return self

    def predict(self, X: List[Dict[str, Any]]):
        if self.model is None:
            raise RuntimeError("model not fitted")
        df = self._to_frame(X)
        return self.model.predict(df)


if __name__ == "__main__":
    sys.modules.setdefault("scripts.train_algo_ranker", sys.modules[__name__])
    CatBoostDictRegressor.__module__ = "scripts.train_algo_ranker"


@dataclass(frozen=True)
class Candidate:
    optimizer: str
    method: str

    def key(self) -> str:
        return f"{self.optimizer}-{self.method}"


def _read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def _problem_type_from_row(row: Dict[str, Any]) -> str:
    pt = str(row.get("problem_type") or "").strip()
    if pt:
        return pt
    typed_cfg = row.get("typed_config_json")
    if isinstance(typed_cfg, dict):
        pt = str(typed_cfg.get("problem_type") or "").strip()
        if pt:
            return pt
    iid = str(row.get("instance_id") or "").strip()
    if iid.startswith("inst_corca_"):
        parts = iid.split("_")
        if len(parts) >= 3:
            return f"{parts[1]}_{parts[2]}"
    return ""


def _exclude_row(row: Dict[str, Any]) -> bool:
    return _problem_type_from_row(row) in EXCLUDED_PROBLEM_TYPES


def _collect_paths(out_dir: str) -> Tuple[List[str], List[str]]:
    out_dir = os.path.abspath(out_dir)
    shard_glob = os.path.join(out_dir, "shard_*")
    shards = sorted([p for p in glob.glob(shard_glob) if os.path.isdir(p)])
    runs: List[str] = []
    dss: List[str] = []
    if shards:
        for s in shards:
            rp = os.path.join(s, "runs.jsonl")
            dp = os.path.join(s, "dataset.jsonl")
            if os.path.exists(rp):
                runs.append(rp)
            if os.path.exists(dp):
                dss.append(dp)
    else:
        rp = os.path.join(out_dir, "runs.jsonl")
        dp = os.path.join(out_dir, "dataset.jsonl")
        if os.path.exists(rp):
            runs.append(rp)
        if os.path.exists(dp):
            dss.append(dp)
    return runs, dss


def _collect_optional_paths(out_dir: str, filename: str) -> List[str]:
    out_dir = os.path.abspath(out_dir)
    shard_glob = os.path.join(out_dir, "shard_*")
    shards = sorted([p for p in glob.glob(shard_glob) if os.path.isdir(p)])
    out: List[str] = []
    if shards:
        for s in shards:
            fp = os.path.join(s, filename)
            if os.path.exists(fp):
                out.append(fp)
    else:
        fp = os.path.join(out_dir, filename)
        if os.path.exists(fp):
            out.append(fp)
    return out


def _safe_float(x: Any, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            cap = float(FEATURE_ABS_CAP)
            if v > cap:
                v = cap
            elif v < -cap:
                v = -cap
            return float(v)
    except Exception:
        pass
    return None if default is None else float(default)


def _drop_family_like_features(f: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(f)
    for k in [
        "problem_type",
        "template_name",
        "typed_template_name",
        "family_version",
        "generation_bucket",
        "task_name",
    ]:
        out.pop(k, None)
    return out


def _typed_config(inst: Dict[str, Any], pi_row: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if isinstance(pi_row, dict) and isinstance(pi_row.get("typed_config_json"), dict):
        return dict(pi_row.get("typed_config_json") or {})
    if isinstance(inst.get("typed_config_json"), dict):
        return dict(inst.get("typed_config_json") or {})
    return {}


def _add_scalar_feature(f: Dict[str, Any], key: str, value: Any) -> None:
    try:
        v = float(value)
        if math.isfinite(v):
            cap = float(FEATURE_ABS_CAP)
            if v > cap:
                v = cap
            elif v < -cap:
                v = -cap
            f[key] = float(v)
    except Exception:
        pass


def _features_from_instance(inst: Dict[str, Any], pi_row: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    f: Dict[str, Any] = {}
    f["problem_type"] = str(inst.get("problem_type"))
    if pi_row:
        f["physics_domain"] = str(pi_row.get("physics_domain") or "")
        f["template_name"] = str(pi_row.get("template_name") or "")
        f["pde_type"] = str(pi_row.get("pde_type") or "")
        f["pi_status"] = str(pi_row.get("status") or "")

    opt_params = inst.get("opt_params") or []
    f["n_params"] = int(len(opt_params))
    for p in opt_params:
        f[f"param_{p}"] = 1

    pr = inst.get("param_ranges") or {}
    widths: List[float] = []
    for _, bounds in pr.items():
        try:
            lo, hi = bounds
            widths.append(float(hi) - float(lo))
        except Exception:
            continue
    if widths:
        f["range_w_mean"] = float(sum(widths) / len(widths))
        f["range_w_max"] = float(max(widths))
        f["range_w_min"] = float(min(widths))
    else:
        f["range_w_mean"] = 0.0
        f["range_w_max"] = 0.0
        f["range_w_min"] = 0.0

    budget = inst.get("budget") or {}
    try:
        f["budget_iter"] = int(budget.get("max_iterations", 0))
    except Exception:
        f["budget_iter"] = 0
    try:
        f["budget_time"] = float(budget.get("time_limit_s", 0.0))
    except Exception:
        f["budget_time"] = 0.0

    expr = str(inst.get("objective_expression") or "")
    f["expr_len"] = int(len(expr))
    f["has_abs"] = int("abs(" in expr)
    keys = re.findall(r"result\.get\(\s*'([^']+)'", expr)
    for k in keys:
        f[f"expr_key_{k}"] = 1

    if pi_row:
        for src_key, dst_key in [
            ("decision_dim", "pi_decision_dim"),
            ("num_constraints", "pi_num_constraints"),
            ("constraint_density", "pi_constraint_density"),
            ("budget_per_dim", "pi_budget_per_dim"),
            ("avg_var_range", "pi_avg_var_range"),
            ("max_var_range", "pi_max_var_range"),
            ("min_var_range", "pi_min_var_range"),
            ("budget_eval", "pi_budget_eval"),
            ("budget_walltime_sec", "pi_budget_walltime_sec"),
        ]:
            _add_scalar_feature(f, dst_key, pi_row.get(src_key))

        smoke = pi_row.get("smoke") or {}
        if isinstance(smoke, dict):
            f["smoke_success"] = int(bool(smoke.get("success", False)))
            for sk in [
                "execution_time_mid",
                "target_value",
                "gap_mid",
                "pressuredrop_mid",
                "avgpressure_mid",
                "pmax_mid",
                "velmagmax_mid",
                "pressuredrop_lo",
                "pressuredrop_hi",
                "avgpressure_lo",
                "avgpressure_hi",
                "pmax_lo",
                "pmax_hi",
                "velmagmax_lo",
                "velmagmax_hi",
            ]:
                _add_scalar_feature(f, f"smoke_{sk}", smoke.get(sk))
            target_metric = smoke.get("target_metric")
            if target_metric:
                f["smoke_target_metric"] = str(target_metric)

    typed = _typed_config(inst, pi_row)
    if typed:
        f["task_type"] = str(typed.get("task_type") or "")
        f["objective_type"] = str(typed.get("objective_type") or "")
        f["primary_metric"] = str(typed.get("primary_metric") or "")
        f["task_name"] = str(typed.get("task_name") or "")
        f["generation_bucket"] = str(typed.get("generation_bucket") or "")
        f["family_version"] = str(typed.get("family_version") or "")
        f["typed_template_name"] = str(typed.get("template_name") or "")
        if typed.get("pde_type") is not None:
            f["pde_type"] = str(typed.get("pde_type") or f.get("pde_type") or "")
        _add_scalar_feature(f, "typed_decision_dim", typed.get("decision_dim"))
        _add_scalar_feature(f, "typed_num_constraints", typed.get("num_constraints"))
        _add_scalar_feature(f, "typed_budget_eval", typed.get("budget_eval"))
        _add_scalar_feature(f, "typed_budget_walltime_sec", typed.get("budget_walltime_sec"))

        for p in typed.get("variable_name_json") or []:
            f[f"typed_var_{p}"] = 1
        for cname in typed.get("constraint_name_json") or []:
            if cname:
                f[f"constraint_name_{cname}"] = 1
        for ctype in typed.get("constraint_type_json") or []:
            if ctype:
                f[f"constraint_type_{ctype}"] = 1

        targets = typed.get("targets") or {}
        if isinstance(targets, dict):
            for tk, tv in targets.items():
                f[f"has_target_{tk}"] = 1
                _add_scalar_feature(f, f"target_{tk}", tv)

        metric_est = typed.get("metric_estimates") or {}
        if isinstance(metric_est, dict):
            for mk, mv in metric_est.items():
                if not isinstance(mv, dict):
                    continue
                _add_scalar_feature(f, f"metric_est_{mk}_lo", mv.get("lo"))
                _add_scalar_feature(f, f"metric_est_{mk}_hi", mv.get("hi"))
                lo_v = _safe_float(mv.get("lo"), None)
                hi_v = _safe_float(mv.get("hi"), None)
                if lo_v is not None and hi_v is not None:
                    f[f"metric_est_{mk}_span"] = float(hi_v - lo_v)
    return f


def _features_from_run(run: Dict[str, Any]) -> Dict[str, Any]:
    f: Dict[str, Any] = {}
    f["optimizer"] = str(run.get("optimizer"))
    f["method"] = str(run.get("method"))
    return f


def _load_instance_map(dataset_paths: List[str]) -> Dict[str, Dict[str, Any]]:
    m: Dict[str, Dict[str, Any]] = {}
    for p in dataset_paths:
        for row in _read_jsonl(p):
            if _exclude_row(row):
                continue
            iid = row.get("instance_id")
            if not iid:
                continue
            if iid not in m:
                m[str(iid)] = row
    return m


def _load_runs(runs_paths: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in runs_paths:
        base_dir = os.path.abspath(os.path.dirname(p))
        for row in _read_jsonl(p):
            if _exclude_row(row):
                continue
            if "__base_dir" not in row:
                row["__base_dir"] = base_dir
            out.append(row)
    return out


def _load_problem_instance_map(paths: List[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for p in paths:
        for row in _read_jsonl(p):
            if _exclude_row(row):
                continue
            iid = row.get("instance_id")
            if not iid:
                continue
            iid = str(iid)
            if iid not in out:
                out[iid] = row
    return out


def _load_run_checkpoint_map(paths: List[str], max_ratio: int = 20) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    max_ratio_i = int(max_ratio)
    for p in paths:
        for row in _read_jsonl(p):
            if _exclude_row(row):
                continue
            rid = row.get("run_id")
            if not rid:
                continue
            try:
                ratio = int(row.get("checkpoint_ratio", 10**9))
            except Exception:
                ratio = 10**9
            if ratio > max_ratio_i:
                continue
            out[str(rid)].append(row)
    for rid, rows in out.items():
        rows.sort(key=lambda r: (int(r.get("checkpoint_ratio", 10**9)), int(r.get("eval_index", 10**9))))
    return out


def _infer_candidates(runs: List[Dict[str, Any]]) -> List[Candidate]:
    seen = set()
    out: List[Candidate] = []
    for r in runs:
        o = r.get("optimizer")
        m = r.get("method")
        if not o or not m:
            continue
        k = (str(o), str(m))
        if k in seen:
            continue
        seen.add(k)
        out.append(Candidate(optimizer=str(o), method=str(m)))
    out.sort(key=lambda c: c.key())
    return out


def _true_winners_by_instance(runs: List[Dict[str, Any]], candidate_key_set: set) -> Dict[str, str]:
    by_i: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        iid = r.get("instance_id")
        if not iid:
            continue
        k = f"{r.get('optimizer')}-{r.get('method')}"
        if k not in candidate_key_set:
            continue
        if not r.get("success", False):
            continue
        bv = r.get("best_value")
        if bv is None:
            continue
        try:
            by_i[str(iid)][k].append(float(bv))
        except Exception:
            continue

    winners: Dict[str, str] = {}
    for iid, mp in by_i.items():
        best_k = None
        best_v = float("inf")
        for k, vals in mp.items():
            if not vals:
                continue
            v = sum(vals) / float(len(vals))
            if v < best_v:
                best_v = v
                best_k = k
        if best_k is not None:
            winners[iid] = best_k
    return winners


def _trajectory_path(run: Dict[str, Any]) -> Optional[str]:
    base_dir = run.get("__base_dir")
    if not base_dir:
        return None
    run_id = run.get("run_id")
    problem_type = run.get("problem_type")
    optimizer = run.get("optimizer")
    method = run.get("method")
    if not run_id or not problem_type or not optimizer or not method:
        return None
    p = os.path.join(
        str(base_dir),
        "runs",
        str(problem_type),
        f"{optimizer}_{method}",
        str(run_id),
        "trajectory.json",
    )
    return p if os.path.exists(p) else None


def _warm_features(run: Dict[str, Any], warmup_k: int) -> Dict[str, Any]:
    k = int(warmup_k)
    out: Dict[str, Any] = {
        "warm_n": 0,
        "warm_best": 0.0,
        "warm_first": 0.0,
        "warm_last": 0.0,
        "warm_improve": 0.0,
        "warm_slope": 0.0,
        "warm_std": 0.0,
        "warm_fail_rate": 1.0,
        "warm_best_pos": 0.0,
    }
    if k <= 0:
        return out

    tp = _trajectory_path(run)
    if not tp:
        return out
    try:
        with open(tp, "r", encoding="utf-8") as f:
            traj = json.load(f)
    except Exception:
        return out
    if not isinstance(traj, list) or not traj:
        return out

    seg = traj[:k] if k > 0 else traj
    vals: List[float] = []
    succ: List[int] = []
    for step in seg:
        try:
            v = float(step.get("objective"))
            if not math.isfinite(v):
                continue
            cap = 1e12
            if v > cap:
                v = cap
            elif v < -cap:
                v = -cap
            vals.append(v)
            s = step.get("result", {}).get("success", True)
            succ.append(1 if bool(s) else 0)
        except Exception:
            continue
    if not vals:
        return out

    n = len(vals)
    best = min(vals)
    first = vals[0]
    last = vals[-1]
    improve = float(first - best)
    slope = float((last - first) / float(max(1, n - 1)))
    mean = 0.0
    m2 = 0.0
    for i, v in enumerate(vals, start=1):
        delta = v - mean
        mean += delta / float(i)
        m2 += delta * (v - mean)
    var = m2 / float(max(1, n - 1))
    if var < 0.0:
        var = 0.0
    std = math.sqrt(var)
    best_pos = float(vals.index(best) / float(max(1, n - 1)))
    fail_rate = float(1.0 - (sum(succ) / float(len(succ)))) if succ else 1.0

    out.update(
        {
            "warm_n": n,
            "warm_best": float(best),
            "warm_first": float(first),
            "warm_last": float(last),
            "warm_improve": float(improve),
            "warm_slope": float(slope),
            "warm_std": float(std),
            "warm_fail_rate": float(fail_rate),
            "warm_best_pos": float(best_pos),
        }
    )
    return out


def _checkpoint_features(run: Dict[str, Any], checkpoint_map: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ckpt_has_prefix": 0,
    }
    run_id = run.get("run_id")
    if not run_id:
        return out
    rows = checkpoint_map.get(str(run_id)) or []
    if not rows:
        return out
    out["ckpt_has_prefix"] = 1

    def _pick_at_or_before(target_ratio: int) -> Optional[Dict[str, Any]]:
        chosen = None
        for row in rows:
            try:
                ratio = int(row.get("checkpoint_ratio", 10**9))
            except Exception:
                continue
            if ratio <= int(target_ratio):
                chosen = row
            else:
                break
        return chosen

    selected: Dict[str, Dict[str, Any]] = {}
    for target in (1, 5, 10, 20):
        row = _pick_at_or_before(target)
        if row:
            selected[str(target)] = row
            out[f"ckpt_r{target}_seen"] = 1
            _add_scalar_feature(out, f"ckpt_r{target}_ratio", row.get("checkpoint_ratio"))
            _add_scalar_feature(out, f"ckpt_r{target}_eval_index", row.get("eval_index"))
            _add_scalar_feature(out, f"ckpt_r{target}_best_loss", row.get("best_loss"))
            _add_scalar_feature(out, f"ckpt_r{target}_best_feasible_loss", row.get("best_feasible_loss"))
            _add_scalar_feature(out, f"ckpt_r{target}_constraint_violation_total", row.get("constraint_violation_total"))
            _add_scalar_feature(out, f"ckpt_r{target}_recent_improvement_rate", row.get("recent_improvement_rate"))
            _add_scalar_feature(out, f"ckpt_r{target}_stagnation_length", row.get("stagnation_length"))
            _add_scalar_feature(out, f"ckpt_r{target}_feasible_ratio", row.get("feasible_ratio"))
            _add_scalar_feature(out, f"ckpt_r{target}_first_feasible_eval_index", row.get("first_feasible_eval_index"))
            _add_scalar_feature(out, f"ckpt_r{target}_avg_eval_time", row.get("avg_eval_time"))
            _add_scalar_feature(out, f"ckpt_r{target}_recent_failure_rate", row.get("recent_failure_rate"))
            _add_scalar_feature(out, f"ckpt_r{target}_normalized_improvement", row.get("normalized_improvement"))
            _add_scalar_feature(out, f"ckpt_r{target}_auc_regret_prefix", row.get("auc_regret_prefix"))
            _add_scalar_feature(out, f"ckpt_r{target}_incumbent_age", row.get("incumbent_age"))
            out[f"ckpt_r{target}_feasible_flag"] = int(bool(row.get("feasible_flag", False)))

    row1 = selected.get("1")
    row10 = selected.get("10")
    row20 = selected.get("20") or selected.get("10") or selected.get("5") or selected.get("1")
    if row1 and row10:
        b1 = _safe_float(row1.get("best_loss"), None)
        b10 = _safe_float(row10.get("best_loss"), None)
        if b1 is not None and b10 is not None:
            out["ckpt_delta_best_1_10"] = float(b1 - b10)
    if row10 and row20:
        b10 = _safe_float(row10.get("best_loss"), None)
        b20 = _safe_float(row20.get("best_loss"), None)
        if b10 is not None and b20 is not None:
            out["ckpt_delta_best_10_20"] = float(b10 - b20)
    if row20:
        _add_scalar_feature(out, "ckpt_prefix_last_ratio", row20.get("checkpoint_ratio"))
        _add_scalar_feature(out, "ckpt_prefix_last_feasible_ratio", row20.get("feasible_ratio"))
        _add_scalar_feature(out, "ckpt_prefix_last_auc_regret", row20.get("auc_regret_prefix"))
    return out


def train_and_eval(
    out_dir: str,
    out_model_path: str,
    test_size: float,
    random_state: int,
    candidate_whitelist: Optional[List[str]],
    target: str,
    model_type: str,
    warmup_k: int,
    drop_family_like: bool = False,
) -> int:
    runs_paths, dataset_paths = _collect_paths(out_dir)
    if not runs_paths or not dataset_paths:
        raise RuntimeError("未找到 runs.jsonl 或 dataset.jsonl（可传入 out_dir 或包含 shard_*/ 的目录）")
    problem_instance_paths = _collect_optional_paths(out_dir, "problem_instance.jsonl")
    run_checkpoint_paths = _collect_optional_paths(out_dir, "run_checkpoint.jsonl")

    inst_map = _load_instance_map(dataset_paths)
    pi_map = _load_problem_instance_map(problem_instance_paths)
    checkpoint_map = _load_run_checkpoint_map(run_checkpoint_paths, max_ratio=20)
    runs = _load_runs(runs_paths)
    if not runs:
        raise RuntimeError("runs 为空")

    candidates = _infer_candidates(runs)
    if candidate_whitelist:
        allow = set(candidate_whitelist)
        candidates = [c for c in candidates if c.key() in allow]
    if not candidates:
        raise RuntimeError("候选算法为空")
    cand_keys = [c.key() for c in candidates]
    cand_key_set = set(cand_keys)

    target = str(target).strip().lower()
    if target not in {"best_value", "rank"}:
        raise ValueError("target 必须是 best_value 或 rank")

    rank_target: Dict[Tuple[str, str], float] = {}
    if target == "rank":
        by_i: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        for r in runs:
            iid = r.get("instance_id")
            if not iid:
                continue
            k = f"{r.get('optimizer')}-{r.get('method')}"
            if k not in cand_key_set:
                continue
            if not r.get("success", False):
                continue
            bv = r.get("best_value")
            if bv is None:
                continue
            try:
                by_i[str(iid)][k].append(float(bv))
            except Exception:
                continue

        for iid, mp in by_i.items():
            items = []
            for k, vals in mp.items():
                if vals:
                    items.append((k, sum(vals) / float(len(vals))))
            if not items:
                continue
            items.sort(key=lambda x: x[1])
            denom = float(len(items) - 1) if len(items) > 1 else 1.0
            for idx, (k, _) in enumerate(items):
                rank_target[(iid, k)] = float(idx) / denom

    X: List[Dict[str, Any]] = []
    y: List[float] = []
    groups: List[str] = []
    runs_used: List[Dict[str, Any]] = []

    dropped = 0
    for r in runs:
        iid = r.get("instance_id")
        if not iid:
            dropped += 1
            continue
        iid = str(iid)
        inst = inst_map.get(iid)
        if not inst:
            dropped += 1
            continue
        ckey = f"{r.get('optimizer')}-{r.get('method')}"
        if ckey not in cand_key_set:
            continue
        bv = r.get("best_value")
        if bv is None:
            dropped += 1
            continue
        try:
            bv_f = float(bv)
        except Exception:
            dropped += 1
            continue
        if target == "rank":
            if not r.get("success", False):
                continue
            t = rank_target.get((iid, ckey))
            if t is None:
                continue
            y_val = float(t)
        else:
            y_val = bv_f
        feat = {}
        feat.update(_features_from_instance(inst, pi_map.get(iid)))
        feat.update(_features_from_run(r))
        feat["success"] = int(bool(r.get("success", False)))
        feat.update(_warm_features(r, warmup_k))
        feat.update(_checkpoint_features(r, checkpoint_map))
        if bool(drop_family_like):
            feat = _drop_family_like_features(feat)
        X.append(feat)
        y.append(y_val)
        groups.append(iid)
        runs_used.append(r)

    if not X:
        raise RuntimeError("可训练样本为空（检查 best_value / instance_id 是否齐全）")

    from sklearn.feature_extraction import DictVectorizer
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import Pipeline
    import joblib
    import numpy as np

    splitter = GroupShuffleSplit(n_splits=1, test_size=float(test_size), random_state=int(random_state))
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    model_type = str(model_type).strip().lower()
    if model_type == "sgd":
        from sklearn.linear_model import SGDRegressor

        pipe = Pipeline(
            steps=[
                ("vec", DictVectorizer(sparse=True)),
                ("reg", SGDRegressor(loss="huber", max_iter=5000, tol=1e-4, random_state=int(random_state))),
            ]
        )
    elif model_type == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor

        pipe = Pipeline(
            steps=[
                ("vec", DictVectorizer(sparse=False)),
                (
                    "reg",
                    ExtraTreesRegressor(
                        n_estimators=400,
                        random_state=int(random_state),
                        n_jobs=-1,
                        min_samples_leaf=2,
                    ),
                ),
            ]
        )
    elif model_type == "catboost":
        pipe = CatBoostDictRegressor(random_state=int(random_state))
    else:
        raise ValueError("model 必须是 sgd、extra_trees 或 catboost")

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    groups_test = [groups[i] for i in test_idx]

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    mae = float(np.mean(np.abs(pred - np.array(y_test, dtype=float))))

    test_runs = [runs_used[i] for i in test_idx]
    true_winner = _true_winners_by_instance(test_runs, cand_key_set)
    rec = {}
    warm_pick: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if int(warmup_k) > 0:
        by_i_alg: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for r in test_runs:
            iid = str(r.get("instance_id"))
            k = f"{r.get('optimizer')}-{r.get('method')}"
            if k in cand_key_set:
                by_i_alg[(iid, k)].append(r)
        for key, lst in by_i_alg.items():
            lst.sort(key=lambda x: int(x.get("seed", 10**9)))
            warm_pick[key] = lst[0]
    for iid in set(groups_test):
        inst = inst_map.get(iid)
        if not inst:
            continue
        pi_row = pi_map.get(iid)
        rows = []
        keys = []
        for c in candidates:
            ckey = c.key()
            feat = {}
            feat.update(_features_from_instance(inst, pi_row))
            feat["optimizer"] = c.optimizer
            feat["method"] = c.method
            feat["success"] = 1
            r0 = None
            if int(warmup_k) > 0:
                r0 = warm_pick.get((iid, ckey))
                if r0:
                    feat.update(_warm_features(r0, warmup_k))
                else:
                    feat.update(_warm_features({}, warmup_k))
            if r0:
                feat.update(_checkpoint_features(r0, checkpoint_map))
            else:
                feat.update(_checkpoint_features({}, checkpoint_map))
            if bool(drop_family_like):
                feat = _drop_family_like_features(feat)
            rows.append(feat)
            keys.append(ckey)
        scores = pipe.predict(rows)
        best_i = int(np.argmin(scores))
        rec[iid] = keys[best_i]

    both = [iid for iid in rec.keys() if iid in true_winner]
    acc = (sum(1 for iid in both if rec[iid] == true_winner[iid]) / float(len(both))) if both else 0.0

    os.makedirs(os.path.dirname(out_model_path), exist_ok=True)
    payload = {
        "model": pipe,
        "candidates": [c.__dict__ for c in candidates],
        "out_dir": os.path.abspath(out_dir),
        "stats": {
            "runs_paths": runs_paths,
            "dataset_paths": dataset_paths,
            "problem_instance_paths": problem_instance_paths,
            "run_checkpoint_paths": run_checkpoint_paths,
            "samples": len(X),
            "dropped": dropped,
            "instances_in_map": len(inst_map),
            "problem_instances_in_map": len(pi_map),
            "runs_with_checkpoint_prefix": int(sum(1 for r in runs_used if (r.get("run_id") and checkpoint_map.get(str(r.get("run_id")))))),
            "test_instances_scored": len(both),
            "mae": mae,
            "rec_top1_acc": acc,
            "target": target,
            "model": model_type,
            "warmup_k": int(warmup_k),
            "drop_family_like": bool(drop_family_like),
        },
    }
    joblib.dump(payload, out_model_path)

    print("samples", len(X), "instances", len(inst_map), "dropped", dropped)
    print("candidates", len(candidates), Counter([c.key() for c in candidates]))
    print("mae", mae)
    print("rec_top1_acc", acc, "test_instances", len(both))
    print("saved", out_model_path)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--candidate_whitelist", default="")
    parser.add_argument("--target", default="rank")
    parser.add_argument("--model", default="extra_trees")
    parser.add_argument("--warmup_k", type=int, default=0)
    parser.add_argument("--drop_family_like", action="store_true")
    args = parser.parse_args()

    allow = [x.strip() for x in str(args.candidate_whitelist).split(",") if x.strip()]
    return train_and_eval(
        out_dir=args.out_dir,
        out_model_path=args.out,
        test_size=float(args.test_size),
        random_state=int(args.random_state),
        candidate_whitelist=allow or None,
        target=str(args.target),
        model_type=str(args.model),
        warmup_k=int(args.warmup_k),
        drop_family_like=bool(args.drop_family_like),
    )


if __name__ == "__main__":
    raise SystemExit(main())
