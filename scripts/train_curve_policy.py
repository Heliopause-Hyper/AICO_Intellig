import argparse
import glob
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


AUC_CAP = 1e6
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
            depth=6,
            learning_rate=0.05,
            iterations=500,
            l2_leaf_reg=5.0,
            bootstrap_type="No",
            verbose=False,
        )
        self.model.fit(df, y, cat_features=self.cat_features, verbose=False)
        return self

    def predict(self, X: List[Dict[str, Any]]):
        if self.model is None:
            raise RuntimeError("model not fitted")
        df = self._to_frame(X)
        return self.model.predict(df)


def _dump_regressor(path: str, reg: CatBoostDictRegressor) -> None:
    import joblib

    payload = {
        "model": reg.model,
        "columns": list(reg.columns),
        "cat_features": list(reg.cat_features),
    }
    joblib.dump(payload, path)


if __name__ == "__main__":
    sys.modules.setdefault("scripts.train_curve_policy", sys.modules[__name__])
    CatBoostDictRegressor.__module__ = "scripts.train_curve_policy"


@dataclass(frozen=True)
class RunKey:
    instance_id: str
    seed: int

    def key(self) -> str:
        return f"{self.instance_id}__seed{self.seed}"


def _read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


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


def _collect_shards(out_dir: str) -> List[str]:
    out_dir = os.path.abspath(out_dir)
    shards = sorted([p for p in glob.glob(os.path.join(out_dir, "shard_*")) if os.path.isdir(p)])
    if shards:
        return shards
    return [out_dir]


def _safe_float(x: Any, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            return float(v)
    except Exception:
        pass
    if default is None:
        return None
    return float(default)


def _infer_family_from_row(row: Optional[Dict[str, Any]]) -> str:
    if not row:
        return ""

    fam = str(row.get("family_version") or "").strip()
    if fam:
        return fam

    problem_type = str(row.get("problem_type") or "").strip()
    if problem_type:
        return problem_type

    typed_cfg = row.get("typed_config_json")
    if isinstance(typed_cfg, dict):
        fam = str(typed_cfg.get("family_version") or "").strip()
        if fam:
            return fam
        problem_type = str(typed_cfg.get("problem_type") or "").strip()
        if problem_type:
            return problem_type

    template_name = str(row.get("template_name") or "").strip()
    if template_name == "thmf":
        return "thermal_fins_family_v1"
    if template_name == "sns":
        return "sns_flow_family_v1"
    if template_name == "iaea":
        return "neutron_diffusion_family_v1"
    if template_name == "ex_heat_time":
        return "ex_heat_time_family_v1"
    if template_name == "ex_advection2d":
        return "ex_advection2d_family_v1"
    if template_name == "ex_blackscholes2d":
        return "ex_blackscholes2d_family_v1"

    instance_id = str(row.get("instance_id") or "").strip()
    if "thermal_fins_family_v1" in instance_id:
        return "thermal_fins_family_v1"
    if "ex_heat_time_family_v1" in instance_id:
        return "ex_heat_time_family_v1"
    if "ex_advection2d_family_v1" in instance_id:
        return "ex_advection2d_family_v1"
    if "ex_blackscholes2d_family_v1" in instance_id:
        return "ex_blackscholes2d_family_v1"
    if "neutron_diffusion_family_v1" in instance_id:
        return "neutron_diffusion_family_v1"
    if "sns_flow_family_v3" in instance_id:
        return "sns_flow_family_v3"
    if "sns_flow_family_v1" in instance_id:
        return "sns_flow_family_v1"
    if "heat_v1" in instance_id:
        return "heat_v1"
    if "iaea_v1" in instance_id:
        return "iaea_v1"

    return ""


def _family_name(pi: Optional[Dict[str, Any]], fallback_row: Optional[Dict[str, Any]] = None) -> str:
    fam = _infer_family_from_row(pi)
    if fam:
        return fam
    fam = _infer_family_from_row(fallback_row)
    if fam:
        return fam
    return "unknown"


def _instance_features(pi: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    f: Dict[str, Any] = {}
    if not pi:
        return f
    # STRICTLY NO FAMILY-LIKE FEATURES HERE TO ENFORCE GENERALIZATION
    f["decision_dim"] = int(pi.get("decision_dim") or 0)
    f["num_constraints"] = int(pi.get("num_constraints") or 0)
    f["constraint_density"] = _safe_float(pi.get("constraint_density"), 0.0)
    f["budget_eval"] = int(pi.get("budget_eval") or 0)
    f["budget_walltime_sec"] = _safe_float(pi.get("budget_walltime_sec"), 0.0)
    f["avg_var_range"] = _safe_float(pi.get("avg_var_range"), 0.0)
    f["max_var_range"] = _safe_float(pi.get("max_var_range"), 0.0)
    f["min_var_range"] = _safe_float(pi.get("min_var_range"), 0.0)
    f["budget_per_dim"] = _safe_float(pi.get("budget_per_dim"), 0.0)
    return f


def _checkpoint_features_from_events(
    events: List[Dict[str, Any]],
    eval_points: List[int],
) -> Dict[int, Dict[str, Any]]:
    if not events:
        return {}
    eval_points = sorted(set(int(x) for x in eval_points if int(x) > 0))
    max_k = int(max(eval_points)) if eval_points else 0

    best_loss = float("inf")
    best_feas = float("inf")
    first_loss: Optional[float] = None
    first_feasible_idx: Optional[int] = None
    feasible_count = 0
    total_eval_time = 0.0
    best_last_update = 0
    best_prefix_area = 0.0

    last_window = 5
    losses: List[float] = []
    feas_flags: List[bool] = []
    improved_flags: List[bool] = []

    out: Dict[int, Dict[str, Any]] = {}

    for ev in events:
        idx = int(ev.get("eval_index") or 0)
        if idx <= 0:
            continue
        if max_k and idx > max_k:
            break
        loss = _safe_float(ev.get("loss_value"), float("inf"))
        feas = bool(ev.get("feasible_flag", False))
        if first_loss is None and math.isfinite(loss):
            first_loss = float(loss)
        if feas:
            feasible_count += 1
            if first_feasible_idx is None:
                first_feasible_idx = int(idx)
        total_eval_time += _safe_float(ev.get("eval_time_sec"), 0.0)
        improved = False
        if loss < best_loss:
            best_loss = float(loss)
            best_last_update = int(idx)
            improved = True
        if feas and loss < best_feas:
            best_feas = float(loss)
        best_prefix_area += float(best_loss if math.isfinite(best_loss) else 0.0)

        losses.append(float(loss))
        feas_flags.append(bool(feas))
        improved_flags.append(bool(improved))
        if len(losses) > last_window:
            losses.pop(0)
            feas_flags.pop(0)
            improved_flags.pop(0)

        if idx not in eval_points:
            continue

        seg_losses = list(losses)
        seg_improved = list(improved_flags)
        seg_feas = list(feas_flags)
        recent_best = min(seg_losses) if seg_losses else float("inf")
        recent_impr = 0.0
        if seg_losses:
            recent_impr = max(0.0, float(seg_losses[0]) - float(recent_best))
        stagnation = 0
        for flag in reversed(seg_improved):
            if flag:
                break
            stagnation += 1
        recent_fail = 0.0
        if seg_feas:
            recent_fail = sum(1 for x in seg_feas if not x) / float(len(seg_feas))
        feas_ratio = feasible_count / float(idx) if idx > 0 else 0.0
        avg_eval_time = total_eval_time / float(max(1, idx))

        best_feasible_loss = None
        if best_feas != float("inf"):
            best_feasible_loss = float(best_feas)

        norm_impr = 0.0
        if first_loss is not None:
            denom = max(1e-9, abs(float(first_loss)))
            norm_impr = max(0.0, (float(first_loss) - float(best_loss)) / denom) if best_loss != float("inf") else 0.0

        area_norm = 0.0
        if first_loss is not None:
            denom = max(1e-9, abs(float(first_loss)))
            area_norm = float(best_prefix_area) / (float(idx) * denom)

        out[int(idx)] = {
            "best_loss": float(best_loss) if best_loss != float("inf") else None,
            "best_feasible_loss": best_feasible_loss,
            "feasible_ratio": float(feas_ratio),
            "first_feasible_eval_index": int(first_feasible_idx) if first_feasible_idx is not None else None,
            "recent_improvement_rate": float(recent_impr),
            "stagnation_length": int(stagnation),
            "recent_failure_rate": float(recent_fail),
            "avg_eval_time": float(avg_eval_time),
            "normalized_improvement": float(norm_impr),
            "incumbent_age": int(idx - best_last_update) if best_last_update else int(idx),
            "prefix_best_area_norm": float(area_norm),
        }
    return out


def _collect_run_trajectories(
    iteration_event_paths: List[str],
    run_id_allow: Optional[set],
    max_eval: int,
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for fp in iteration_event_paths:
        for ev in _read_jsonl(fp):
            rid = str(ev.get("run_id") or "")
            if not rid:
                continue
            if run_id_allow is not None and rid not in run_id_allow:
                continue
            idx = int(ev.get("eval_index") or 0)
            if idx <= 0:
                continue
            if max_eval and idx > max_eval:
                continue
            out.setdefault(rid, []).append(ev)
    for rid, seq in out.items():
        seq.sort(key=lambda x: int(x.get("eval_index") or 0))
    return out


def _rel_gain(default_v: float, cand_v: float, lower_is_better: bool = True) -> float:
    denom = max(1e-9, abs(float(default_v)))
    if lower_is_better:
        return float(default_v - cand_v) / float(denom)
    return float(cand_v - default_v) / float(denom)


def _build_pairwise_features(f_inst: Dict[str, Any], default_ck: Dict[str, Any], cand_ck: Dict[str, Any], cand_algo: str) -> Dict[str, Any]:
    f = dict(f_inst)
    f["cand_algo"] = str(cand_algo)

    d_best = _safe_float(default_ck.get("best_loss"), 1e12)
    c_best = _safe_float(cand_ck.get("best_loss"), 1e12)
    d_auc_reg = _safe_float(default_ck.get("prefix_best_area_norm"), 1e12)
    c_auc_reg = _safe_float(cand_ck.get("prefix_best_area_norm"), 1e12)
    d_feas = _safe_float(default_ck.get("feasible_ratio"), 0.0)
    c_feas = _safe_float(cand_ck.get("feasible_ratio"), 0.0)
    d_stag = _safe_float(default_ck.get("stagnation_length"), 0.0)
    c_stag = _safe_float(cand_ck.get("stagnation_length"), 0.0)
    d_eval_t = max(1e-9, _safe_float(default_ck.get("avg_eval_time"), 1e-9))
    c_eval_t = max(1e-9, _safe_float(cand_ck.get("avg_eval_time"), 1e-9))
    d_fail = _safe_float(default_ck.get("recent_failure_rate"), 1.0)
    c_fail = _safe_float(cand_ck.get("recent_failure_rate"), 1.0)

    f["rel_best_gain"] = _rel_gain(d_best, c_best, True)
    f["rel_auc_gain"] = _rel_gain(d_auc_reg, c_auc_reg, True)
    f["diff_feasible"] = float(c_feas - d_feas)
    f["diff_stagnation"] = float(d_stag - c_stag)
    f["eval_time_ratio"] = float(c_eval_t / d_eval_t)
    f["diff_failure"] = float(d_fail - c_fail)

    return f


def _build_pairwise_dataset(
    out_dirs: List[str],
    default_algo: str,
    candidate_pool: set,
    decision_eval: int,
) -> Tuple[List[Dict[str, Any]], List[float], List[str], List[str], Dict[str, Dict[str, float]], Dict[str, str]]:
    
    run_summaries: Dict[str, Dict[str, Any]] = {}
    pi_by_instance: Dict[str, Dict[str, Any]] = {}
    iteration_event_paths: List[str] = []

    for out_dir in out_dirs:
        shards = _collect_shards(out_dir)
        for s in shards:
            rp = os.path.join(s, "run_summary.jsonl")
            pip = os.path.join(s, "problem_instance.jsonl")
            iep = os.path.join(s, "iteration_event.jsonl")
            if os.path.exists(rp):
                for row in _read_jsonl(rp):
                    if _exclude_row(row):
                        continue
                    rid = str(row.get("run_id") or "")
                    if rid:
                        run_summaries[rid] = row
            if os.path.exists(pip):
                for row in _read_jsonl(pip):
                    if _exclude_row(row):
                        continue
                    iid = str(row.get("instance_id") or "")
                    if iid:
                        pi_by_instance[iid] = row
            if os.path.exists(iep):
                iteration_event_paths.append(iep)

    if not run_summaries or not iteration_event_paths:
        return [], [], [], [], {}, {}

    trajectories = _collect_run_trajectories(iteration_event_paths, set(run_summaries.keys()), max_eval=decision_eval)

    # Group runs by RunKey
    group_runs: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for rid, srow in run_summaries.items():
        inst_id = str(srow.get("instance_id") or "")
        if not inst_id:
            continue
        seed = int(srow.get("seed") or 0)
        algo = f"{srow.get('backend_lib') or ''}-{srow.get('method') or ''}"
        rk = RunKey(inst_id, seed).key()
        group_runs[rk][algo] = {
            "summary": srow,
            "rid": rid
        }

    rows: List[Dict[str, Any]] = []
    y_auc_gain: List[float] = []
    group_keys: List[str] = []
    families: List[str] = []
    
    actual_aucs_dict: Dict[str, Dict[str, float]] = defaultdict(dict)
    family_by_group: Dict[str, str] = {}

    for rk, algos in group_runs.items():
        if default_algo not in algos:
            continue
            
        def_run = algos[default_algo]
        def_rid = def_run["rid"]
        def_summary = def_run["summary"]
        inst_id = str(def_summary.get("instance_id") or "")
        pi = pi_by_instance.get(inst_id)
        
        fam = _family_name(pi, def_summary)
        family_by_group[rk] = fam
        
        def_auc = _safe_float(def_summary.get("anytime_auc"), None)
        if def_auc is None:
            continue
        def_auc_score = math.log1p(min(float(AUC_CAP), max(0.0, float(def_auc))))
        actual_aucs_dict[rk][default_algo] = def_auc_score
            
        def_events = trajectories.get(def_rid, [])
        def_ck = _checkpoint_features_from_events(def_events, [decision_eval]).get(decision_eval)
        if not def_ck:
            continue

        f_inst = _instance_features(pi)

        for cand_algo, cand_run in algos.items():
            if cand_algo == default_algo:
                continue
            if candidate_pool and cand_algo not in candidate_pool:
                continue
                
            cand_rid = cand_run["rid"]
            cand_summary = cand_run["summary"]
            cand_auc = _safe_float(cand_summary.get("anytime_auc"), None)
            if cand_auc is None:
                continue
            cand_auc_score = math.log1p(min(float(AUC_CAP), max(0.0, float(cand_auc))))
            actual_aucs_dict[rk][cand_algo] = cand_auc_score
            
            cand_events = trajectories.get(cand_rid, [])
            cand_ck = _checkpoint_features_from_events(cand_events, [decision_eval]).get(decision_eval)
            if not cand_ck:
                continue

            f_pair = _build_pairwise_features(f_inst, def_ck, cand_ck, cand_algo)
            gain = cand_auc_score - def_auc_score
            
            rows.append(f_pair)
            y_auc_gain.append(gain)
            group_keys.append(rk)
            families.append(fam)

    return rows, y_auc_gain, group_keys, families, actual_aucs_dict, family_by_group


def _metric_summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0}
    xs = sorted(float(v) for v in values)
    n = len(xs)
    mid = n // 2
    median = xs[mid] if n % 2 == 1 else 0.5 * (xs[mid - 1] + xs[mid])
    return {
        "mean": float(sum(xs) / float(n)),
        "median": float(median),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dirs", nargs="+", required=True)
    ap.add_argument("--default_algo", type=str, default="SciPy-Nelder-Mead")
    ap.add_argument("--candidate_pool", type=str, default="Nevergrad-NGOpt,SciPy-COBYLA,SciPy-DE")
    ap.add_argument("--decision_eval", type=int, default=20)
    ap.add_argument("--random_state", type=int, default=0)
    ap.add_argument("--out_model_prefix", type=str, default="")
    ap.add_argument("--switch_threshold", type=float, default=0.0)
    args = ap.parse_args()

    decision_eval = int(args.decision_eval)
    candidate_pool = {x.strip() for x in str(args.candidate_pool).split(",") if x.strip()}
    
    rows, y_auc_gain, group_keys, families, actual_aucs_dict, family_by_group = _build_pairwise_dataset(
        out_dirs=args.out_dirs,
        default_algo=args.default_algo,
        candidate_pool=candidate_pool,
        decision_eval=decision_eval,
    )

    if not rows:
        raise RuntimeError("no pairwise training rows collected")

    unique_families = sorted(set(families))
    print(f"Loaded {len(rows)} pairwise rows across {len(set(group_keys))} groups in {len(unique_families)} families.")

    lofo_results = []
    
    # Leave-One-Family-Out (LOFO) Evaluation
    wins_vs_default_auc = 0
    total_groups = 0
    n_switched = 0
    auc_gains_vs_default = []
    family_stats = defaultdict(lambda: {"n": 0, "switched": 0, "auc_gains": []})
    
    # We also keep track of test predictions to group by family
    for test_fam in unique_families:
        train_idxs = [i for i, f in enumerate(families) if f != test_fam]
        test_idxs = [i for i, f in enumerate(families) if f == test_fam]
        
        if not train_idxs or not test_idxs:
            continue
            
        X_train = [rows[i] for i in train_idxs]
        y_train = [y_auc_gain[i] for i in train_idxs]
        
        X_test = [rows[i] for i in test_idxs]
        
        m_auc = CatBoostDictRegressor(random_state=args.random_state).fit(X_train, y_train)
        pred_gains = m_auc.predict(X_test)
        
        # Group predictions by group_key
        preds_by_group = defaultdict(list)
        for i, pred in zip(test_idxs, pred_gains):
            gk = group_keys[i]
            cand = rows[i]["cand_algo"]
            preds_by_group[gk].append((pred, cand))
            
        for gk, preds in preds_by_group.items():
            best_pred, best_cand = max(preds, key=lambda x: x[0])
            
            total_groups += 1
            fam = family_by_group[gk]
            fs = family_stats[fam]
            fs["n"] += 1
            
            def_auc = actual_aucs_dict[gk][args.default_algo]
            
            if best_pred > args.switch_threshold:
                # Switch!
                n_switched += 1
                fs["switched"] += 1
                cand_auc = actual_aucs_dict[gk][best_cand]
                actual_gain = cand_auc - def_auc
            else:
                # Stay with default
                actual_gain = 0.0
                
            auc_gains_vs_default.append(actual_gain)
            fs["auc_gains"].append(actual_gain)
            if actual_gain > 0:
                wins_vs_default_auc += 1

    # Train final model on ALL data
    final_model = CatBoostDictRegressor(random_state=args.random_state).fit(rows, y_auc_gain)

    # Compile metrics
    family_out = {}
    for fam, stat in sorted(family_stats.items()):
        family_out[fam] = {
            "n_groups": int(stat["n"]),
            "switch_rate": float(stat["switched"]) / float(max(1, stat["n"])),
            "auc_gain_vs_default": _metric_summary(stat["auc_gains"]),
        }

    metrics = {
        "n_groups_total": int(total_groups),
        "n_families": len(unique_families),
        "decision_eval": decision_eval,
        "default_algo": args.default_algo,
        "candidate_pool": sorted(candidate_pool),
        "switch_threshold": args.switch_threshold,
        "switch_rate": float(n_switched) / float(max(1, total_groups)),
        "win_rate_vs_default_auc": float(wins_vs_default_auc) / float(max(1, total_groups)),
        "auc_gain_vs_default": _metric_summary(auc_gains_vs_default),
        "by_family": family_out,
    }

    out_prefix = str(args.out_model_prefix or "").strip()
    if not out_prefix:
        import time
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_prefix = os.path.join("/home/ycl/AICO-Intellig/results", f"pairwise_curve_policy_eval{decision_eval}_{ts}")

    os.makedirs(out_prefix, exist_ok=True)
    _dump_regressor(os.path.join(out_prefix, "model_pairwise_auc_gain.joblib"), final_model)
    with open(os.path.join(out_prefix, "lofo_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(out_prefix)

if __name__ == "__main__":
    main()
