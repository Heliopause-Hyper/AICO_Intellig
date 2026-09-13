from __future__ import annotations

import math
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from scripts.train_algo_ranker import _drop_family_like_features, _features_from_instance
from scripts.train_curve_policy import (
    AUC_CAP,
    RunKey,
    _build_pairwise_dataset,
    _checkpoint_features_from_events,
    _collect_run_trajectories,
    _collect_shards,
    _exclude_row,
    _family_name,
    _instance_features,
    _read_jsonl,
    _safe_float,
)


PROJECT_ROOT = "/home/ycl/AICO-Intellig"
DEFAULT_ALGO = "SciPy-Nelder-Mead"
CANDIDATE_POOL = {"Nevergrad-NGOpt", "SciPy-COBYLA", "SciPy-DE"}
DECISION_EVAL = 20
STATIC_DATASET_PATH = os.path.join(
    PROJECT_ROOT, "results", "static_ranker_baseline_latest", "dataset.jsonl"
)

OUT_DIRS = [
    os.path.join(PROJECT_ROOT, "results", "ex_advection2d_v1_full_200_m100_t3600_r1_20260709_231148"),
    os.path.join(PROJECT_ROOT, "results", "ex_blackscholes2d_v1_full_200_m100_t3600_r1_20260709_104116"),
    os.path.join(PROJECT_ROOT, "results", "ex_heat_time_v1_full_200_m100_t3600_r1_"),
    os.path.join(PROJECT_ROOT, "results", "heat_v1_full_180_m30_t90_r3"),
    os.path.join(PROJECT_ROOT, "results", "iaea_v1_full_180_m30_t90_r3_20260708_102705"),
    os.path.join(PROJECT_ROOT, "results", "sns_v1_full_150_m40_t90_r4_20260429_234621"),
    os.path.join(PROJECT_ROOT, "results", "sns_v3_full_240_m30_t90_r3_more1"),
    os.path.join(PROJECT_ROOT, "results", "thmf_v1_full_180_m30_t90_r3_20260708_102705"),
]


def compute_cvar(values: Iterable[float], alpha: float = 0.1) -> float:
    xs = np.asarray(list(values), dtype=float)
    if xs.size == 0:
        return 0.0
    threshold = float(np.percentile(xs, alpha * 100.0))
    tail = xs[xs <= threshold]
    if tail.size == 0:
        return 0.0
    return float(np.mean(tail))


def compute_gain_metrics(gains: Iterable[float]) -> Dict[str, float]:
    xs = np.asarray(list(gains), dtype=float)
    if xs.size == 0:
        return {
            "n": 0,
            "mean_gain": 0.0,
            "win_rate": 0.0,
            "loss_rate": 0.0,
            "switch_rate": 0.0,
            "cond_mean_gain": 0.0,
            "cvar_10": 0.0,
            "worst_case": 0.0,
        }
    switched = xs != 0.0
    return {
        "n": int(xs.size),
        "mean_gain": float(np.mean(xs)),
        "win_rate": float(np.mean(xs > 0.0)),
        "loss_rate": float(np.mean(xs < 0.0)),
        "switch_rate": float(np.mean(switched)),
        "cond_mean_gain": float(np.mean(xs[switched])) if np.any(switched) else 0.0,
        "cvar_10": compute_cvar(xs, 0.1),
        "worst_case": float(np.min(xs)),
    }


def load_static_feature_dict() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in _read_jsonl(STATIC_DATASET_PATH):
        iid = str(row.get("instance_id") or "")
        if not iid:
            continue
        seed = int(row.get("seed") or 0)
        gk = f"{iid}__seed{seed}"
        feats = _features_from_instance(row, row)
        out[gk] = _drop_family_like_features(feats)
    return out


def load_pairwise_dataset():
    return _build_pairwise_dataset(
        out_dirs=OUT_DIRS,
        default_algo=DEFAULT_ALGO,
        candidate_pool=CANDIDATE_POOL,
        decision_eval=DECISION_EVAL,
    )


def load_pairwise_dataset_for_k(decision_eval: int):
    return _build_pairwise_dataset(
        out_dirs=OUT_DIRS,
        default_algo=DEFAULT_ALGO,
        candidate_pool=CANDIDATE_POOL,
        decision_eval=int(decision_eval),
    )


def load_group_level_trajectory_features(
    decision_eval: int = DECISION_EVAL,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, float]], Dict[str, str]]:
    run_summaries: Dict[str, Dict[str, Any]] = {}
    pi_by_instance: Dict[str, Dict[str, Any]] = {}
    iteration_event_paths: List[str] = []

    for out_dir in OUT_DIRS:
        for shard in _collect_shards(out_dir):
            rp = os.path.join(shard, "run_summary.jsonl")
            pip = os.path.join(shard, "problem_instance.jsonl")
            iep = os.path.join(shard, "iteration_event.jsonl")
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

    decision_eval = int(decision_eval)
    trajectories = _collect_run_trajectories(
        iteration_event_paths, set(run_summaries.keys()), max_eval=decision_eval
    )

    group_runs: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for rid, srow in run_summaries.items():
        inst_id = str(srow.get("instance_id") or "")
        if not inst_id:
            continue
        seed = int(srow.get("seed") or 0)
        algo = f"{srow.get('backend_lib') or ''}-{srow.get('method') or ''}"
        rk = RunKey(inst_id, seed).key()
        group_runs[rk][algo] = {"summary": srow, "rid": rid}

    group_features: Dict[str, Dict[str, Any]] = {}
    actual_aucs_dict: Dict[str, Dict[str, float]] = defaultdict(dict)
    family_by_group: Dict[str, str] = {}

    for gk, algos in group_runs.items():
        if DEFAULT_ALGO not in algos:
            continue
        def_run = algos[DEFAULT_ALGO]
        def_summary = def_run["summary"]
        def_rid = def_run["rid"]
        inst_id = str(def_summary.get("instance_id") or "")
        pi = pi_by_instance.get(inst_id)

        fam = _family_name(pi, def_summary)
        family_by_group[gk] = fam

        def_events = trajectories.get(def_rid, [])
        def_ck = _checkpoint_features_from_events(def_events, [decision_eval]).get(decision_eval)
        if not def_ck:
            continue

        f_inst = _instance_features(pi)
        feats = dict(f_inst)
        for key, value in def_ck.items():
            if value is None:
                continue
            feats[f"base_{key}"] = value
        group_features[gk] = feats

        for algo, payload in algos.items():
            auc = _safe_float(payload["summary"].get("anytime_auc"), None)
            if auc is None:
                continue
            actual_aucs_dict[gk][algo] = math.log1p(min(float(AUC_CAP), max(0.0, float(auc))))

    return group_features, actual_aucs_dict, family_by_group


def group_truth_labels(actual_aucs_dict: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    truth: Dict[str, str] = {}
    for gk, aucs in actual_aucs_dict.items():
        if not aucs:
            continue
        truth[gk] = min(aucs, key=aucs.get)
    return truth
