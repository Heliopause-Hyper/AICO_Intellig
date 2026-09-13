import copy
import json
import math
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath("/home/ycl/AICO-Intellig"))

from scripts.paper_experiments.theory_validation_common import (
    CANDIDATE_POOL,
    DECISION_EVAL,
    DEFAULT_ALGO,
    OUT_DIRS,
)
from scripts.train_curve_policy import (
    AUC_CAP,
    RunKey,
    _build_pairwise_features,
    _checkpoint_features_from_events,
    _collect_run_trajectories,
    _collect_shards,
    _exclude_row,
    _instance_features,
    _read_jsonl,
    _safe_float,
)


print("=" * 80)
print("THEORY VALIDATION E5: AFFINE ROBUSTNESS AUDIT")
print("=" * 80)

TRANSFORMS = [
    (0.5, 0.0),
    (2.0, 0.0),
    (1.0, 10.0),
    (3.0, 5.0),
]


def transform_events(events, a, b):
    out = []
    for ev in events:
        new_ev = copy.deepcopy(ev)
        loss = _safe_float(new_ev.get("loss_value"), None)
        if loss is not None:
            new_ev["loss_value"] = a * float(loss) + b
        out.append(new_ev)
    return out


run_summaries = {}
pi_by_instance = {}
iteration_event_paths = []
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

trajectories = _collect_run_trajectories(iteration_event_paths, set(run_summaries.keys()), max_eval=DECISION_EVAL)
group_runs = defaultdict(dict)
actual_aucs = defaultdict(dict)

for rid, srow in run_summaries.items():
    inst_id = str(srow.get("instance_id") or "")
    if not inst_id:
        continue
    seed = int(srow.get("seed") or 0)
    algo = f"{srow.get('backend_lib') or ''}-{srow.get('method') or ''}"
    gk = RunKey(inst_id, seed).key()
    group_runs[gk][algo] = {"summary": srow, "rid": rid}
    auc = _safe_float(srow.get("anytime_auc"), None)
    if auc is not None:
        actual_aucs[gk][algo] = math.log1p(min(float(AUC_CAP), max(0.0, float(auc))))


orig_rows = []
orig_targets = []
orig_group_keys = []
transform_rows = {f"a{a}_b{b}": [] for a, b in TRANSFORMS}
feature_deviation = defaultdict(list)
decision_consistency = {f"a{a}_b{b}": [] for a, b in TRANSFORMS}

for gk, algos in group_runs.items():
    if DEFAULT_ALGO not in algos:
        continue
    if not all(c in algos for c in CANDIDATE_POOL):
        continue

    def_run = algos[DEFAULT_ALGO]
    inst_id = str(def_run["summary"].get("instance_id") or "")
    pi = pi_by_instance.get(inst_id)
    f_inst = _instance_features(pi)
    def_events = trajectories.get(def_run["rid"], [])
    def_ck = _checkpoint_features_from_events(def_events, [DECISION_EVAL]).get(DECISION_EVAL)
    if not def_ck:
        continue

    for cand in sorted(CANDIDATE_POOL):
        cand_run = algos[cand]
        cand_events = trajectories.get(cand_run["rid"], [])
        cand_ck = _checkpoint_features_from_events(cand_events, [DECISION_EVAL]).get(DECISION_EVAL)
        if not cand_ck:
            continue
        row = _build_pairwise_features(f_inst, def_ck, cand_ck, cand)
        orig_rows.append(row)
        orig_targets.append(actual_aucs[gk][DEFAULT_ALGO] - actual_aucs[gk][cand])
        orig_group_keys.append(gk)

        for a, b in TRANSFORMS:
            key = f"a{a}_b{b}"
            def_ck_t = _checkpoint_features_from_events(transform_events(def_events, a, b), [DECISION_EVAL]).get(DECISION_EVAL)
            cand_ck_t = _checkpoint_features_from_events(transform_events(cand_events, a, b), [DECISION_EVAL]).get(DECISION_EVAL)
            if not def_ck_t or not cand_ck_t:
                continue
            row_t = _build_pairwise_features(f_inst, def_ck_t, cand_ck_t, cand)
            transform_rows[key].append((gk, cand, row_t))
            for feat, val in row.items():
                if isinstance(val, str):
                    continue
                new_val = row_t.get(feat)
                if new_val is None:
                    continue
                denom = max(1e-9, abs(float(val)))
                feature_deviation[(key, feat)].append(abs(float(new_val) - float(val)) / denom)


df_train = pd.DataFrame(orig_rows).fillna(0.0)
cat_features = [
    c
    for c in df_train.columns
    if df_train[c].dtype == object or isinstance(df_train[c].iloc[0], str)
]
for c in cat_features:
    df_train[c] = df_train[c].astype(str)
reg = CatBoostRegressor(
    iterations=600,
    learning_rate=0.03,
    depth=4,
    l2_leaf_reg=10.0,
    verbose=False,
    random_state=42,
    thread_count=1,
)
reg.fit(df_train, orig_targets, cat_features=cat_features)

orig_pred_by_group = defaultdict(dict)
for idx, row in enumerate(orig_rows):
    gk = orig_group_keys[idx]
    cand = row["cand_algo"]
    df_p = pd.DataFrame([row]).reindex(columns=df_train.columns).fillna(0.0)
    for c in cat_features:
        df_p[c] = df_p[c].astype(str)
    orig_pred_by_group[gk][cand] = reg.predict(df_p)[0]

for key, entries in transform_rows.items():
    pred_by_group = defaultdict(dict)
    pred_delta = []
    for gk, cand, row_t in entries:
        df_p = pd.DataFrame([row_t]).reindex(columns=df_train.columns).fillna(0.0)
        for c in cat_features:
            df_p[c] = df_p[c].astype(str)
        pred = reg.predict(df_p)[0]
        pred_by_group[gk][cand] = pred
        pred_delta.append(abs(pred - orig_pred_by_group[gk][cand]))

    stable = 0
    total = 0
    for gk, preds in pred_by_group.items():
        if gk not in orig_pred_by_group:
            continue
        orig_best = max(orig_pred_by_group[gk], key=orig_pred_by_group[gk].get)
        new_best = max(preds, key=preds.get)
        orig_switch = orig_pred_by_group[gk][orig_best] > 0.1
        new_switch = preds[new_best] > 0.1
        stable += 1 if (orig_best == new_best and orig_switch == new_switch) else 0
        total += 1
    decision_consistency[key] = {
        "mean_abs_prediction_shift": float(np.mean(pred_delta)) if pred_delta else 0.0,
        "route_consistency": float(stable / max(1, total)),
        "n_groups": int(total),
    }


summary = {
    "feature_deviation": {},
    "decision_consistency": decision_consistency,
}

print("")
for key in sorted(decision_consistency):
    info = decision_consistency[key]
    print(f"[{key}]")
    print(f"  - Mean abs prediction shift: {info['mean_abs_prediction_shift']:.4f}")
    print(f"  - Route consistency:         {info['route_consistency'] * 100:.2f}%")
    print(f"  - Groups audited:            {info['n_groups']}")

for (key, feat), vals in sorted(feature_deviation.items()):
    if not vals:
        continue
    summary["feature_deviation"].setdefault(key, {})[feat] = {
        "median_rel_dev": float(np.median(vals)),
        "mean_rel_dev": float(np.mean(vals)),
        "max_rel_dev": float(np.max(vals)),
    }

out_dir = "/home/ycl/AICO-Intellig/results/theory_validation_latest"
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "affine_invariance_audit.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
    f.write("\n")
