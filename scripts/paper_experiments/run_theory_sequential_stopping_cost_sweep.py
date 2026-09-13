import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath("/home/ycl/AICO-Intellig"))

from scripts.paper_experiments.theory_validation_common import (
    CANDIDATE_POOL,
    DEFAULT_ALGO,
    compute_gain_metrics,
    load_group_level_trajectory_features,
)


def _conformal_quantile(residuals, alpha: float) -> float:
    xs = np.asarray(list(residuals), dtype=float)
    if xs.size == 0:
        return 0.0
    n = int(xs.size)
    q_level = min(1.0, (n + 1.0) * (1.0 - float(alpha)) / float(n))
    return float(np.quantile(xs, q_level))


def _fit_gain_regressor(rows, ys):
    df_train = pd.DataFrame(rows).fillna(0.0)
    cat_features = [
        c
        for c in df_train.columns
        if df_train[c].dtype == object or isinstance(df_train[c].iloc[0], str)
    ]
    for c in cat_features:
        df_train[c] = df_train[c].astype(str)
    reg = CatBoostRegressor(
        iterations=200,
        learning_rate=0.05,
        depth=4,
        l2_leaf_reg=10.0,
        verbose=False,
        random_state=42,
        thread_count=1,
    )
    reg.fit(df_train, ys, cat_features=cat_features)
    return reg, df_train.columns.tolist(), cat_features


def _predict_gain(reg, columns, cat_features, row):
    df = pd.DataFrame([row]).reindex(columns=columns).fillna(0.0)
    for c in cat_features:
        df[c] = df[c].astype(str)
    return float(reg.predict(df)[0])


def _best_candidate_prediction(reg, columns, cat_features, base_features, candidates):
    preds = {}
    for cand in candidates:
        row = dict(base_features)
        row["cand_algo"] = cand
        preds[cand] = _predict_gain(reg, columns, cat_features, row)
    best_cand = max(preds, key=preds.get)
    return best_cand, float(preds[best_cand])


def _budget_fraction(feats, k: int) -> float:
    b = float(feats.get("budget_eval") or 0.0)
    if b <= 0.0:
        return 0.0
    return float(k) / float(b)


print("=" * 80)
print("THEORY VALIDATION E3b: SEQUENTIAL STOPPING WITH BUDGET-SQUEEZE REGULARIZATION")
print("=" * 80)

K_VALUES = [5, 10, 15, 20, 30]
ALPHA = 0.10
LAMBDAS = [0.0, 0.05, 0.10, 0.20, 0.40, 0.80]

feature_by_k = {}
aucs_by_k = {}
family_by_k = {}
for k in K_VALUES:
    feats, aucs, fam_by_group = load_group_level_trajectory_features(decision_eval=int(k))
    feature_by_k[int(k)] = feats
    aucs_by_k[int(k)] = aucs
    family_by_k[int(k)] = fam_by_group

fam_by_group = family_by_k[int(K_VALUES[-1])]

valid_groups = []
for gk in sorted(fam_by_group.keys()):
    aucs = aucs_by_k[int(K_VALUES[-1])].get(gk) or {}
    if DEFAULT_ALGO not in aucs:
        continue
    if not any(c in aucs for c in CANDIDATE_POOL):
        continue
    valid_groups.append(gk)

unique_families = sorted({fam_by_group[gk] for gk in valid_groups})

test_cache = {}
aucs_last = aucs_by_k[int(K_VALUES[-1])]

for test_fam in unique_families:
    train_gks = [gk for gk in valid_groups if fam_by_group[gk] != test_fam]
    test_gks = [gk for gk in valid_groups if fam_by_group[gk] == test_fam]
    if not train_gks or not test_gks:
        continue

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    proper_train_idx, calib_idx = next(kf.split(train_gks))
    proper_train_gks = [train_gks[i] for i in proper_train_idx]
    calib_gks = [train_gks[i] for i in calib_idx]

    per_k = {}
    for k in K_VALUES:
        feats_k = feature_by_k[int(k)]
        aucs_k = aucs_by_k[int(k)]

        train_rows = []
        y_train = []
        for gk in proper_train_gks:
            if gk not in feats_k:
                continue
            aucs = aucs_k.get(gk) or {}
            if DEFAULT_ALGO not in aucs:
                continue
            def_auc = float(aucs.get(DEFAULT_ALGO))
            for cand in sorted(CANDIDATE_POOL):
                if cand not in aucs:
                    continue
                train_rows.append({**feats_k[gk], "cand_algo": cand})
                y_train.append(def_auc - float(aucs[cand]))
        if not train_rows:
            continue

        reg, columns, cat_features = _fit_gain_regressor(train_rows, y_train)

        residuals = []
        for gk in calib_gks:
            if gk not in feats_k:
                continue
            aucs = aucs_k.get(gk) or {}
            candidates = [c for c in CANDIDATE_POOL if c in aucs]
            if not candidates:
                continue
            best_cand, best_pred = _best_candidate_prediction(
                reg, columns, cat_features, feats_k[gk], candidates
            )
            true_gain = float(aucs[DEFAULT_ALGO]) - float(aucs[best_cand])
            residuals.append(float(best_pred - true_gain))

        q_val = _conformal_quantile(residuals, ALPHA)

        per_group = {}
        for gk in test_gks:
            if gk not in feats_k:
                continue
            aucs = aucs_k.get(gk) or {}
            candidates = [c for c in CANDIDATE_POOL if c in aucs]
            if not candidates:
                continue
            best_cand, best_pred = _best_candidate_prediction(
                reg, columns, cat_features, feats_k[gk], candidates
            )
            lower_bound = float(best_pred - q_val)
            frac = _budget_fraction(feats_k[gk], int(k))

            last = aucs_last.get(gk) or {}
            def_auc = float(last.get(DEFAULT_ALGO))
            raw_gain = float(def_auc - float(last.get(best_cand, def_auc)))

            per_group[gk] = {
                "best_cand": str(best_cand),
                "lower_bound": float(lower_bound),
                "k_over_b": float(frac),
                "raw_gain": float(raw_gain),
            }

        per_k[int(k)] = {
            "q_val": float(q_val),
            "groups": per_group,
        }

    test_cache[str(test_fam)] = {
        "test_groups": list(test_gks),
        "per_k": per_k,
    }


def _metrics_fixed_k(k: int, lam: float):
    gains_raw = []
    gains_util = []
    fracs = []

    for fam, payload in test_cache.items():
        per_k = payload["per_k"].get(int(k))
        if not per_k:
            continue
        for gk, row in per_k["groups"].items():
            frac = float(row["k_over_b"])
            fracs.append(frac)
            switched = float(row["lower_bound"]) > float(lam) * frac
            raw = float(row["raw_gain"]) if switched else 0.0
            util = float(raw - float(lam) * frac) if switched else 0.0
            gains_raw.append(raw)
            gains_util.append(util)

    return {
        "k": int(k),
        "lambda": float(lam),
        "alpha": float(ALPHA),
        "raw": compute_gain_metrics(gains_raw),
        "utility": compute_gain_metrics(gains_util),
        "k_over_b_mean": float(np.mean(fracs)) if fracs else 0.0,
    }


def _metrics_sequential(lam: float):
    gains_raw = []
    gains_util = []
    k_used = []
    fracs = []

    for fam, payload in test_cache.items():
        per_k = payload["per_k"]
        groups = set(payload["test_groups"])
        for gk in groups:
            chosen_k = None
            chosen_row = None
            for k in K_VALUES:
                pk = per_k.get(int(k))
                if not pk:
                    continue
                row = pk["groups"].get(gk)
                if not row:
                    continue
                if float(row["lower_bound"]) > float(lam) * float(row["k_over_b"]):
                    chosen_k = int(k)
                    chosen_row = row
                    break
            if chosen_k is None or chosen_row is None:
                gains_raw.append(0.0)
                gains_util.append(0.0)
                k_used.append(int(K_VALUES[-1]))
                fracs.append(0.0)
                continue

            raw = float(chosen_row["raw_gain"])
            frac = float(chosen_row["k_over_b"])
            util = float(raw - float(lam) * frac)

            gains_raw.append(raw)
            gains_util.append(util)
            k_used.append(int(chosen_k))
            fracs.append(frac)

    return {
        "k_values": [int(x) for x in K_VALUES],
        "lambda": float(lam),
        "alpha": float(ALPHA),
        "raw": compute_gain_metrics(gains_raw),
        "utility": compute_gain_metrics(gains_util),
        "k_used_mean": float(np.mean(k_used)) if k_used else 0.0,
        "k_used_median": float(np.median(k_used)) if k_used else 0.0,
        "k_over_b_mean": float(np.mean(fracs)) if fracs else 0.0,
    }


out = {"fixed_k": {}, "sequential": {}}
for lam in LAMBDAS:
    out["fixed_k"][str(lam)] = {str(k): _metrics_fixed_k(int(k), float(lam)) for k in K_VALUES}
    out["sequential"][str(lam)] = _metrics_sequential(float(lam))

out_dir = "/home/ycl/AICO-Intellig/results/theory_validation_latest"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "sequential_stopping_cost_sweep.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
    f.write("\n")

print("")
for lam in LAMBDAS:
    s = out["sequential"][str(lam)]
    print(
        f"[lambda={lam:>4.2f}] "
        f"util_mean={s['utility']['mean_gain']:+.3f} "
        f"util_cvar10={s['utility']['cvar_10']:+.3f} "
        f"raw_mean={s['raw']['mean_gain']:+.3f} "
        f"raw_cvar10={s['raw']['cvar_10']:+.3f} "
        f"switch={s['raw']['switch_rate']*100:5.1f}% "
        f"k_med={s['k_used_median']:>4.0f}"
    )
