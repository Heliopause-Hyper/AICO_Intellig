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


def _best_policy_value(pairs, tau):
    gains = []
    for pred, true_gain in pairs:
        gains.append(float(true_gain) if float(pred) > float(tau) else 0.0)
    return compute_gain_metrics(gains)


print("=" * 80)
print("THEORY VALIDATION E4: TAIL-RISK OBJECTIVE (CVaR-OPTIMAL THRESHOLDING)")
print("=" * 80)

K = 20
ALPHA = 0.10
CVaR_ALPHA = 0.10

feats, aucs_by_group, family_by_group = load_group_level_trajectory_features(decision_eval=K)

valid_groups = []
for gk, fam in family_by_group.items():
    aucs = aucs_by_group.get(gk) or {}
    if DEFAULT_ALGO not in aucs:
        continue
    if not any(c in aucs for c in CANDIDATE_POOL):
        continue
    if gk not in feats:
        continue
    valid_groups.append(gk)

unique_families = sorted({family_by_group[gk] for gk in valid_groups})

gains_fixed0 = []
gains_tau_mean = []
gains_tau_cvar = []
gains_conformal = []

taus_mean = []
taus_cvar = []
qs = []

for test_fam in unique_families:
    train_gks = [gk for gk in valid_groups if family_by_group[gk] != test_fam]
    test_gks = [gk for gk in valid_groups if family_by_group[gk] == test_fam]
    if not train_gks or not test_gks:
        continue

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    proper_train_idx, calib_idx = next(kf.split(train_gks))
    proper_train_gks = [train_gks[i] for i in proper_train_idx]
    calib_gks = [train_gks[i] for i in calib_idx]

    train_rows = []
    y_train = []
    for gk in proper_train_gks:
        aucs = aucs_by_group.get(gk) or {}
        def_auc = float(aucs.get(DEFAULT_ALGO))
        for cand in sorted(CANDIDATE_POOL):
            if cand not in aucs:
                continue
            train_rows.append({**feats[gk], "cand_algo": cand})
            y_train.append(def_auc - float(aucs[cand]))

    if not train_rows:
        continue

    reg, columns, cat_features = _fit_gain_regressor(train_rows, y_train)

    calib_pairs = []
    residuals = []
    for gk in calib_gks:
        aucs = aucs_by_group.get(gk) or {}
        candidates = [c for c in CANDIDATE_POOL if c in aucs]
        if not candidates:
            continue
        best_cand, best_pred = _best_candidate_prediction(
            reg, columns, cat_features, feats[gk], candidates
        )
        true_gain = float(aucs[DEFAULT_ALGO]) - float(aucs[best_cand])
        calib_pairs.append((float(best_pred), float(true_gain)))
        residuals.append(float(best_pred - true_gain))

    q_val = _conformal_quantile(residuals, ALPHA)
    qs.append(float(q_val))

    if not calib_pairs:
        continue

    preds = sorted(set(float(p) for p, _ in calib_pairs))
    grid = []
    if preds:
        qs_grid = np.quantile(np.asarray(preds, dtype=float), np.linspace(0.0, 1.0, 41))
        grid = sorted(set(float(x) for x in qs_grid))
    grid = sorted(set(grid + [0.0]))

    best_mean = None
    best_cvar = None
    tau_mean = 0.0
    tau_cvar = 0.0

    for tau in grid:
        m = _best_policy_value(calib_pairs, tau)
        if best_mean is None or m["mean_gain"] > best_mean:
            best_mean = float(m["mean_gain"])
            tau_mean = float(tau)
        if best_cvar is None or m["cvar_10"] > best_cvar:
            best_cvar = float(m["cvar_10"])
            tau_cvar = float(tau)

    taus_mean.append(float(tau_mean))
    taus_cvar.append(float(tau_cvar))

    for gk in test_gks:
        aucs = aucs_by_group.get(gk) or {}
        candidates = [c for c in CANDIDATE_POOL if c in aucs]
        if not candidates:
            continue
        best_cand, best_pred = _best_candidate_prediction(
            reg, columns, cat_features, feats[gk], candidates
        )
        true_gain = float(aucs[DEFAULT_ALGO]) - float(aucs[best_cand])

        gains_fixed0.append(float(true_gain) if float(best_pred) > 0.0 else 0.0)
        gains_tau_mean.append(float(true_gain) if float(best_pred) > float(tau_mean) else 0.0)
        gains_tau_cvar.append(float(true_gain) if float(best_pred) > float(tau_cvar) else 0.0)

        lower_bound = float(best_pred) - float(q_val)
        gains_conformal.append(float(true_gain) if lower_bound > 0.0 else 0.0)


summary = {
    "k": int(K),
    "alpha": float(ALPHA),
    "cvar_alpha": float(CVaR_ALPHA),
    "tau_mean_avg": float(np.mean(taus_mean)) if taus_mean else 0.0,
    "tau_mean_median": float(np.median(taus_mean)) if taus_mean else 0.0,
    "tau_cvar_avg": float(np.mean(taus_cvar)) if taus_cvar else 0.0,
    "tau_cvar_median": float(np.median(taus_cvar)) if taus_cvar else 0.0,
    "q_1_minus_alpha_avg": float(np.mean(qs)) if qs else 0.0,
    "policies": {
        "tau0_mean_opt": compute_gain_metrics(gains_fixed0),
        "tau_star_mean": compute_gain_metrics(gains_tau_mean),
        "tau_star_cvar": compute_gain_metrics(gains_tau_cvar),
        "conformal_lb": compute_gain_metrics(gains_conformal),
    },
}

out_dir = "/home/ycl/AICO-Intellig/results/theory_validation_latest"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "tail_risk_objective.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
    f.write("\n")

print("")
for name, m in summary["policies"].items():
    print(
        f"[{name}] mean={m['mean_gain']:+.3f} "
        f"cvar10={m['cvar_10']:+.3f} "
        f"loss={m['loss_rate']*100:5.1f}% "
        f"switch={m['switch_rate']*100:5.1f}%"
    )
