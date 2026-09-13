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


print("=" * 80)
print("THEORY VALIDATION E3: K TRADE-OFF DECOMPOSITION + SEQUENTIAL STOPPING")
print("=" * 80)

K_VALUES = [5, 10, 15, 20, 30]
ALPHA = 0.10

feature_by_k = {}
aucs_by_k = {}
family_by_k = {}

for k in K_VALUES:
    feats, aucs, fam_by_group = load_group_level_trajectory_features(decision_eval=k)
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

summary = {"fixed_k": {}, "sequential": {}}

for k in K_VALUES:
    feats_k = feature_by_k[int(k)]
    aucs_k = aucs_by_k[int(k)]

    all_gains = []
    all_q = []
    remain_fracs = []

    for test_fam in unique_families:
        train_gks = [gk for gk in valid_groups if fam_by_group[gk] != test_fam and gk in feats_k]
        test_gks = [gk for gk in valid_groups if fam_by_group[gk] == test_fam and gk in feats_k]
        if not train_gks or not test_gks:
            continue

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        proper_train_idx, calib_idx = next(kf.split(train_gks))
        proper_train_gks = [train_gks[i] for i in proper_train_idx]
        calib_gks = [train_gks[i] for i in calib_idx]

        train_rows = []
        y_train = []
        for gk in proper_train_gks:
            aucs = aucs_k.get(gk) or {}
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
        all_q.append(q_val)

        for gk in test_gks:
            aucs = aucs_k.get(gk) or {}
            candidates = [c for c in CANDIDATE_POOL if c in aucs]
            if not candidates:
                continue
            best_cand, best_pred = _best_candidate_prediction(
                reg, columns, cat_features, feats_k[gk], candidates
            )
            lower_bound = float(best_pred - q_val)
            switched = lower_bound > 0.0
            gain = float(aucs[DEFAULT_ALGO]) - float(aucs[best_cand]) if switched else 0.0
            all_gains.append(float(gain))

            budget_eval = float(feats_k[gk].get("budget_eval") or 0.0)
            remain = 0.0
            if budget_eval > 0.0:
                remain = max(0.0, (budget_eval - float(k)) / budget_eval)
            remain_fracs.append(float(remain))

    m = compute_gain_metrics(all_gains)
    summary["fixed_k"][str(k)] = {
        "k": int(k),
        "alpha": float(ALPHA),
        "n": int(m["n"]),
        "mean_gain": float(m["mean_gain"]),
        "cond_mean_gain": float(m["cond_mean_gain"]),
        "loss_rate": float(m["loss_rate"]),
        "switch_rate": float(m["switch_rate"]),
        "cvar_10": float(m["cvar_10"]),
        "worst_case": float(m["worst_case"]),
        "q_1_minus_alpha_mean": float(np.mean(all_q)) if all_q else 0.0,
        "q_1_minus_alpha_median": float(np.median(all_q)) if all_q else 0.0,
        "remaining_budget_fraction_mean": float(np.mean(remain_fracs)) if remain_fracs else 0.0,
    }


seq_gains = []
seq_k_used = []
seq_remain_fracs = []

for test_fam in unique_families:
    train_gks = [gk for gk in valid_groups if fam_by_group[gk] != test_fam]
    test_gks = [gk for gk in valid_groups if fam_by_group[gk] == test_fam]
    if not train_gks or not test_gks:
        continue

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    proper_train_idx, calib_idx = next(kf.split(train_gks))
    proper_train_gks = [train_gks[i] for i in proper_train_idx]
    calib_gks = [train_gks[i] for i in calib_idx]

    models = {}
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
        models[int(k)] = {
            "reg": reg,
            "columns": columns,
            "cat_features": cat_features,
            "q_val": float(q_val),
        }

    if not models:
        continue

    for gk in test_gks:
        decision = DEFAULT_ALGO
        k_used = int(K_VALUES[-1])
        best_cand_final = None

        for k in K_VALUES:
            pack = models.get(int(k))
            if pack is None:
                continue
            feats_k = feature_by_k[int(k)]
            aucs_k = aucs_by_k[int(k)]
            if gk not in feats_k:
                continue
            aucs = aucs_k.get(gk) or {}
            candidates = [c for c in CANDIDATE_POOL if c in aucs]
            if not candidates:
                continue
            best_cand, best_pred = _best_candidate_prediction(
                pack["reg"], pack["columns"], pack["cat_features"], feats_k[gk], candidates
            )
            lower_bound = float(best_pred - pack["q_val"])
            if lower_bound > 0.0:
                decision = best_cand
                k_used = int(k)
                best_cand_final = best_cand
                break

        aucs = aucs_by_k[int(K_VALUES[-1])].get(gk) or {}
        def_auc = float(aucs.get(DEFAULT_ALGO))
        gain = def_auc - float(aucs.get(decision, def_auc))
        seq_gains.append(float(gain if decision != DEFAULT_ALGO else 0.0))
        seq_k_used.append(int(k_used))

        feats_last = feature_by_k[int(k_used)].get(gk) or {}
        budget_eval = float(feats_last.get("budget_eval") or 0.0)
        remain = 0.0
        if budget_eval > 0.0:
            remain = max(0.0, (budget_eval - float(k_used)) / budget_eval)
        seq_remain_fracs.append(float(remain))

m_seq = compute_gain_metrics(seq_gains)
summary["sequential"] = {
    "k_values": [int(x) for x in K_VALUES],
    "alpha": float(ALPHA),
    "n": int(m_seq["n"]),
    "mean_gain": float(m_seq["mean_gain"]),
    "cond_mean_gain": float(m_seq["cond_mean_gain"]),
    "loss_rate": float(m_seq["loss_rate"]),
    "switch_rate": float(m_seq["switch_rate"]),
    "cvar_10": float(m_seq["cvar_10"]),
    "worst_case": float(m_seq["worst_case"]),
    "k_used_mean": float(np.mean(seq_k_used)) if seq_k_used else 0.0,
    "k_used_median": float(np.median(seq_k_used)) if seq_k_used else 0.0,
    "remaining_budget_fraction_mean": float(np.mean(seq_remain_fracs)) if seq_remain_fracs else 0.0,
}

out_dir = "/home/ycl/AICO-Intellig/results/theory_validation_latest"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "k_tradeoff_decomposition.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
    f.write("\n")

print("")
for k in K_VALUES:
    s = summary["fixed_k"][str(k)]
    print(
        f"[K={k:>2d}] mean={s['mean_gain']:+.3f} "
        f"cvar10={s['cvar_10']:+.3f} "
        f"switch={s['switch_rate']*100:5.1f}% "
        f"q={s['q_1_minus_alpha_mean']:.4f} "
        f"remain={s['remaining_budget_fraction_mean']*100:5.1f}%"
    )

s = summary["sequential"]
print("")
print(
    "[Sequential] "
    f"mean={s['mean_gain']:+.3f} "
    f"cvar10={s['cvar_10']:+.3f} "
    f"switch={s['switch_rate']*100:5.1f}% "
    f"k_used_mean={s['k_used_mean']:.1f} "
    f"remain={s['remaining_budget_fraction_mean']*100:5.1f}%"
)
