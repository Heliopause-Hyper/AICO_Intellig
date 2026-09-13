import json
import math
import os
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath("/home/ycl/AICO-Intellig"))

from scripts.paper_experiments.theory_validation_common import (
    DEFAULT_ALGO,
    compute_cvar,
    load_pairwise_dataset,
)


print("=" * 80)
print("THEORY VALIDATION E2: HARMFUL-SWITCH CALIBRATION UNDER LOFO")
print("=" * 80)


ALPHAS = [0.05, 0.10, 0.20, 0.30]

rows, _, group_keys, families, actual_aucs_dict, family_by_group = load_pairwise_dataset()
unique_groups = sorted(set(group_keys))
unique_families = sorted(set(families))

stats = {
    str(alpha): {
        "gains": [],
        "switched": 0,
        "harmful": 0,
        "n": 0,
    }
    for alpha in ALPHAS
}

for test_fam in unique_families:
    train_gks = [gk for gk in unique_groups if family_by_group[gk] != test_fam]
    test_gks = [gk for gk in unique_groups if family_by_group[gk] == test_fam]
    if not train_gks or not test_gks:
        continue

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    proper_train_idx, calib_idx = next(kf.split(train_gks))
    proper_train_gks = [train_gks[i] for i in proper_train_idx]
    calib_gks = [train_gks[i] for i in calib_idx]

    t_idxs = [i for i, gk in enumerate(group_keys) if gk in proper_train_gks]
    X_train = [rows[i] for i in t_idxs]
    y_train = []
    for i in t_idxs:
        gk = group_keys[i]
        cand = rows[i]["cand_algo"]
        y_train.append(actual_aucs_dict[gk][DEFAULT_ALGO] - actual_aucs_dict[gk][cand])
    df_train = pd.DataFrame(X_train).fillna(0.0)
    cat_features = [
        c
        for c in df_train.columns
        if df_train[c].dtype == object or isinstance(df_train[c].iloc[0], str)
    ]
    for c in cat_features:
        df_train[c] = df_train[c].astype(str)
    reg = CatBoostRegressor(
        iterations=500,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=10.0,
        verbose=False,
        random_state=42,
        thread_count=1,
    )
    reg.fit(df_train, y_train, cat_features=cat_features)

    calib_residuals = []
    c_idxs = [i for i, gk in enumerate(group_keys) if gk in calib_gks]
    for i in c_idxs:
        gk = group_keys[i]
        cand = rows[i]["cand_algo"]
        true_gain = actual_aucs_dict[gk][DEFAULT_ALGO] - actual_aucs_dict[gk][cand]
        df_calib = pd.DataFrame([rows[i]]).reindex(columns=df_train.columns).fillna(0.0)
        for c in cat_features:
            df_calib[c] = df_calib[c].astype(str)
        pred_gain = reg.predict(df_calib)[0]
        calib_residuals.append(pred_gain - true_gain)

    quantiles = {}
    n_calib = len(calib_residuals)
    for alpha in ALPHAS:
        if n_calib == 0:
            quantiles[alpha] = 0.0
            continue
        q_level = min(1.0, (n_calib + 1.0) * (1.0 - alpha) / n_calib)
        quantiles[alpha] = float(np.quantile(calib_residuals, q_level))

    for gk in test_gks:
        def_auc = actual_aucs_dict[gk][DEFAULT_ALGO]
        cands = [rows[i]["cand_algo"] for i, key in enumerate(group_keys) if key == gk]
        preds = {}
        for cand in cands:
            idx = [i for i, key in enumerate(group_keys) if key == gk and rows[i]["cand_algo"] == cand][0]
            df_test = pd.DataFrame([rows[idx]]).reindex(columns=df_train.columns).fillna(0.0)
            for c in cat_features:
                df_test[c] = df_test[c].astype(str)
            preds[cand] = reg.predict(df_test)[0]
        best_cand = max(preds, key=preds.get)
        best_pred_gain = preds[best_cand]
        true_gain_if_switch = def_auc - actual_aucs_dict[gk][best_cand]

        for alpha in ALPHAS:
            lower_bound = best_pred_gain - quantiles[alpha]
            switched = lower_bound > 0.0
            gain = true_gain_if_switch if switched else 0.0
            slot = stats[str(alpha)]
            slot["n"] += 1
            slot["gains"].append(gain)
            if switched:
                slot["switched"] += 1
                if gain < 0.0:
                    slot["harmful"] += 1


summary = {}
print("")
for alpha in ALPHAS:
    slot = stats[str(alpha)]
    n = max(1, slot["n"])
    switch_mass = slot["switched"] / float(n)
    harmful_mass = slot["harmful"] / float(n)
    cond_harm = slot["harmful"] / float(max(1, slot["switched"]))
    mean_gain = float(np.mean(slot["gains"])) if slot["gains"] else 0.0
    cvar10 = compute_cvar(slot["gains"], 0.1)
    summary[str(alpha)] = {
        "n": int(slot["n"]),
        "switch_mass": switch_mass,
        "harmful_switch_mass": harmful_mass,
        "harmful_switch_rate_given_switch": cond_harm,
        "mean_gain": mean_gain,
        "cvar_10": cvar10,
        "alpha_ratio": harmful_mass / alpha if alpha > 0 else math.inf,
    }
    print(f"[alpha={alpha:.2f}]")
    print(f"  - Switch Mass:                {switch_mass * 100:.2f}%")
    print(f"  - Harmful-Switch Mass:        {harmful_mass * 100:.2f}%")
    print(f"  - Harmful Rate | Switched:    {cond_harm * 100:.2f}%")
    print(f"  - Mean Gain:                  {mean_gain:+.3f}")
    print(f"  - CVaR_10:                    {cvar10:+.3f}")
    print(f"  - Harmful-Mass / alpha:       {summary[str(alpha)]['alpha_ratio']:.3f}")

out_dir = "/home/ycl/AICO-Intellig/results/theory_validation_latest"
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "harmful_switch_calibration.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
    f.write("\n")
