import json
import os
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath("/home/ycl/AICO-Intellig"))

from scripts.paper_experiments.theory_validation_common import (
    DEFAULT_ALGO,
    compute_gain_metrics,
    group_truth_labels,
    load_group_level_trajectory_features,
    load_pairwise_dataset,
    load_static_feature_dict,
)


print("=" * 80)
print("THEORY VALIDATION E1: UTILITY MISALIGNMENT UNDER LOFO")
print("=" * 80)


static_feature_dict = load_static_feature_dict()
traj_feature_dict, actual_aucs_group, family_by_group = load_group_level_trajectory_features()
truth_by_group = group_truth_labels(actual_aucs_group)
rows, _, group_keys, families, actual_aucs_pair, _ = load_pairwise_dataset()

valid_groups = sorted(
    set(truth_by_group)
    & set(static_feature_dict)
    & set(traj_feature_dict)
    & set(actual_aucs_pair)
)
unique_families = sorted({family_by_group[gk] for gk in valid_groups})

acc_static = []
acc_traj = []
acc_router = []

gains_static = []
gains_traj = []
gains_router = []

per_family_rows = []

for test_fam in unique_families:
    train_gks = [gk for gk in valid_groups if family_by_group[gk] != test_fam]
    test_gks = [gk for gk in valid_groups if family_by_group[gk] == test_fam]
    if not train_gks or not test_gks:
        continue

    # Static Top-1 classifier
    X_stat_train = [static_feature_dict[gk] for gk in train_gks]
    y_train = [truth_by_group[gk] for gk in train_gks]
    df_stat_train = pd.DataFrame(X_stat_train).fillna(0.0)
    for col in df_stat_train.select_dtypes(include=["object"]).columns:
        df_stat_train[col] = df_stat_train[col].astype("category").cat.codes
    clf_static = CatBoostClassifier(
        iterations=200,
        depth=6,
        verbose=False,
        random_state=42,
        thread_count=1,
    )
    clf_static.fit(df_stat_train.values, y_train)

    # Trajectory Top-1 classifier
    X_traj_train = [traj_feature_dict[gk] for gk in train_gks]
    df_traj_train = pd.DataFrame(X_traj_train).fillna(0.0)
    for col in df_traj_train.select_dtypes(include=["object"]).columns:
        df_traj_train[col] = df_traj_train[col].astype("category").cat.codes
    clf_traj = CatBoostClassifier(
        iterations=200,
        depth=6,
        verbose=False,
        random_state=42,
        thread_count=1,
    )
    clf_traj.fit(df_traj_train.values, y_train)

    # Pairwise gain router
    train_idxs = [i for i, gk in enumerate(group_keys) if gk in train_gks]
    X_pair_train = [rows[i] for i in train_idxs]
    y_pair_train = []
    for i in train_idxs:
        gk = group_keys[i]
        cand = rows[i]["cand_algo"]
        y_pair_train.append(actual_aucs_pair[gk][DEFAULT_ALGO] - actual_aucs_pair[gk][cand])
    df_pair_train = pd.DataFrame(X_pair_train).fillna(0.0)
    cat_features = [
        c
        for c in df_pair_train.columns
        if df_pair_train[c].dtype == object or isinstance(df_pair_train[c].iloc[0], str)
    ]
    for c in cat_features:
        df_pair_train[c] = df_pair_train[c].astype(str)
    reg_router = CatBoostRegressor(
        iterations=600,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=10.0,
        verbose=False,
        random_state=42,
        thread_count=1,
    )
    reg_router.fit(df_pair_train, y_pair_train, cat_features=cat_features)

    for gk in test_gks:
        truth = truth_by_group[gk]
        def_auc = actual_aucs_group[gk][DEFAULT_ALGO]

        df_stat_test = pd.DataFrame([static_feature_dict[gk]]).reindex(columns=df_stat_train.columns).fillna(0.0)
        for col in df_stat_test.select_dtypes(include=["object"]).columns:
            df_stat_test[col] = df_stat_test[col].astype("category").cat.codes
        pred_static = clf_static.predict(df_stat_test.values)[0][0]

        df_traj_test = pd.DataFrame([traj_feature_dict[gk]]).reindex(columns=df_traj_train.columns).fillna(0.0)
        for col in df_traj_test.select_dtypes(include=["object"]).columns:
            df_traj_test[col] = df_traj_test[col].astype("category").cat.codes
        pred_traj = clf_traj.predict(df_traj_test.values)[0][0]

        cands = [rows[i]["cand_algo"] for i, key in enumerate(group_keys) if key == gk]
        preds = {}
        for cand in cands:
            idx = [i for i, key in enumerate(group_keys) if key == gk and rows[i]["cand_algo"] == cand][0]
            df_p = pd.DataFrame([rows[idx]]).reindex(columns=df_pair_train.columns).fillna(0.0)
            for c in cat_features:
                df_p[c] = df_p[c].astype(str)
            preds[cand] = reg_router.predict(df_p)[0]
        best_cand = max(preds, key=preds.get)
        pred_router = best_cand if preds[best_cand] > 0.1 else DEFAULT_ALGO

        acc_static.append(1.0 if pred_static == truth else 0.0)
        acc_traj.append(1.0 if pred_traj == truth else 0.0)
        acc_router.append(1.0 if pred_router == truth else 0.0)

        gain_static = def_auc - actual_aucs_group[gk].get(pred_static, def_auc)
        gain_traj = def_auc - actual_aucs_group[gk].get(pred_traj, def_auc)
        gain_router = def_auc - actual_aucs_group[gk].get(pred_router, def_auc)

        gains_static.append(gain_static)
        gains_traj.append(gain_traj)
        gains_router.append(gain_router)

    per_family_rows.append(
        {
            "family": test_fam,
            "n_test": len(test_gks),
        }
    )


def summarize(name, acc_list, gain_list):
    m = compute_gain_metrics(gain_list)
    print(f"\n[{name}]")
    print(f"  - Top-1 Accuracy: {np.mean(acc_list) * 100:.1f}%")
    print(f"  - Mean Gain:      {m['mean_gain']:+.3f}")
    print(f"  - Loss Rate:      {m['loss_rate'] * 100:.1f}%")
    print(f"  - Switch Rate:    {m['switch_rate'] * 100:.1f}%")
    print(f"  - CVaR_10:        {m['cvar_10']:+.3f}")
    return {
        "accuracy": float(np.mean(acc_list)),
        **m,
    }


summary = {
    "static_top1": summarize("1. Static Top-1", acc_static, gains_static),
    "trajectory_top1": summarize("2. Trajectory Top-1", acc_traj, gains_traj),
    "pairwise_router": summarize("3. Pairwise Gain Router", acc_router, gains_router),
}

out_dir = "/home/ycl/AICO-Intellig/results/theory_validation_latest"
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "utility_misalignment.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
    f.write("\n")
