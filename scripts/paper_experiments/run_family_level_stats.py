import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# Load the previously generated unified data
# Since we didn't save the raw instance-level data to a file in the last script, 
# I will quickly re-run the evaluation logic just to get the family-level aggregation.

import os
import sys
sys.path.append(os.path.abspath("/home/ycl/AICO-Intellig"))
from scripts.train_algo_ranker import _features_from_instance, _drop_family_like_features
from scripts.train_curve_policy import _build_pairwise_dataset
from catboost import CatBoostClassifier, CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

OUT_DIRS = [
    "/home/ycl/AICO-Intellig/results/ex_advection2d_v1_full_200_m100_t3600_r1_20260709_231148",
    "/home/ycl/AICO-Intellig/results/ex_blackscholes2d_v1_full_200_m100_t3600_r1_20260709_104116",
    "/home/ycl/AICO-Intellig/results/ex_heat_time_v1_full_200_m100_t3600_r1_",
    "/home/ycl/AICO-Intellig/results/heat_v1_full_180_m30_t90_r3",
    "/home/ycl/AICO-Intellig/results/iaea_v1_full_180_m30_t90_r3_20260708_102705",
    "/home/ycl/AICO-Intellig/results/sns_v1_full_150_m40_t90_r4_20260429_234621",
    "/home/ycl/AICO-Intellig/results/sns_v3_full_240_m30_t90_r3_more1",
    "/home/ycl/AICO-Intellig/results/thmf_v1_full_180_m30_t90_r3_20260708_102705"
]

rows, y_auc_gain, group_keys, families, actual_aucs_dict, family_by_group = _build_pairwise_dataset(
    out_dirs=OUT_DIRS,
    default_algo="SciPy-Nelder-Mead",
    candidate_pool={"Nevergrad-NGOpt", "SciPy-COBYLA", "SciPy-DE"},
    decision_eval=20
)

static_features_dict = {}
with open("/home/ycl/AICO-Intellig/results/static_ranker_baseline_latest/dataset.jsonl", "r") as f:
    for line in f:
        d = json.loads(line)
        iid = str(d.get("instance_id"))
        seed = int(d.get("seed", 0))
        gk = f"{iid}__seed{seed}"
        f_stat = _features_from_instance(d, d)
        f_stat = _drop_family_like_features(f_stat)
        static_features_dict[gk] = f_stat

unique_groups = sorted(set(group_keys))
unique_families = sorted(set(families))

family_results = {}

for test_fam in unique_families:
    train_gks = [gk for gk in unique_groups if family_by_group[gk] != test_fam]
    test_gks = [gk for gk in unique_groups if family_by_group[gk] == test_fam]
    if not train_gks or not test_gks: continue
    
    # Static Train
    X_stat_train, y_stat_train = [], []
    for gk in train_gks:
        if gk in static_features_dict and gk in actual_aucs_dict:
            best_algo = min(actual_aucs_dict[gk], key=actual_aucs_dict[gk].get)
            X_stat_train.append(static_features_dict[gk])
            y_stat_train.append(best_algo)
            
    df_stat_train = pd.DataFrame(X_stat_train).fillna(0.0)
    for col in df_stat_train.select_dtypes(include=['object']).columns:
        df_stat_train[col] = df_stat_train[col].astype('category').cat.codes
    clf_static = CatBoostClassifier(iterations=200, depth=6, verbose=False, random_state=42, thread_count=1)
    clf_static.fit(df_stat_train.values, y_stat_train)
    
    from collections import Counter
    maj_label = Counter(y_stat_train).most_common(1)[0][0]
    
    # Ours Train
    train_idxs = [i for i, gk in enumerate(group_keys) if gk in train_gks]
    X_pair_train = [rows[i] for i in train_idxs]
    y_pair_train = []
    for i in train_idxs:
        gk = group_keys[i]
        cand = rows[i]["cand_algo"]
        y_pair_train.append(actual_aucs_dict[gk]["SciPy-Nelder-Mead"] - actual_aucs_dict[gk][cand])
        
    df_pair_train = pd.DataFrame(X_pair_train).fillna(0.0)
    cat_features = [c for c in df_pair_train.columns if df_pair_train[c].dtype == object or isinstance(df_pair_train[c].iloc[0], str)]
    for c in cat_features: df_pair_train[c] = df_pair_train[c].astype(str)
    
    reg_ours = CatBoostRegressor(
        iterations=600,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=10.0,
        verbose=False,
        random_state=42,
        thread_count=1,
    )
    reg_ours.fit(df_pair_train, y_pair_train, cat_features=cat_features)
    
    fam_gains_static = []
    fam_gains_maj = []
    fam_gains_ours = []
    
    # Evaluate
    for gk in test_gks:
        if gk not in static_features_dict or gk not in actual_aucs_dict: continue
        def_auc = actual_aucs_dict[gk]["SciPy-Nelder-Mead"]
        
        # 1. Static
        df_stat_test = pd.DataFrame([static_features_dict[gk]]).reindex(columns=df_stat_train.columns).fillna(0.0)
        for col in df_stat_test.select_dtypes(include=['object']).columns:
            df_stat_test[col] = df_stat_test[col].astype('category').cat.codes
        try: pred_static_algo = clf_static.predict(df_stat_test.values)[0][0]
        except: pred_static_algo = maj_label
        
        # 2. Majority
        pred_maj_algo = maj_label
        
        # 3. Ours
        cands = [rows[i]["cand_algo"] for i, k in enumerate(group_keys) if k == gk]
        preds = {}
        for cand in cands:
            idx = [i for i, k in enumerate(group_keys) if k == gk and rows[i]["cand_algo"] == cand][0]
            df_p = pd.DataFrame([rows[idx]]).reindex(columns=df_pair_train.columns).fillna(0.0)
            for c in cat_features: df_p[c] = df_p[c].astype(str)
            preds[cand] = reg_ours.predict(df_p)[0]
            
        best_cand = max(preds, key=preds.get)
        pred_our_algo = best_cand if preds[best_cand] > 0.1 else "SciPy-Nelder-Mead"
        
        fam_gains_static.append(def_auc - actual_aucs_dict[gk].get(pred_static_algo, def_auc))
        fam_gains_maj.append(def_auc - actual_aucs_dict[gk].get(pred_maj_algo, def_auc))
        fam_gains_ours.append(def_auc - actual_aucs_dict[gk].get(pred_our_algo, def_auc))
        
    family_results[test_fam] = {
        "n_instances": len(fam_gains_ours),
        "mean_static": np.mean(fam_gains_static),
        "mean_maj": np.mean(fam_gains_maj),
        "mean_ours": np.mean(fam_gains_ours)
    }

# Filter out empty families
valid_families = {fam: res for fam, res in family_results.items() if res['n_instances'] > 0}

# Compute Macro-Averages across Families
macro_static = np.mean([v["mean_static"] for v in valid_families.values()])
macro_maj = np.mean([v["mean_maj"] for v in valid_families.values()])
macro_ours = np.mean([v["mean_ours"] for v in valid_families.values()])

print("Family-Level Aggregation (Macro-Averages):")
for fam, res in valid_families.items():
    print(f"  {fam:<30} (N={res['n_instances']:<3}) | Maj: {res['mean_maj']:+.3f} | Stat: {res['mean_static']:+.3f} | Ours: {res['mean_ours']:+.3f}")

print("-" * 80)
print(f"Macro-Avg Gain (Family Level) | Maj: {macro_maj:+.3f} | Stat: {macro_static:+.3f} | Ours: {macro_ours:+.3f}")

# Family-Level Wilcoxon Test
gains_fam_ours = [v["mean_ours"] for v in valid_families.values()]
gains_fam_stat = [v["mean_static"] for v in valid_families.values()]
gains_fam_maj = [v["mean_maj"] for v in valid_families.values()]

# We use Wilcoxon on the valid family-level means
try:
    # Since n=6 is very small, Wilcoxon test might lack statistical power
    # But let's run a two-sided test just to see
    w_fam_stat, p_fam_stat = wilcoxon(gains_fam_ours, gains_fam_stat, alternative='two-sided')
    w_fam_maj, p_fam_maj = wilcoxon(gains_fam_ours, gains_fam_maj, alternative='two-sided')
    print(f"\nFamily-Level Wilcoxon p-value (Ours vs Static): {p_fam_stat:.3f}")
    print(f"Family-Level Wilcoxon p-value (Ours vs Maj):    {p_fam_maj:.3f}")
    
    # Also print the variance to show robustness
    print(f"\nRobustness Check (Variance of Mean Gains across Families):")
    print(f"  Maj Variance:  {np.var(gains_fam_maj):.3f}")
    print(f"  Stat Variance: {np.var(gains_fam_stat):.3f}")
    print(f"  Ours Variance: {np.var(gains_fam_ours):.3f}")
except Exception as e:
    print("\nWilcoxon test warning:", e)
