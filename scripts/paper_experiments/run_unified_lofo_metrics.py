import os
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath("/home/ycl/AICO-Intellig"))
from scripts.train_algo_ranker import _features_from_instance, _drop_family_like_features
from scripts.train_curve_policy import _build_pairwise_dataset
from catboost import CatBoostClassifier, CatBoostRegressor

print("="*80)
print("APPLES-TO-APPLES UNIFIED METRICS & STATISTICAL SIGNIFICANCE (LOFO)")
print("="*80)

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

results_static = []
results_majority = []
results_ours = []

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
    # To predict absolute Gain: we want model to predict (AUC_default - AUC_candidate)
    y_pair_train = []
    for i in train_idxs:
        gk = group_keys[i]
        cand = rows[i]["cand_algo"]
        def_auc = actual_aucs_dict[gk]["SciPy-Nelder-Mead"]
        cand_auc = actual_aucs_dict[gk][cand]
        y_pair_train.append(def_auc - cand_auc) # Positive means candidate is better
        
    df_pair_train = pd.DataFrame(X_pair_train).fillna(0.0)
    cat_features = [c for c in df_pair_train.columns if df_pair_train[c].dtype == object or isinstance(df_pair_train[c].iloc[0], str)]
    for c in cat_features: df_pair_train[c] = df_pair_train[c].astype(str)
    
    # We want a model that predicts the raw gain robustly
    # For CatBoostRegressor on small datasets, lowering depth and increasing L2 can prevent overfitting
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
        
        # 3. Ours Prediction (using the exactly same logic as the main script)
        cands = [rows[i]["cand_algo"] for i, k in enumerate(group_keys) if k == gk]
        preds = {}
        for cand in cands:
            idx = [i for i, k in enumerate(group_keys) if k == gk and rows[i]["cand_algo"] == cand][0]
            df_p = pd.DataFrame([rows[idx]]).reindex(columns=df_pair_train.columns).fillna(0.0)
            for c in cat_features: df_p[c] = df_p[c].astype(str)
            preds[cand] = reg_ours.predict(df_p)[0]
            
        best_cand = max(preds, key=preds.get)
        # We only switch if the predicted expected gain is strictly positive (e.g. > 0.1 to be safe against noise)
        pred_our_algo = best_cand if preds[best_cand] > 0.1 else "SciPy-Nelder-Mead"
        
        # Gains: def_auc - pred_auc
        results_static.append(def_auc - actual_aucs_dict[gk].get(pred_static_algo, def_auc))
        results_majority.append(def_auc - actual_aucs_dict[gk].get(pred_maj_algo, def_auc))
        results_ours.append(def_auc - actual_aucs_dict[gk].get(pred_our_algo, def_auc))

def compute_metrics(gains, name):
    gains = np.array(gains)
    uncond_mean = np.mean(gains)
    win_rate = np.mean(gains > 0)
    loss_rate = np.mean(gains < 0)
    triggered = gains != 0
    switch_rate = np.mean(triggered)
    cond_mean = np.mean(gains[triggered]) if np.sum(triggered) > 0 else 0.0
    
    print(f"[{name}]")
    print(f"  - Win Rate (Strictly Positive Gain): {win_rate*100:.1f}%")
    print(f"  - Loss Rate (Strictly Negative Gain): {loss_rate*100:.1f}%")
    print(f"  - Switch/Trigger Rate:               {switch_rate*100:.1f}%")
    print(f"  - Unconditional Mean Log-AUC Gain:   +{uncond_mean:.3f}")
    print(f"  - Conditional Mean Log-AUC Gain:     +{cond_mean:.3f}")
    return gains

print("\n--- UNIFIED ENGINEERING LOFO RESULTS ---")
g_maj = compute_metrics(results_majority, "1. Majority Baseline")
g_stat = compute_metrics(results_static, "2. Static ELA (Top-1 Classification)")
g_ours = compute_metrics(results_ours, "3. Ours (Risk-Aware Pairwise Gated)")

print("\n--- STATISTICAL SIGNIFICANCE (Wilcoxon Signed-Rank Test) ---")
w_stat_maj, p_maj = wilcoxon(g_ours, g_maj, alternative='two-sided')
w_stat_stat, p_stat = wilcoxon(g_ours, g_stat, alternative='two-sided')

print(f"Ours vs Majority:   p-value = {p_maj:.2e} {'(Significant!)' if p_maj < 0.05 else ''}")
print(f"Ours vs Static ELA: p-value = {p_stat:.2e} {'(Significant!)' if p_stat < 0.05 else ''}")

n_boot = 1000
boot_means = [np.mean(np.random.choice(g_ours, size=len(g_ours), replace=True)) for _ in range(n_boot)]
ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])
print(f"\nOurs 95% CI for Unconditional Mean Gain: [+{ci_lower:.3f}, +{ci_upper:.3f}]")
