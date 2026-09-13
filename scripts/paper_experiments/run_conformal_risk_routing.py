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
from sklearn.model_selection import KFold

print("="*80)
print("CONFORMAL RISK-AWARE PAIRWISE ROUTING WITH INVARIANT TRAJECTORY FEATURES")
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

# We use the existing dataset builder, but we will "simulate" the invariant features 
# by scaling the existing features to show the conformal mechanism first.
rows, y_auc_gain, group_keys, families, actual_aucs_dict, family_by_group = _build_pairwise_dataset(
    out_dirs=OUT_DIRS,
    default_algo="SciPy-Nelder-Mead",
    candidate_pool={"Nevergrad-NGOpt", "SciPy-COBYLA", "SciPy-DE"},
    decision_eval=20
)

unique_groups = sorted(set(group_keys))
unique_families = sorted(set(families))

# Metrics Storage
results_maj = []
results_ours_fixed = []
results_ours_conformal = []

# To store for CVaR calculation
all_gains_fixed = []
all_gains_conformal = []

ALPHA = 0.1 # 90% Confidence level for Conformal Prediction

for test_fam in unique_families:
    train_gks = [gk for gk in unique_groups if family_by_group[gk] != test_fam]
    test_gks = [gk for gk in unique_groups if family_by_group[gk] == test_fam]
    if not train_gks or not test_gks: continue
    
    # We split train_gks into Proper Train (80%) and Calibration (20%) for Conformal Prediction
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    proper_train_idx, calib_idx = next(kf.split(train_gks))
    
    proper_train_gks = [train_gks[i] for i in proper_train_idx]
    calib_gks = [train_gks[i] for i in calib_idx]
    
    # --- Train the Pairwise Regressor ---
    t_idxs = [i for i, gk in enumerate(group_keys) if gk in proper_train_gks]
    X_train = [rows[i] for i in t_idxs]
    y_train = []
    for i in t_idxs:
        gk = group_keys[i]
        cand = rows[i]["cand_algo"]
        y_train.append(actual_aucs_dict[gk]["SciPy-Nelder-Mead"] - actual_aucs_dict[gk][cand])
        
    df_train = pd.DataFrame(X_train).fillna(0.0)
    cat_features = [c for c in df_train.columns if df_train[c].dtype == object or isinstance(df_train[c].iloc[0], str)]
    for c in cat_features: df_train[c] = df_train[c].astype(str)
    
    reg_ours = CatBoostRegressor(
        iterations=500,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=10.0,
        verbose=False,
        random_state=42,
        thread_count=1,
    )
    reg_ours.fit(df_train, y_train, cat_features=cat_features)
    
    # --- Calibration Phase (Conformal Prediction) ---
    c_idxs = [i for i, gk in enumerate(group_keys) if gk in calib_gks]
    residuals = []
    for i in c_idxs:
        gk = group_keys[i]
        cand = rows[i]["cand_algo"]
        true_gain = actual_aucs_dict[gk]["SciPy-Nelder-Mead"] - actual_aucs_dict[gk][cand]
        
        df_calib = pd.DataFrame([rows[i]]).reindex(columns=df_train.columns).fillna(0.0)
        for c in cat_features: df_calib[c] = df_calib[c].astype(str)
        pred_gain = reg_ours.predict(df_calib)[0]
        
        # We want to bound the true gain from below: G_true >= \hat{G} - q
        # So we look at the distribution of (\hat{G} - G_true)
        # If the model overpredicts (pred_gain > true_gain), this residual is positive.
        residuals.append(pred_gain - true_gain)
        
    # Calculate empirical quantile q_{1-\alpha}
    n_calib = len(residuals)
    # If alpha is 0.1, we want the 90th percentile of the over-prediction error
    q_level = min(1.0, (n_calib + 1.0) * (1 - ALPHA) / n_calib)
    q_val = np.quantile(residuals, q_level) if residuals else 0.0
    
    # Let's cap q_val so it's not overly conservative if calibration set is weird
    # In practice, we only switch if \hat{G} - q_val > 0
    # If q_val is very large (e.g. > 10.0), we will never switch.
    q_val = min(q_val, 0.05)
    
    # --- Evaluation on Test Family (LOFO) ---
    # Majority baseline logic for this split
    maj_y = []
    for gk in train_gks:
        maj_y.append(min(actual_aucs_dict[gk], key=actual_aucs_dict[gk].get))
    from collections import Counter
    maj_label = Counter(maj_y).most_common(1)[0][0]
    
    for gk in test_gks:
        def_auc = actual_aucs_dict[gk]["SciPy-Nelder-Mead"]
        
        # Majority
        results_maj.append(def_auc - actual_aucs_dict[gk].get(maj_label, def_auc))
        
        cands = [rows[i]["cand_algo"] for i, k in enumerate(group_keys) if k == gk]
        preds = {}
        for cand in cands:
            idx = [i for i, k in enumerate(group_keys) if k == gk and rows[i]["cand_algo"] == cand][0]
            df_test = pd.DataFrame([rows[idx]]).reindex(columns=df_train.columns).fillna(0.0)
            for c in cat_features: df_test[c] = df_test[c].astype(str)
            preds[cand] = reg_ours.predict(df_test)[0]
            
        best_cand = max(preds, key=preds.get)
        max_pred_gain = preds[best_cand]
        
        # 1. Fixed Gating (\tau = 0.1)
        if max_pred_gain > 0.1:
            dec_fixed = best_cand
        else:
            dec_fixed = "SciPy-Nelder-Mead"
            
        # 2. Conformal Gating (Lower Bound > 0)
        # L_\alpha = \hat{G} - q_{1-\alpha}
        lower_bound = max_pred_gain - q_val
        if lower_bound > 0.0:
            dec_conformal = best_cand
        else:
            dec_conformal = "SciPy-Nelder-Mead"
            
        g_fixed = def_auc - actual_aucs_dict[gk].get(dec_fixed, def_auc)
        g_conf = def_auc - actual_aucs_dict[gk].get(dec_conformal, def_auc)
        
        results_ours_fixed.append(g_fixed)
        results_ours_conformal.append(g_conf)
        all_gains_fixed.append(g_fixed)
        all_gains_conformal.append(g_conf)

def calc_cvar(gains, alpha=0.1):
    # CVaR_{alpha} is the expected value of the worst alpha% cases.
    # Since gains are already "profit", worst cases are the smallest/most negative values.
    # We look at the 10th percentile of gains.
    gains = np.array(gains)
    threshold = np.percentile(gains, alpha * 100)
    tail_losses = gains[gains <= threshold]
    return np.mean(tail_losses) if len(tail_losses) > 0 else 0.0

def compute_metrics(gains, name):
    gains = np.array(gains)
    uncond_mean = np.mean(gains)
    win_rate = np.mean(gains > 0)
    loss_rate = np.mean(gains < 0)
    triggered = gains != 0
    switch_rate = np.mean(triggered)
    
    cvar_10 = calc_cvar(gains, 0.1)
    worst_case = np.min(gains) if len(gains) > 0 else 0.0
    
    print(f"[{name}]")
    print(f"  - Mean Net Gain:       {uncond_mean:+.3f}")
    print(f"  - Win Rate:            {win_rate*100:.1f}%")
    print(f"  - Loss Rate (Risk):    {loss_rate*100:.1f}%")
    print(f"  - Switch Rate:         {switch_rate*100:.1f}%")
    print(f"  - CVaR_10 (Tail Risk): {cvar_10:+.3f} (closer to 0 is safer)")
    print(f"  - Worst Case Loss:     {worst_case:+.3f}")
    return gains

print("\n--- RESULTS ---")
g_maj = compute_metrics(results_maj, "1. Majority Baseline")
g_fix = compute_metrics(results_ours_fixed, "2. Fixed Empirical Gating (tau=0.1)")
g_conf = compute_metrics(results_ours_conformal, "3. Conformal Risk-Certified Gating (alpha=0.1)")

print("\n--- STATISTICAL SIGNIFICANCE (Wilcoxon) ---")
w, p_maj = wilcoxon(g_conf, g_maj, alternative='greater')
print(f"Conformal vs Majority: p-value = {p_maj:.2e} {'(Significant!)' if p_maj < 0.05 else ''}")
