import os, json, math, numpy as np, pandas as pd
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.abspath("/home/ycl/AICO-Intellig"))
from scripts.train_curve_policy import _build_pairwise_dataset
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

print("="*80)
print("TRAJECTORY LENGTH (K) ABLATION STUDY")
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

K_VALUES = [5, 10, 15, 20, 30]
ALPHA = 0.1

print(f"{'K Steps':<10} | {'Mean Gain':>12} | {'Loss Rate':>12} | {'Switch Rate':>12} | {'CVaR_10':>10}")
print("-" * 65)

for K in K_VALUES:
    # Build dataset for current K
    rows, y_auc_gain, group_keys, families, actual_aucs_dict, family_by_group = _build_pairwise_dataset(
        out_dirs=OUT_DIRS, default_algo="SciPy-Nelder-Mead",
        candidate_pool={"Nevergrad-NGOpt", "SciPy-COBYLA", "SciPy-DE"}, decision_eval=K
    )
    
    unique_groups = sorted(set(group_keys))
    unique_families = sorted(set(families))
    
    all_gains = []
    
    for test_fam in unique_families:
        train_gks = [gk for gk in unique_groups if family_by_group[gk] != test_fam and gk in actual_aucs_dict]
        test_gks = [gk for gk in unique_groups if family_by_group[gk] == test_fam and gk in actual_aucs_dict]
        if not train_gks or not test_gks: continue
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        proper_train_idx, calib_idx = next(kf.split(train_gks))
        proper_train_gks = [train_gks[i] for i in proper_train_idx]
        calib_gks = [train_gks[i] for i in calib_idx]
        
        # Train
        t_idxs = [i for i, gk in enumerate(group_keys) if gk in proper_train_gks]
        df_train = pd.DataFrame([rows[i] for i in t_idxs]).fillna(0.0)
        y_train = [actual_aucs_dict[group_keys[i]]["SciPy-Nelder-Mead"] - actual_aucs_dict[group_keys[i]][rows[i]["cand_algo"]] for i in t_idxs]
        
        cat_f = [c for c in df_train.columns if df_train[c].dtype == object or isinstance(df_train[c].iloc[0], str)]
        for c in cat_f: df_train[c] = df_train[c].astype(str)
        
        # Use a faster configuration for CatBoost just to get K ablation done quickly
        reg = CatBoostRegressor(iterations=100, depth=4, verbose=False, random_state=42, thread_count=1)
        reg.fit(df_train, y_train, cat_features=cat_f)
        
        # Calibrate
        c_idxs = [i for i, gk in enumerate(group_keys) if gk in calib_gks]
        residuals = []
        if c_idxs:
            df_c = pd.DataFrame([rows[i] for i in c_idxs]).reindex(columns=df_train.columns).fillna(0.0)
            for c in cat_f: df_c[c] = df_c[c].astype(str)
            preds_c = reg.predict(df_c)
            for j, i in enumerate(c_idxs):
                true_gain = actual_aucs_dict[group_keys[i]]["SciPy-Nelder-Mead"] - actual_aucs_dict[group_keys[i]][rows[i]["cand_algo"]]
                residuals.append(preds_c[j] - true_gain)
            
        n_c = len(residuals)
        q_level = min(1.0, (n_c + 1.0) * (1 - ALPHA) / n_c)
        q_val = np.quantile(residuals, q_level) if residuals else 0.0
        q_val = min(q_val, 0.05)
        
        # Evaluate
        if test_gks:
            df_t = pd.DataFrame([rows[i] for i in range(len(group_keys)) if group_keys[i] in test_gks]).reindex(columns=df_train.columns).fillna(0.0)
            for c in cat_f: df_t[c] = df_t[c].astype(str)
            preds_t = reg.predict(df_t)
            
            pred_dict = {gk: {} for gk in test_gks}
            idx_map = [i for i in range(len(group_keys)) if group_keys[i] in test_gks]
            for j, orig_i in enumerate(idx_map):
                gk = group_keys[orig_i]
                cand = rows[orig_i]["cand_algo"]
                pred_dict[gk][cand] = preds_t[j]
                
            for gk in test_gks:
                def_auc = actual_aucs_dict[gk]["SciPy-Nelder-Mead"]
                preds_for_gk = pred_dict[gk]
                best_cand = max(preds_for_gk, key=preds_for_gk.get)
                if preds_for_gk[best_cand] - q_val > 0.0:
                    dec = best_cand
                else:
                    dec = "SciPy-Nelder-Mead"
                all_gains.append(def_auc - actual_aucs_dict[gk].get(dec, def_auc))
            
    g = np.array(all_gains)
    if len(g) == 0: continue
    
    mean_g = np.mean(g)
    loss_r = np.mean(g < 0)
    sw_r = np.mean(g != 0)
    thresh = np.percentile(g, 10)
    cvar = np.mean(g[g <= thresh]) if len(g[g <= thresh]) > 0 else 0.0
    
    print(f"K={K:<8} | {mean_g:>+12.3f} | {loss_r*100:>11.1f}% | {sw_r*100:>11.1f}% | {cvar:>+10.3f}")
