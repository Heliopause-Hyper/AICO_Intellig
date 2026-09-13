import os, json, math, numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.abspath("/home/ycl/AICO-Intellig"))
from scripts.train_algo_ranker import _features_from_instance, _drop_family_like_features
from scripts.train_curve_policy import _build_pairwise_dataset
from catboost import CatBoostRegressor

print("="*80)
print("FINAL ABLATIONS: FAIRNESS, CONFORMAL GATING & ALPHA SENSITIVITY")
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
    out_dirs=OUT_DIRS, default_algo="SciPy-Nelder-Mead",
    candidate_pool={"Nevergrad-NGOpt", "SciPy-COBYLA", "SciPy-DE"}, decision_eval=20
)

static_features_dict = {}
with open("/home/ycl/AICO-Intellig/results/static_ranker_baseline_latest/dataset.jsonl", "r") as f:
    for line in f:
        d = json.loads(line)
        gk = f"{d.get('instance_id')}__seed{d.get('seed', 0)}"
        static_features_dict[gk] = _drop_family_like_features(_features_from_instance(d, d))

unique_groups = sorted(set(group_keys))
unique_families = sorted(set(families))

methods = [
    "1. Static_NoGate", "2. Static_PointGate", "3. Static_ConfGate(a=0.1)",
    "4. Traj_NoGate", "5. Traj_PointGate", 
    "6. Traj_ConfGate(a=0.3)", "7. Traj_ConfGate(a=0.2)", "8. Traj_ConfGate(a=0.1)", "9. Traj_ConfGate(a=0.05)"
]

results = {m: [] for m in methods}

for test_fam in unique_families:
    train_gks = [gk for gk in unique_groups if family_by_group[gk] != test_fam and gk in static_features_dict and gk in actual_aucs_dict]
    test_gks = [gk for gk in unique_groups if family_by_group[gk] == test_fam and gk in static_features_dict and gk in actual_aucs_dict]
    if not train_gks or not test_gks: continue
    
    # Split for Conformal Calibration (80/20)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    proper_train_idx, calib_idx = next(kf.split(train_gks))
    proper_train_gks = [train_gks[i] for i in proper_train_idx]
    calib_gks = [train_gks[i] for i in calib_idx]
    
    # --- Prepare Data ---
    def prep_data(gks):
        X_s, X_t, y = [], [], []
        for i, gk in enumerate(group_keys):
            if gk in gks:
                cand = rows[i]["cand_algo"]
                sf = dict(static_features_dict[gk])
                sf["cand_algo"] = cand
                X_s.append(sf)
                X_t.append(rows[i])
                y.append(actual_aucs_dict[gk]["SciPy-Nelder-Mead"] - actual_aucs_dict[gk][cand])
        df_s = pd.DataFrame(X_s).fillna(0.0)
        df_t = pd.DataFrame(X_t).fillna(0.0)
        
        # Enforce column alignment by sorting columns alphabetically
        df_s = df_s.reindex(sorted(df_s.columns), axis=1)
        df_t = df_t.reindex(sorted(df_t.columns), axis=1)
        
        cs = [c for c in df_s.columns if df_s[c].dtype == object or isinstance(df_s[c].iloc[0], str)]
        ct = [c for c in df_t.columns if df_t[c].dtype == object or isinstance(df_t[c].iloc[0], str)]
        for c in cs: df_s[c] = df_s[c].astype(str)
        for c in ct: df_t[c] = df_t[c].astype(str)
        return df_s, df_t, y, cs, ct
        
    df_s_train, df_t_train, y_train, cat_s, cat_t = prep_data(proper_train_gks)
    df_s_calib, df_t_calib, y_calib, _, _ = prep_data(calib_gks)
    
    # --- Train Regressors ---
    reg_stat = CatBoostRegressor(iterations=400, depth=4, l2_leaf_reg=5.0, verbose=False, random_state=42, thread_count=1)
    reg_stat.fit(df_s_train, y_train, cat_features=cat_s)
    
    reg_traj = CatBoostRegressor(iterations=400, depth=4, l2_leaf_reg=5.0, verbose=False, random_state=42, thread_count=1)
    reg_traj.fit(df_t_train, y_train, cat_features=cat_t)
    
    # --- Calibration Phase ---
    # Reindex calib to match train columns exactly
    df_s_calib = df_s_calib.reindex(columns=df_s_train.columns).fillna(0.0)
    df_t_calib = df_t_calib.reindex(columns=df_t_train.columns).fillna(0.0)
    for c in cat_s: df_s_calib[c] = df_s_calib[c].astype(str)
    for c in cat_t: df_t_calib[c] = df_t_calib[c].astype(str)

    pred_s_calib = reg_stat.predict(df_s_calib)
    pred_t_calib = reg_traj.predict(df_t_calib)
    
    # Residuals: \hat{G} - G_true
    res_s = pred_s_calib - y_calib
    res_t = pred_t_calib - y_calib
    
    def get_q(residuals, alpha):
        n = len(residuals)
        q_level = min(1.0, (n + 1.0) * (1 - alpha) / n)
        return np.quantile(residuals, q_level) if len(residuals) > 0 else 0.0
        
    q_s_01 = get_q(res_s, 0.1)
    q_t_03 = get_q(res_t, 0.3)
    q_t_02 = get_q(res_t, 0.2)
    q_t_01 = get_q(res_t, 0.1)
    q_t_005 = get_q(res_t, 0.05)
    
    # --- Evaluation ---
    for gk in test_gks:
        def_auc = actual_aucs_dict[gk]["SciPy-Nelder-Mead"]
        
        cands = [rows[i]["cand_algo"] for i, k in enumerate(group_keys) if k == gk]
        p_s, p_t = {}, {}
        for cand in cands:
            idx = [i for i, k in enumerate(group_keys) if k == gk and rows[i]["cand_algo"] == cand][0]
            sf = dict(static_features_dict[gk]); sf["cand_algo"] = cand
            df_s_test = pd.DataFrame([sf]).reindex(columns=df_s_train.columns).fillna(0.0)
            df_t_test = pd.DataFrame([rows[idx]]).reindex(columns=df_t_train.columns).fillna(0.0)
            for c in cat_s: df_s_test[c] = df_s_test[c].astype(str)
            for c in cat_t: df_t_test[c] = df_t_test[c].astype(str)
            
            p_s[cand] = reg_stat.predict(df_s_test)[0]
            p_t[cand] = reg_traj.predict(df_t_test)[0]
            
        best_c_s = max(p_s, key=p_s.get)
        best_c_t = max(p_t, key=p_t.get)
        
        # Decisions
        decs = {
            "1. Static_NoGate": best_c_s,
            "2. Static_PointGate": best_c_s if p_s[best_c_s] > 0 else "SciPy-Nelder-Mead",
            "3. Static_ConfGate(a=0.1)": best_c_s if p_s[best_c_s] - q_s_01 > 0 else "SciPy-Nelder-Mead",
            "4. Traj_NoGate": best_c_t,
            "5. Traj_PointGate": best_c_t if p_t[best_c_t] > 0 else "SciPy-Nelder-Mead",
            "6. Traj_ConfGate(a=0.3)": best_c_t if p_t[best_c_t] - q_t_03 > 0 else "SciPy-Nelder-Mead",
            "7. Traj_ConfGate(a=0.2)": best_c_t if p_t[best_c_t] - q_t_02 > 0 else "SciPy-Nelder-Mead",
            "8. Traj_ConfGate(a=0.1)": best_c_t if p_t[best_c_t] - q_t_01 > 0 else "SciPy-Nelder-Mead",
            "9. Traj_ConfGate(a=0.05)": best_c_t if p_t[best_c_t] - q_t_005 > 0 else "SciPy-Nelder-Mead"
        }
        
        for m, dec in decs.items():
            results[m].append(def_auc - actual_aucs_dict[gk].get(dec, def_auc))

def calc_cvar(gains, alpha=0.1):
    gains = np.array(gains)
    threshold = np.percentile(gains, alpha * 100)
    tail_losses = gains[gains <= threshold]
    return np.mean(tail_losses) if len(tail_losses) > 0 else 0.0

print(f"{'Method':<26} | {'Mean Gain':>10} | {'Win Rate':>9} | {'Loss Rate':>9} | {'Switch %':>9} | {'CVaR_10':>8}")
print("-" * 85)
for m in methods:
    g = np.array(results[m])
    mean_g = np.mean(g)
    win_r = np.mean(g > 0)
    loss_r = np.mean(g < 0)
    sw_r = np.mean(g != 0)
    cvar = calc_cvar(g, 0.1)
    print(f"{m:<26} | {mean_g:>+10.3f} | {win_r*100:>8.1f}% | {loss_r*100:>8.1f}% | {sw_r*100:>8.1f}% | {cvar:>+8.3f}")
