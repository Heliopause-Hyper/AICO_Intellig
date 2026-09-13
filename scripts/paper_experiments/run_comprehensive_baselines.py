import os, json, math, numpy as np, pandas as pd
from collections import defaultdict
from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.abspath("/home/ycl/AICO-Intellig"))
from scripts.train_algo_ranker import _features_from_instance, _drop_family_like_features
from scripts.train_curve_policy import _build_pairwise_dataset
from catboost import CatBoostClassifier, CatBoostRegressor

print("="*80)
print("COMPREHENSIVE ORTHOGONAL ABLATION & RISK-AWARE GATING ANALYSIS")
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

print("Loading data...")
rows, y_auc_gain, group_keys, families, actual_aucs_dict, family_by_group = _build_pairwise_dataset(
    out_dirs=OUT_DIRS, default_algo="SciPy-Nelder-Mead",
    candidate_pool={"Nevergrad-NGOpt", "SciPy-COBYLA", "SciPy-DE"}, decision_eval=20
)

static_features_dict = {}
with open("/home/ycl/AICO-Intellig/results/static_ranker_baseline_latest/dataset.jsonl", "r") as f:
    for line in f:
        d = json.loads(line)
        gk = f"{d.get('instance_id')}__seed{d.get('seed', 0)}"
        f_stat = _drop_family_like_features(_features_from_instance(d, d))
        static_features_dict[gk] = f_stat

unique_groups = sorted(set(group_keys))
unique_families = sorted(set(families))
CANDS = ["Nevergrad-NGOpt", "SciPy-COBYLA", "SciPy-DE"]

methods = [
    "B1_Default", "B2_Random", "B3_Oracle", 
    "B4_StatTop1", "B5_TrajTop1", 
    "B6_StatGate", "B7_TrajNoGate", "Ours_TrajGate_0.1",
    "Ours_tau_0.0", "Ours_tau_0.05", "Ours_tau_0.2", "Ours_tau_0.5"
]

results = {m: {fam: {"gains": [], "accs": []} for fam in unique_families} for m in methods}

print("Running LOFO Cross-Validation (This may take a minute)...")

for test_fam in unique_families:
    train_gks = [gk for gk in unique_groups if family_by_group[gk] != test_fam and gk in static_features_dict and gk in actual_aucs_dict]
    test_gks = [gk for gk in unique_groups if family_by_group[gk] == test_fam and gk in static_features_dict and gk in actual_aucs_dict]
    if not train_gks or not test_gks: continue
    
    # 1. Classification Data
    X_stat_train_clf, X_traj_train_clf, y_train_clf = [], [], []
    for gk in train_gks:
        best_algo = min(actual_aucs_dict[gk], key=actual_aucs_dict[gk].get)
        X_stat_train_clf.append(static_features_dict[gk])
        
        idx = group_keys.index(gk)
        t_f = {k:v for k,v in rows[idx].items() if not k.startswith("rel_") and not k.startswith("diff_") and k != "cand_algo" and k != "eval_time_ratio"}
        X_traj_train_clf.append(t_f)
        y_train_clf.append(best_algo)
        
    df_stat_clf = pd.DataFrame(X_stat_train_clf).fillna(0.0)
    df_traj_clf = pd.DataFrame(X_traj_train_clf).fillna(0.0)
    cat_stat_clf = [c for c in df_stat_clf.columns if df_stat_clf[c].dtype == object or isinstance(df_stat_clf[c].iloc[0], str)]
    cat_traj_clf = [c for c in df_traj_clf.columns if df_traj_clf[c].dtype == object or isinstance(df_traj_clf[c].iloc[0], str)]
    for c in cat_stat_clf: df_stat_clf[c] = df_stat_clf[c].astype(str)
    for c in cat_traj_clf: df_traj_clf[c] = df_traj_clf[c].astype(str)
    
    clf_stat = CatBoostClassifier(iterations=200, depth=6, verbose=False, random_state=42, thread_count=1)
    if len(set(y_train_clf)) > 1: clf_stat.fit(df_stat_clf, y_train_clf, cat_features=cat_stat_clf)
    
    clf_traj = CatBoostClassifier(iterations=200, depth=6, verbose=False, random_state=42, thread_count=1)
    if len(set(y_train_clf)) > 1: clf_traj.fit(df_traj_clf, y_train_clf, cat_features=cat_traj_clf)
    
    # 2. Regression Data
    X_stat_train_reg, X_traj_train_reg, y_train_reg = [], [], []
    train_idxs = [i for i, gk in enumerate(group_keys) if gk in train_gks]
    for i in train_idxs:
        gk = group_keys[i]
        cand = rows[i]["cand_algo"]
        def_auc = actual_aucs_dict[gk]["SciPy-Nelder-Mead"]
        cand_auc = actual_aucs_dict[gk][cand]
        # For Regression: we want positive gain when cand is better
        # In actual_aucs_dict, AUC is log1p of area. Lower is better.
        gain = def_auc - cand_auc
        
        sf = dict(static_features_dict[gk])
        sf["cand_algo"] = cand
        X_stat_train_reg.append(sf)
        X_traj_train_reg.append(rows[i])
        y_train_reg.append(gain)
        
    df_stat_reg = pd.DataFrame(X_stat_train_reg).fillna(0.0)
    df_traj_reg = pd.DataFrame(X_traj_train_reg).fillna(0.0)
    
    cat_stat_reg = [c for c in df_stat_reg.columns if df_stat_reg[c].dtype == object or isinstance(df_stat_reg[c].iloc[0], str)]
    cat_traj_reg = [c for c in df_traj_reg.columns if df_traj_reg[c].dtype == object or isinstance(df_traj_reg[c].iloc[0], str)]
    for c in cat_stat_reg: df_stat_reg[c] = df_stat_reg[c].astype(str)
    for c in cat_traj_reg: df_traj_reg[c] = df_traj_reg[c].astype(str)
    
    reg_stat = CatBoostRegressor(iterations=300, depth=4, l2_leaf_reg=5.0, verbose=False, random_state=42, thread_count=1)
    reg_stat.fit(df_stat_reg, y_train_reg, cat_features=cat_stat_reg)
    
    reg_traj = CatBoostRegressor(iterations=300, depth=4, l2_leaf_reg=5.0, verbose=False, random_state=42, thread_count=1)
    reg_traj.fit(df_traj_reg, y_train_reg, cat_features=cat_traj_reg)
    
    # 3. Evaluate
    for gk in test_gks:
        def_auc = actual_aucs_dict[gk]["SciPy-Nelder-Mead"]
        best_algo = min(actual_aucs_dict[gk], key=actual_aucs_dict[gk].get)
        
        # Predict Clf
        df_s_test = pd.DataFrame([static_features_dict[gk]]).reindex(columns=df_stat_clf.columns).fillna(0.0)
        for c in cat_stat_clf: df_s_test[c] = df_s_test[c].astype(str)
        try: pred_stat_top1 = clf_stat.predict(df_s_test)[0][0]
        except: pred_stat_top1 = "SciPy-Nelder-Mead"
        
        idx = group_keys.index(gk)
        t_f = {k:v for k,v in rows[idx].items() if not k.startswith("rel_") and not k.startswith("diff_") and k != "cand_algo" and k != "eval_time_ratio"}
        df_t_test = pd.DataFrame([t_f]).reindex(columns=df_traj_clf.columns).fillna(0.0)
        for c in cat_traj_clf: df_t_test[c] = df_t_test[c].astype(str)
        try: pred_traj_top1 = clf_traj.predict(df_t_test)[0][0]
        except: pred_traj_top1 = "SciPy-Nelder-Mead"
        
        # Predict Reg
        preds_stat_reg = {}
        preds_traj_reg = {}
        for cand in CANDS:
            sf = dict(static_features_dict[gk])
            sf["cand_algo"] = cand
            df_sr = pd.DataFrame([sf]).reindex(columns=df_stat_reg.columns).fillna(0.0)
            for c in cat_stat_reg: df_sr[c] = df_sr[c].astype(str)
            preds_stat_reg[cand] = reg_stat.predict(df_sr)[0]
            
            c_idx = [i for i, k in enumerate(group_keys) if k == gk and rows[i]["cand_algo"] == cand]
            if c_idx:
                df_tr = pd.DataFrame([rows[c_idx[0]]]).reindex(columns=df_traj_reg.columns).fillna(0.0)
                for c in cat_traj_reg: df_tr[c] = df_tr[c].astype(str)
                preds_traj_reg[cand] = reg_traj.predict(df_tr)[0]
            else:
                preds_traj_reg[cand] = -999.0
                
        best_cand_stat = max(preds_stat_reg, key=preds_stat_reg.get)
        best_cand_traj = max(preds_traj_reg, key=preds_traj_reg.get)
        
        # Decisions
        import hashlib
        np.random.seed(int(hashlib.md5(gk.encode('utf-8')).hexdigest(), 16) % (2**32))
        decisions = {
            "B1_Default": "SciPy-Nelder-Mead",
            "B2_Random": np.random.choice(CANDS + ["SciPy-Nelder-Mead"]),
            "B3_Oracle": best_algo,
            "B4_StatTop1": pred_stat_top1,
            "B5_TrajTop1": pred_traj_top1,
            "B6_StatGate": best_cand_stat if preds_stat_reg[best_cand_stat] > 0.1 else "SciPy-Nelder-Mead",
            "B7_TrajNoGate": best_cand_traj if preds_traj_reg[best_cand_traj] > 0.0 else "SciPy-Nelder-Mead",
            "Ours_TrajGate_0.1": best_cand_traj if preds_traj_reg[best_cand_traj] > 0.1 else "SciPy-Nelder-Mead",
            "Ours_tau_0.0": best_cand_traj if preds_traj_reg[best_cand_traj] > 0.0 else "SciPy-Nelder-Mead",
            "Ours_tau_0.05": best_cand_traj if preds_traj_reg[best_cand_traj] > 0.05 else "SciPy-Nelder-Mead",
            "Ours_tau_0.2": best_cand_traj if preds_traj_reg[best_cand_traj] > 0.2 else "SciPy-Nelder-Mead",
            "Ours_tau_0.5": best_cand_traj if preds_traj_reg[best_cand_traj] > 0.5 else "SciPy-Nelder-Mead"
        }
        
        for m, dec in decisions.items():
            gain = def_auc - actual_aucs_dict[gk].get(dec, def_auc)
            acc = 1 if dec == best_algo else 0
            results[m][test_fam]["gains"].append(gain)
            results[m][test_fam]["accs"].append(acc)

def print_metrics(methods_to_print, title):
    print(f"\n{title}")
    print(f"{'Method':<20} | {'Mean Gain':>12} | {'Win Rate':>12} | {'Loss Rate':>12} | {'Top1 Acc':>12}")
    print("-" * 75)
    for m in methods_to_print:
        fam_gains, fam_wins, fam_losses, fam_accs = [], [], [], []
        for fam in unique_families:
            g = np.array(results[m][fam]["gains"])
            a = np.array(results[m][fam]["accs"])
            if len(g) == 0: continue
            fam_gains.append(np.mean(g))
            fam_wins.append(np.mean(g > 0))
            fam_losses.append(np.mean(g < 0))
            fam_accs.append(np.mean(a))
        
        mg = np.mean(fam_gains)
        mw = np.mean(fam_wins)
        ml = np.mean(fam_losses)
        ma = np.mean(fam_accs)
        print(f"{m:<20} | {mg:>+12.3f} | {mw*100:>11.1f}% | {ml*100:>11.1f}% | {ma*100:>11.1f}%")

print_metrics([
    "B1_Default", "B2_Random", "B3_Oracle", 
    "B4_StatTop1", "B5_TrajTop1", 
    "B6_StatGate", "B7_TrajNoGate", "Ours_TrajGate_0.1"
], "--- 1. ORTHOGONAL ABLATION (FAMILY-LEVEL MACRO AVERAGES) ---")

print_metrics([
    "Ours_tau_0.0", "Ours_tau_0.05", "Ours_TrajGate_0.1", "Ours_tau_0.2", "Ours_tau_0.5"
], "--- 2. RISK-RETURN FRONTIER: TAU THRESHOLD ABLATION ---")

gains_ours = [np.mean(results["Ours_TrajGate_0.1"][fam]["gains"]) for fam in unique_families if results["Ours_TrajGate_0.1"][fam]["gains"]]
gains_b4 = [np.mean(results["B4_StatTop1"][fam]["gains"]) for fam in unique_families if results["B4_StatTop1"][fam]["gains"]]
gains_b6 = [np.mean(results["B6_StatGate"][fam]["gains"]) for fam in unique_families if results["B6_StatGate"][fam]["gains"]]

_, p_b4 = wilcoxon(gains_ours, gains_b4, alternative='two-sided')
_, p_b6 = wilcoxon(gains_ours, gains_b6, alternative='two-sided')

print(f"\n--- STATISTICAL SIGNIFICANCE (Family-Level Wilcoxon) ---")
print(f"Ours vs Static Top-1 (B4) : p-value = {p_b4:.3f}")
print(f"Ours vs Static Gated (B6) : p-value = {p_b6:.3f}")
