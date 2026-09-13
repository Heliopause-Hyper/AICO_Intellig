import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesClassifier
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath("/home/ycl/AICO-Intellig"))
from scripts.train_algo_ranker import _features_from_instance, _drop_family_like_features

print("="*60)
print("ENGINEERING DATASET: 5-FOLD CROSS VALIDATION (SEEN FAMILIES)")
print("="*60)

# --- 1. Static ELA + Top-1 ---
print("1. Evaluating Static Top-1 Model...")
static_data = []
with open("/home/ycl/AICO-Intellig/results/static_ranker_baseline_latest/dataset.jsonl", "r") as f:
    for line in f:
        static_data.append(json.loads(line))

X_static = []
y_static = []
for row in static_data:
    f = _features_from_instance(row, row)
    f = _drop_family_like_features(f)
    X_static.append(f)
    y_static.append(row["label"])

# Filter categorical values just to be safe
df_static = pd.DataFrame(X_static)
for col in df_static.select_dtypes(include=['object']).columns:
    df_static[col] = df_static[col].astype('category').cat.codes
df_static = df_static.fillna(0.0)

X_static_mat = df_static.values
y_static_arr = np.array(y_static)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
static_accs = []
for train_idx, test_idx in kf.split(X_static_mat):
    clf = ExtraTreesClassifier(n_estimators=100, random_state=42, max_depth=8)
    clf.fit(X_static_mat[train_idx], y_static_arr[train_idx])
    acc = clf.score(X_static_mat[test_idx], y_static_arr[test_idx])
    static_accs.append(acc)

print(f"   -> [Static Top-1] K-Fold Accuracy: {np.mean(static_accs):.1%} (+/- {np.std(static_accs):.1%})")


# --- 2. Pairwise Gated Switching ---
print("\n2. Evaluating Pairwise Gated Switching Model (20-Step Trajectory)...")
from scripts.train_curve_policy import _build_pairwise_dataset, CatBoostDictRegressor

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

unique_groups = list(set(group_keys))
kf_groups = KFold(n_splits=5, shuffle=True, random_state=42)

pair_wins = 0
pair_gains = []
total_groups = 0

for train_g_idx, test_g_idx in kf_groups.split(unique_groups):
    train_groups = set([unique_groups[i] for i in train_g_idx])
    test_groups = set([unique_groups[i] for i in test_g_idx])
    
    train_idxs = [i for i, gk in enumerate(group_keys) if gk in train_groups]
    test_idxs = [i for i, gk in enumerate(group_keys) if gk in test_groups]
    
    X_train = [rows[i] for i in train_idxs]
    y_train = [y_auc_gain[i] for i in train_idxs]
    X_test = [rows[i] for i in test_idxs]
    
    m_auc = CatBoostDictRegressor(random_state=42).fit(X_train, y_train)
    pred_gains = m_auc.predict(X_test)
    
    preds_by_group = {}
    for i, pred in zip(test_idxs, pred_gains):
        gk = group_keys[i]
        cand = rows[i]["cand_algo"]
        preds_by_group.setdefault(gk, []).append((pred, cand))
        
    for gk, preds in preds_by_group.items():
        best_pred, best_cand = max(preds, key=lambda x: x[0])
        total_groups += 1
        
        def_auc = actual_aucs_dict[gk]["SciPy-Nelder-Mead"]
        
        if best_pred > 0.5:
            cand_auc = actual_aucs_dict[gk][best_cand]
            actual_gain = cand_auc - def_auc
        else:
            actual_gain = 0.0
            
        pair_gains.append(actual_gain)
        if actual_gain > 0:
            pair_wins += 1

print(f"   -> [Pairwise Gated] K-Fold Win Rate vs NM: {pair_wins/total_groups:.1%}")
print(f"   -> [Pairwise Gated] K-Fold Mean AUC Gain: +{np.mean(pair_gains):.3f}")

