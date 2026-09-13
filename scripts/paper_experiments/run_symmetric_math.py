import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import qmc, skew, kurtosis, pearsonr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.linear_model import LinearRegression
import math
import warnings
warnings.filterwarnings('ignore')

# 1. Functions
def get_shifted_func(base_func, dim):
    shift = np.random.uniform(-2, 2, size=dim)
    def f(x): return base_func(x - shift)
    return f

def sphere(x): return np.sum(x**2)
def rastrigin(x): return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
def rosenbrock(x): return np.sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
def ackley(x): return np.sum(20 + np.e - 20 * np.exp(-0.2 * np.sqrt(np.mean(x**2))) - np.exp(np.mean(np.cos(2 * np.pi * x))))
def griewank(x): return 1 + np.sum(x**2)/4000 - np.prod(np.cos(x/np.sqrt(np.arange(1, len(x)+1))))
def schwefel(x): return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

BASE_FUNCTIONS = [
    ("Sphere", sphere), ("Rastrigin", rastrigin), ("Rosenbrock", rosenbrock), 
    ("Ackley", ackley), ("Griewank", griewank), ("Schwefel", schwefel)
]

def extract_classic_ela(func, dim):
    sampler = qmc.LatinHypercube(d=dim)
    X = sampler.random(n=50*dim) * 10 - 5
    y = np.array([func(x) for x in X])
    
    best_idx = np.argmin(y)
    distances = np.linalg.norm(X - X[best_idx], axis=1)
    fdc, _ = pearsonr(y, distances)
    
    try: r2 = LinearRegression().fit(X, y).score(X, y)
    except: r2 = 0.0
        
    return [skew(y), kurtosis(y), np.max(y)-np.min(y), np.mean(y), np.var(y), fdc if not np.isnan(fdc) else 0.0, r2]

def run_algo(func, dim, algo_name, max_evals=100):
    history = []
    evals = 0
    def wrapped_func(x):
        nonlocal evals
        evals += 1
        y = func(x)
        history.append((evals, y))
        return y
    x0 = np.random.uniform(-5, 5, size=dim)
    if algo_name == "Nelder-Mead": minimize(wrapped_func, x0, method="Nelder-Mead", options={"maxfev": max_evals})
    elif algo_name == "COBYLA": minimize(wrapped_func, x0, method="COBYLA", options={"maxiter": max_evals})
    elif algo_name == "Powell": minimize(wrapped_func, x0, method="Powell", options={"maxfev": max_evals})
    
    while len(history) < max_evals: history.append((len(history)+1, history[-1][1] if history else 0.0))
    history = history[:max_evals]
    best_history = []
    curr_best = float('inf')
    for e, y in history:
        curr_best = min(curr_best, y)
        best_history.append(curr_best)
    auc = sum(best_history) / max_evals
    return best_history, math.log1p(min(1e6, max(0.0, auc)))

# Generate Dataset
np.random.seed(42)
instances = []
for name, base_func in BASE_FUNCTIONS:
    for dim in [2, 5, 10]:
        for i in range(25): # 450 instances
            func = get_shifted_func(base_func, dim)
            ela_feats = extract_classic_ela(func, dim)
            
            nm_hist, nm_auc = run_algo(func, dim, "Nelder-Mead")
            cob_hist, cob_auc = run_algo(func, dim, "COBYLA")
            pow_hist, pow_auc = run_algo(func, dim, "Powell")
            
            aucs = {"Nelder-Mead": nm_auc, "COBYLA": cob_auc, "Powell": pow_auc}
            true_best = min(aucs, key=aucs.get)
            
            instances.append({
                "family": name, "dim": dim,
                "ela_feats": ela_feats,
                "aucs": aucs,
                "true_best": true_best,
                "nm_hist": nm_hist, "cob_hist": cob_hist, "pow_hist": pow_hist
            })

# Prepare Evaluation
def evaluate(train_idxs, test_idxs):
    # --- 1. Static ELA + Top-1 ---
    X_static_train = [instances[i]["ela_feats"] for i in train_idxs]
    y_static_train = [instances[i]["true_best"] for i in train_idxs]
    clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_static_train, y_static_train)
    
    X_static_test = [instances[i]["ela_feats"] for i in test_idxs]
    static_preds = clf.predict(X_static_test)
    
    static_acc = 0; static_win = 0; static_gain = 0.0
    
    # --- 2. Trajectory ELA + Pairwise Gated ---
    X_pair_train = []
    y_pair_train = []
    for i in train_idxs:
        inst = instances[i]
        nm_20_best = inst["nm_hist"][19]
        nm_20_area = sum(inst["nm_hist"][:20])
        for cand, cand_hist in [("COBYLA", inst["cob_hist"]), ("Powell", inst["pow_hist"])]:
            feat = [
                inst["dim"], 1 if cand=="COBYLA" else 0,
                (nm_20_best - cand_hist[19]) / max(1e-9, abs(nm_20_best)),
                (nm_20_area - sum(cand_hist[:20])) / max(1e-9, abs(nm_20_area))
            ]
            X_pair_train.append(feat)
            y_pair_train.append(inst["aucs"]["Nelder-Mead"] - inst["aucs"][cand])
            
    reg = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_pair_train, y_pair_train)
    
    pair_win = 0; pair_gain = 0.0
    
    for idx_in_test, i in enumerate(test_idxs):
        inst = instances[i]
        def_auc = inst["aucs"]["Nelder-Mead"]
        
        # Static Eval
        pred_algo = static_preds[idx_in_test]
        if pred_algo == inst["true_best"]: static_acc += 1
        static_actual_gain = def_auc - inst["aucs"][pred_algo]
        if static_actual_gain > 0: static_win += 1
        static_gain += static_actual_gain
        
        # Pairwise Eval
        nm_20_best = inst["nm_hist"][19]
        nm_20_area = sum(inst["nm_hist"][:20])
        preds = {}
        for cand, cand_hist in [("COBYLA", inst["cob_hist"]), ("Powell", inst["pow_hist"])]:
            feat = [
                inst["dim"], 1 if cand=="COBYLA" else 0,
                (nm_20_best - cand_hist[19]) / max(1e-9, abs(nm_20_best)),
                (nm_20_area - sum(cand_hist[:20])) / max(1e-9, abs(nm_20_area))
            ]
            preds[cand] = reg.predict([feat])[0]
            
        best_cand = max(preds, key=preds.get)
        decision = best_cand if preds[best_cand] > 0.0 else "Nelder-Mead"
        
        pair_actual_gain = def_auc - inst["aucs"][decision]
        if pair_actual_gain > 0: pair_win += 1
        pair_gain += pair_actual_gain
        
    n_test = len(test_idxs)
    return {
        "static_acc": static_acc/n_test, "static_win": static_win/n_test, "static_gain": static_gain/n_test,
        "pair_win": pair_win/n_test, "pair_gain": pair_gain/n_test
    }

# K-Fold CV (Random Split)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
kf_res = [evaluate(train, test) for train, test in kf.split(instances)]

# LOFO CV (Leave One Function Out)
groups = [inst["family"] for inst in instances]
logo = LeaveOneGroupOut()
lofo_res = [evaluate(train, test) for train, test in logo.split(instances, groups=groups)]

def agg(res_list, key): return np.mean([r[key] for r in res_list])

print("\n--- SYMMETRIC EXPERIMENT RESULTS ON MATH DATASET ---")
print("1. K-Fold CV (Random Split - Seen Families):")
print(f"   [Static Top-1] Acc: {agg(kf_res, 'static_acc'):.1%}, Win Rate vs NM: {agg(kf_res, 'static_win'):.1%}, AUC Gain: {agg(kf_res, 'static_gain'):.3f}")
print(f"   [Pairwise Gated] Win Rate vs NM: {agg(kf_res, 'pair_win'):.1%}, AUC Gain: {agg(kf_res, 'pair_gain'):.3f}")

print("\n2. LOFO CV (Leave One Family Out - Unseen Families):")
print(f"   [Static Top-1] Acc: {agg(lofo_res, 'static_acc'):.1%}, Win Rate vs NM: {agg(lofo_res, 'static_win'):.1%}, AUC Gain: {agg(lofo_res, 'static_gain'):.3f}")
print(f"   [Pairwise Gated] Win Rate vs NM: {agg(lofo_res, 'pair_win'):.1%}, AUC Gain: {agg(lofo_res, 'pair_gain'):.3f}")

