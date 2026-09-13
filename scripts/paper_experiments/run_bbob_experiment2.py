import sys
import os
sys.path.append(os.path.abspath("/home/ycl/AICO-Intellig"))
from scripts.train_algo_ranker import CatBoostDictRegressor

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import catboost
import joblib
import math

# Load Models
STATIC_MODEL_PATH = "/home/ycl/AICO-Intellig/results/static_ranker_baseline_latest/model_static_ranker_catboost.joblib"
static_model_payload = joblib.load(STATIC_MODEL_PATH)
static_model = static_model_payload["model"]

PAIRWISE_MODEL_PATH = "/home/ycl/AICO-Intellig/results/pairwise_curve_policy_eval20_20260714_174746/model_pairwise_auc_gain.joblib"
pairwise_model_payload = joblib.load(PAIRWISE_MODEL_PATH)
pairwise_model = pairwise_model_payload["model"]

# Simple BBOB-like functions
def sphere(x): return np.sum(x**2)
def rastrigin(x): return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
def rosenbrock(x): return np.sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
def ackley(x): return np.sum(20 + np.e - 20 * np.exp(-0.2 * np.sqrt(np.mean(x**2))) - np.exp(np.mean(np.cos(2 * np.pi * x))))

FUNCTIONS = [
    ("Sphere", sphere, "separable"),
    ("Rastrigin", rastrigin, "multimodal_adequate"),
    ("Rosenbrock", rosenbrock, "moderate_conditioning"),
    ("Ackley", ackley, "multimodal_weak")
]

np.random.seed(42)

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
    
    if algo_name == "Nelder-Mead":
        minimize(wrapped_func, x0, method="Nelder-Mead", options={"maxfev": max_evals, "xatol": 1e-8, "fatol": 1e-8})
    elif algo_name == "COBYLA":
        minimize(wrapped_func, x0, method="COBYLA", options={"maxiter": max_evals, "tol": 1e-8})
    elif algo_name == "Powell":
        minimize(wrapped_func, x0, method="Powell", options={"maxfev": max_evals, "xtol": 1e-8, "ftol": 1e-8})
        
    while len(history) < max_evals:
        history.append((len(history)+1, history[-1][1] if history else 0.0))
        
    history = history[:max_evals]
    best_history = []
    current_best = float('inf')
    for e, y in history:
        current_best = min(current_best, y)
        best_history.append((e, current_best))
        
    auc = sum(y for e, y in best_history) / max_evals
    log_auc = math.log1p(min(1e6, max(0.0, auc)))
    return best_history, log_auc

def calc_20_step_features(hist):
    best_loss = hist[19][1]
    best_prefix_area = sum(y for e, y in hist[:20])
    # Dummy other features
    return {
        "best_loss": best_loss,
        "prefix_best_area_norm": best_prefix_area / (20.0 * max(1e-9, abs(hist[0][1]))),
        "feasible_ratio": 1.0,
        "stagnation_length": 0,
        "avg_eval_time": 0.01,
        "recent_failure_rate": 0.0
    }

results = []

for func_name, func, pde_type in FUNCTIONS:
    for dim in [5, 10]:
        nm_hist, nm_auc = run_algo(func, dim, "Nelder-Mead")
        cob_hist, cob_auc = run_algo(func, dim, "COBYLA")
        pow_hist, pow_auc = run_algo(func, dim, "Powell")
        
        actuals = {
            "SciPy-Nelder-Mead": nm_auc,
            "SciPy-COBYLA": cob_auc,
            "SciPy-Powell": pow_auc
        }
        
        true_best = min(actuals, key=actuals.get)
        
        # 1. Static Model Prediction (Tries to predict AUC directly based on static features)
        # We give it the true dimension, but since it has never seen these "pde_types", it will likely fail to generalize
        static_f = {
            "decision_dim": dim,
            "problem_type": pde_type,
            "budget_eval": 100
        }
        
        # We have to build rows for all 3 algos
        static_preds = {}
        for algo in actuals.keys():
            row = dict(static_f)
            parts = algo.split("-")
            row["optimizer"] = parts[0]
            row["method"] = parts[1] if len(parts) > 1 else parts[0]
            
            # The CatBoostDictRegressor expects a list of dicts, but we'll use pandas directly for the raw model
            df = pd.DataFrame([row])
            for col in static_model_payload["columns"]:
                if col not in df.columns:
                    df[col] = 0.0
            df = df.reindex(columns=static_model_payload["columns"])
            for col in static_model_payload["cat_features"]:
                df[col] = df[col].fillna("").astype(str)
            static_preds[algo] = static_model.predict(df)[0]
            
        static_best = min(static_preds, key=static_preds.get)
        
        # 2. Pairwise Model Prediction (Tries to predict gain of COBYLA/Powell over Nelder-Mead)
        nm_ck = calc_20_step_features(nm_hist)
        
        pairwise_preds = {}
        for cand_algo, cand_hist in [("SciPy-COBYLA", cob_hist), ("SciPy-Powell", pow_hist)]:
            cand_ck = calc_20_step_features(cand_hist)
            
            row = {
                "decision_dim": dim,
                "cand_algo": cand_algo,
                "rel_best_gain": (nm_ck["best_loss"] - cand_ck["best_loss"]) / max(1e-9, abs(nm_ck["best_loss"])),
                "rel_auc_gain": (nm_ck["prefix_best_area_norm"] - cand_ck["prefix_best_area_norm"]) / max(1e-9, abs(nm_ck["prefix_best_area_norm"])),
                "diff_feasible": 0.0,
                "diff_stagnation": 0.0,
                "eval_time_ratio": 1.0,
                "diff_failure": 0.0
            }
            
            df = pd.DataFrame([row])
            for col in pairwise_model_payload["columns"]:
                if col not in df.columns:
                    df[col] = 0.0
            df = df.reindex(columns=pairwise_model_payload["columns"])
            for col in pairwise_model_payload["cat_features"]:
                df[col] = df[col].fillna("").astype(str)
            pairwise_preds[cand_algo] = pairwise_model.predict(df)[0]
            
        # Pairwise Decision Logic
        best_cand = max(pairwise_preds, key=pairwise_preds.get)
        best_gain = pairwise_preds[best_cand]
        
        pairwise_decision = best_cand if best_gain > 0.5 else "SciPy-Nelder-Mead"
        
        results.append({
            "Function": f"{func_name}-{dim}D",
            "True_Best": true_best,
            "Static_Pred": static_best,
            "Pairwise_Pred": pairwise_decision
        })

df = pd.DataFrame(results)
print("\n--- GENERALIZATION TEST ON MATH FUNCTIONS ---")
print(df.to_string())

