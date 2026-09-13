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

# Simple BBOB-like functions
def sphere(x): return np.sum(x**2)
def rastrigin(x): return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
def rosenbrock(x): return np.sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
def ackley(x): return np.sum(20 + np.e - 20 * np.exp(-0.2 * np.sqrt(np.mean(x**2))) - np.exp(np.mean(np.cos(2 * np.pi * x))))

FUNCTIONS = [
    ("Sphere", sphere),
    ("Rastrigin", rastrigin),
    ("Rosenbrock", rosenbrock),
    ("Ackley", ackley)
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
        minimize(wrapped_func, x0, method="COBYLA", options={"maxeval": max_evals, "tol": 1e-8})
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

results = []

for func_name, func in FUNCTIONS:
    for dim in [5, 10]:
        nm_hist, nm_auc = run_algo(func, dim, "Nelder-Mead")
        cob_hist, cob_auc = run_algo(func, dim, "COBYLA")
        pow_hist, pow_auc = run_algo(func, dim, "Powell")
        
        actuals = {
            "Nelder-Mead": nm_auc,
            "COBYLA": cob_auc,
            "Powell": pow_auc
        }
        
        best_algo = min(actuals, key=actuals.get)
        
        results.append({
            "Function": f"{func_name}-{dim}D",
            "NelderMead_AUC": nm_auc,
            "COBYLA_AUC": cob_auc,
            "Powell_AUC": pow_auc,
            "True_Best_Algo": best_algo
        })

df = pd.DataFrame(results)
print("\n--- ACTUAL GROUND TRUTH ON MATH FUNCTIONS ---")
print("Note: Lower AUC is better.")
print(df.to_string())

