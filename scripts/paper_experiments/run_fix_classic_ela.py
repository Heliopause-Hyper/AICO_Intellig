import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import qmc, skew, kurtosis, pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# 1. BBOB-like Functions with random shifts
def get_shifted_func(base_func, dim):
    shift = np.random.uniform(-2, 2, size=dim)
    def f(x):
        return base_func(x - shift)
    return f

def sphere(x): return np.sum(x**2)
def rastrigin(x): return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
def rosenbrock(x): return np.sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
def ackley(x): return np.sum(20 + np.e - 20 * np.exp(-0.2 * np.sqrt(np.mean(x**2))) - np.exp(np.mean(np.cos(2 * np.pi * x))))
def griewank(x): return 1 + np.sum(x**2)/4000 - np.prod(np.cos(x/np.sqrt(np.arange(1, len(x)+1))))
def schwefel(x): return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

BASE_FUNCTIONS = [
    ("Sphere", sphere), 
    ("Rastrigin", rastrigin), 
    ("Rosenbrock", rosenbrock), 
    ("Ackley", ackley), 
    ("Griewank", griewank),
    ("Schwefel", schwefel)
]

def extract_classic_ela(func, dim, n_samples=None):
    if n_samples is None: n_samples = 50 * dim
    sampler = qmc.LatinHypercube(d=dim)
    X = sampler.random(n=n_samples) * 10 - 5
    y = np.array([func(x) for x in X])
    
    features = {
        "y_skew": skew(y), 
        "y_kurtosis": kurtosis(y), 
        "y_span": np.max(y) - np.min(y),
        "y_mean": np.mean(y),
        "y_var": np.var(y)
    }
    best_idx = np.argmin(y)
    distances = np.linalg.norm(X - X[best_idx], axis=1)
    fdc, _ = pearsonr(y, distances)
    features["fdc"] = fdc if not np.isnan(fdc) else 0.0
    try:
        features["linear_r2"] = LinearRegression().fit(X, y).score(X, y)
    except:
        features["linear_r2"] = 0.0
    return features

def find_best_algo(func, dim, max_evals=100):
    x0 = np.random.uniform(-5, 5, size=dim)
    results = {}
    res_nm = minimize(func, x0, method="Nelder-Mead", options={"maxfev": max_evals})
    results["Nelder-Mead"] = res_nm.fun
    res_cob = minimize(func, x0, method="COBYLA", options={"maxiter": max_evals})
    results["COBYLA"] = res_cob.fun
    res_pow = minimize(func, x0, method="Powell", options={"maxfev": max_evals})
    results["Powell"] = res_pow.fun
    res_bfgs = minimize(func, x0, method="L-BFGS-B", options={"maxfun": max_evals})
    results["L-BFGS-B"] = res_bfgs.fun
    
    return min(results, key=results.get)

data_math = []
labels_math = []
np.random.seed(42)

for name, base_func in BASE_FUNCTIONS:
    for dim in [2, 5, 10]:
        for i in range(40): # 40 instances per func/dim to match the 74.5% run
            func = get_shifted_func(base_func, dim)
            ela_feats = extract_classic_ela(func, dim)
            best_algo = find_best_algo(func, dim)
            
            data_math.append(ela_feats)
            labels_math.append(best_algo)

df_math = pd.DataFrame(data_math)
X_math = df_math.values
y_math = np.array(labels_math)

print("--- RE-RUNNING CLASSIC ELA ON MATH BENCHMARKS ---")
print(f"Total instances: {len(X_math)}")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X_math, y_math, cv=cv, scoring='accuracy')
print(f"5-Fold CV Accuracy: {scores.mean():.1%} (+/- {scores.std():.1%})")

