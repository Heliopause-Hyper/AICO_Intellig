import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import qmc, skew, kurtosis, pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# 1. BBOB-like Functions with random shifts to create instances
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

BASE_FUNCTIONS = [sphere, rastrigin, rosenbrock, ackley, griewank]

# 2. Extract "Classic ELA" Features (using Latin Hypercube Sampling)
# Previous papers typically use 50*d points for feature extraction
def extract_ela_features(func, dim, n_samples=None):
    if n_samples is None:
        n_samples = 50 * dim
        
    sampler = qmc.LatinHypercube(d=dim)
    # scale to [-5, 5]
    X = sampler.random(n=n_samples) * 10 - 5
    y = np.array([func(x) for x in X])
    
    # Basic y-distribution features
    features = {
        "y_skew": skew(y),
        "y_kurtosis": kurtosis(y),
        "y_min": np.min(y),
        "y_max": np.max(y),
        "y_span": np.max(y) - np.min(y),
    }
    
    # Fitness Distance Correlation (FDC)
    best_idx = np.argmin(y)
    x_best = X[best_idx]
    distances = np.linalg.norm(X - x_best, axis=1)
    fdc, _ = pearsonr(y, distances)
    features["fdc"] = fdc if not np.isnan(fdc) else 0.0
    
    # Simple Meta-Model feature (Linear R2)
    try:
        lr = LinearRegression().fit(X, y)
        features["linear_r2"] = lr.score(X, y)
    except:
        features["linear_r2"] = 0.0
        
    return features

# 3. Find the True Best Algorithm
def find_best_algo(func, dim, max_evals=100):
    x0 = np.random.uniform(-5, 5, size=dim)
    results = {}
    
    # NM
    res_nm = minimize(func, x0, method="Nelder-Mead", options={"maxfev": max_evals})
    results["Nelder-Mead"] = res_nm.fun
    
    # COBYLA
    res_cob = minimize(func, x0, method="COBYLA", options={"maxiter": max_evals})
    results["COBYLA"] = res_cob.fun
    
    # Powell
    res_pow = minimize(func, x0, method="Powell", options={"maxfev": max_evals})
    results["Powell"] = res_pow.fun
    
    # L-BFGS-B (gradient based)
    res_bfgs = minimize(func, x0, method="L-BFGS-B", options={"maxfun": max_evals})
    results["L-BFGS-B"] = res_bfgs.fun
    
    return min(results, key=results.get)

# 4. Generate Dataset
print("Generating BBOB dataset and extracting Classic ELA features...")
data = []
labels = []

np.random.seed(42)
N_INSTANCES_PER_FUNC = 40
DIMS = [2, 5, 10]

for base_func in BASE_FUNCTIONS:
    for dim in DIMS:
        for _ in range(N_INSTANCES_PER_FUNC):
            func = get_shifted_func(base_func, dim)
            
            # 1. Extract ELA features (Simulating paper's approach)
            ela_feats = extract_ela_features(func, dim)
            
            # 2. Find best algo
            best_algo = find_best_algo(func, dim)
            
            data.append(ela_feats)
            labels.append(best_algo)

df = pd.DataFrame(data)
X = df.values
y = np.array(labels)

print(f"\nDataset generated: {len(X)} instances.")
print("Class distribution:")
print(pd.Series(y).value_counts())

# 5. Train and Evaluate Classic Model (Top-1 Prediction)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')

print(f"\n--- CLASSIC ELA + TOP-1 PREDICTION ON MATH BENCHMARKS ---")
print(f"5-Fold CV Accuracy: {scores.mean()*100:.2f}% (+/- {scores.std()*100:.2f}%)")
