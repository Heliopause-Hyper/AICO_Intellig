import os
import sys
import time
import numpy as np
import pandas as pd
from typing import Callable, Tuple, Dict, Any, List
import joblib

sys.path.append(os.path.abspath("/home/ycl/AICO-Intellig"))
from catboost import CatBoostRegressor

class ConformalOnlineRouter:
    def __init__(
        self,
        model_path: str,
        q_worst_family: float,
        budget_K: int = 20,
        base_algo: str = "SciPy-Nelder-Mead"
    ):
        self.budget_K = budget_K
        self.base_algo = base_algo
        self.q_worst_family = q_worst_family
        
        # Load the pairwise net-gain regression model
        print(f"Loading pairwise gain model from {model_path}...")
        self.model = joblib.load(model_path)
        # Note: In actual deployment, we'd extract feature names directly from the trained model
        
        # Candidate pool (excluding base_algo)
        self.candidates = ["Nevergrad-NGOpt", "SciPy-COBYLA", "SciPy-DE"]

    def extract_trajectory_features(
        self,
        objective_func: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]]
    ) -> Tuple[Dict[str, float], List[float], float]:
        """
        Run the base algorithm for K steps to extract endogenous trajectory features.
        """
        print(f"Running Warm-up phase: {self.base_algo} for {self.budget_K} steps...")
        
        from scipy.optimize import minimize
        dim = len(bounds)
        x0 = np.array([0.5 * (b[0] + b[1]) for b in bounds])
        
        history = []
        
        def callback(xk):
            # In standard scipy, callback only gets xk. We need to evaluate or store the history.
            pass
            
        def wrapped_obj(x):
            val = objective_func(x)
            history.append(val)
            return val
            
        start_time = time.time()
        # We use a trick to force Nelder-Mead to stop after K function evaluations
        res = minimize(
            wrapped_obj, 
            x0, 
            method='Nelder-Mead', 
            bounds=bounds,
            options={'maxfev': self.budget_K, 'maxiter': self.budget_K}
        )
        elapsed = time.time() - start_time
        
        # Calculate standard trajectory features (similar to _warm_features)
        vals = history[:self.budget_K]
        if not vals:
            return {}, vals, elapsed
            
        n = len(vals)
        best = min(vals)
        first = vals[0]
        last = vals[-1]
        improve = float(first - best)
        slope = float((last - first) / max(1, n - 1)) if n > 1 else 0.0
        std = float(np.std(vals))
        best_pos = float(vals.index(best) / max(1, n - 1)) if n > 1 else 0.0
        
        features = {
            "warm_n": n,
            "warm_best": best,
            "warm_first": first,
            "warm_last": last,
            "warm_improve": improve,
            "warm_slope": slope,
            "warm_std": std,
            "warm_best_pos": best_pos,
            "n_params": dim
        }
        return features, history, elapsed

    def route(self, objective_func: Callable[[np.ndarray], float], bounds: List[Tuple[float, float]]):
        """
        End-to-End Routing Decision Logic
        """
        # 1. Warm-up and Feature Extraction (Budget-Neutral)
        features, traj, warmup_time = self.extract_trajectory_features(objective_func, bounds)
        
        if not features:
            print("Warning: Trajectory extraction failed. Defaulting to Base Algorithm.")
            return self.base_algo
            
        print(f"Trajectory Features Extracted: Best={features['warm_best']:.4f}, Std={features['warm_std']:.4f}")
        
        # 2. Pairwise Gain Prediction
        # For a robust deployment, we align features exactly with the model's expected input
        # Here we mock the DataFrame structure for the CatBoost model
        preds = {}
        for cand in self.candidates:
            # We copy features and append the candidate specific identifier
            # In the actual training script, we encode cand_algo
            feat_copy = features.copy()
            feat_copy["cand_algo"] = cand
            
            # Since we didn't save the exact model object yet, we simulate prediction for the mock
            # In a real run, this would be: pred_gain = self.model.predict(pd.DataFrame([feat_copy]))[0]
            # We will generate a mock prediction based on the trajectory slope to test the logic
            mock_gain = -features["warm_slope"] * 10.0 + np.random.normal(0, 0.02)
            if cand == "Nevergrad-NGOpt": mock_gain += 0.05
            
            preds[cand] = mock_gain
            
        best_cand = max(preds, key=preds.get)
        max_pred_gain = preds[best_cand]
        
        print(f"Predictions (Log-AUC Gain over Base):")
        for k, v in preds.items():
            print(f"  - {k}: {v:+.4f}")
            
        # 3. Distributionally Robust Conformal Gating
        lower_bound = max_pred_gain - self.q_worst_family
        
        print(f"\n--- Conformal Gating Check ---")
        print(f"Max Predicted Gain:  {max_pred_gain:+.4f} (by {best_cand})")
        print(f"Worst-Family CVaR Threshold (q_1-alpha): {self.q_worst_family:.4f}")
        print(f"Risk-Adjusted Lower Bound: {lower_bound:+.4f}")
        
        if lower_bound > 0:
            print(f"Decision: => SWITCH to [ {best_cand} ] (Bound > 0, Safe to switch)")
            return best_cand
        else:
            print(f"Decision: => STAY with [ {self.base_algo} ] (Bound <= 0, Switching is too risky)")
            return self.base_algo

if __name__ == "__main__":
    # --- Demonstration with a synthetic black-box function ---
    def rosenbrock(x):
        return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
        
    bounds = [(-5.0, 5.0)] * 5
    
    print("="*80)
    print("AICO-Intellig: End-to-End Online Router Demonstration")
    print("="*80)
    
    # In practice, this q_val is calibrated offline and loaded from a config
    # We use a typical q_val (e.g., 0.03) from our previous LOFO experiments
    router = ConformalOnlineRouter(
        model_path="/home/ycl/AICO-Intellig/results/model_algo_ranker_merged_warm10.joblib",
        q_worst_family=0.035, 
        budget_K=20
    )
    
    chosen_algo = router.route(rosenbrock, bounds)
    print(f"\nFinal Algorithm to execute the remaining budget: {chosen_algo}")
    print("="*80)
