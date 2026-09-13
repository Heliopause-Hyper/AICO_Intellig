import os
import json
import csv
import numpy as np
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

RESULTS_DIR = os.environ.get('AICO_RESULTS_DIR') or os.path.join(REPO_ROOT, 'results')
OUTPUT_FILE = os.environ.get('AICO_AGGREGATED_FILE') or os.path.join(REPO_ROOT, 'aggregated_results.csv')
TRAJECTORY_DIR = os.environ.get('AICO_TRAJECTORY_DIR') or os.path.join(REPO_ROOT, 'trajectories')

if not os.path.exists(TRAJECTORY_DIR):
    os.makedirs(TRAJECTORY_DIR)

def extract_timestamp(filename):
    try:
        parts = filename.replace('.json', '').split('_')
        if len(parts) >= 3:
            date_str = parts[-2]
            time_str = parts[-1]
            return f"{date_str}_{time_str}"
    except:
        pass
    return "unknown"

def calculate_convergence_steps(history, best_val, tolerance=0.05):
    """Calculates iterations to reach within tolerance of best value."""
    if not history or best_val is None or np.isnan(best_val):
        return -1
    
    threshold = best_val * (1.0 + tolerance) if best_val > 0 else best_val * (1.0 - tolerance)
    # Handle negative objectives or minimization context carefully. 
    # Assuming minimization for now as most AICO tasks seem to be.
    # If objective can be negative, relative tolerance is tricky. 
    # Let's use absolute tolerance if value is close to 0, else relative.
    
    # Simple approach for minimization: first time we hit <= best_val * (1+tol)
    # But wait, best_val is the global min found. 
    # We want to know when we first got "close enough" to the final result.
    
    # Using a simpler metric: First step where obj <= best_val * 1.05 (if positive)
    # or obj <= best_val * 0.95 (if negative)
    # Or just use the step index of the best value found.
    
    best_idx = -1
    for idx, item in enumerate(history):
        obj = item.get('objective')
        if obj is not None and isinstance(obj, (int, float)):
            # Check for exact match with best_value (or very close)
            if abs(obj - best_value) < 1e-9:
                return idx + 1
    return -1

def process_file(filepath, filename):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

    optimizer = data.get('optimizer', 'unknown')
    method = data.get('method', 'unknown')
    config = data.get('analysis_config', {})
    
    # Improved Problem Type Inference
    problem_type = config.get('problem_type', 'unknown')
    user_input = config.get('user_input', '').lower()
    
    if problem_type == 'unknown':
         if 'sns' in user_input or 'pressure' in user_input or 'umax' in user_input:
             problem_type = 'sns'
         elif 'thmf' in user_input or 'heat' in user_input or 'thermal' in user_input or 'fin' in user_input:
             problem_type = 'thmf'
         elif 'iaea' in user_input or 'neutron' in user_input or 'keff' in user_input or 'reactor' in user_input:
             problem_type = 'iaea'

    execution_time = data.get('execution_time', float('nan'))
    best_value = data.get('best_value', float('nan'))
    iterations = data.get('iterations', 0)
    success = data.get('success', False)
    message = data.get('message', '')
    
    # --- New Extractions ---
    best_params = data.get('best_params', {})
    # Flatten best params for CSV (stringify)
    best_params_str = json.dumps(best_params)
    
    history = data.get('iteration_history', [])
    objectives = []
    
    # Convergence: Iteration where best value was first achieved
    convergence_step = -1
    found_best = False
    
    # Eval success rate
    successful_evals = 0
    total_evals = 0
    
    for idx, item in enumerate(history):
        total_evals += 1
        res = item.get('result', {})
        if res.get('success', False):
            successful_evals += 1
            
        obj = item.get('objective')
        if obj is not None and isinstance(obj, (int, float)):
            objectives.append(obj)
            # Ensure best_value is numeric before comparison
            if not found_best and best_value is not None and isinstance(best_value, (int, float)) and not np.isnan(best_value):
                try:
                    if abs(obj - best_value) < 1e-7:
                        convergence_step = idx + 1
                        found_best = True
                except:
                    pass

    eval_success_rate = (successful_evals / total_evals) if total_evals > 0 else 0.0
    variance = np.var(objectives) if len(objectives) > 1 else 0.0
    
    # Save trajectory
    timestamp = extract_timestamp(filename)
    traj_filename = f"{problem_type}_{optimizer}_{method}_{timestamp}.csv"
    traj_path = os.path.join(TRAJECTORY_DIR, traj_filename)
    
    if history:
        with open(traj_path, 'w', newline='') as traj_f:
            writer = csv.writer(traj_f)
            writer.writerow(['step', 'objective', 'params'])
            for idx, item in enumerate(history):
                writer.writerow([idx+1, item.get('objective'), json.dumps(item.get('params'))])

    return {
        'file_name': filename,
        'timestamp': timestamp,
        'problem_type': problem_type,
        'optimizer': optimizer,
        'method': method,
        'execution_time': execution_time,
        'best_value': best_value,
        'iterations': iterations,
        'success': success,
        'message': message,
        'history_length': len(history),
        'variance': variance,
        'convergence_step': convergence_step,
        'eval_success_rate': eval_success_rate,
        'best_params': best_params_str,
        'trajectory_file': traj_filename
    }

def main():
    results = []
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        print(f"No results directory found. Created empty directory at {RESULTS_DIR}.")
        print("No valid data found.")
        return

    files = [f for f in os.listdir(RESULTS_DIR) if f.startswith('opt_result_') and f.endswith('.json')]
    print(f"Processing {len(files)} files...")

    for filename in files:
        filepath = os.path.join(RESULTS_DIR, filename)
        row = process_file(filepath, filename)
        if row:
            results.append(row)

    if not results:
        print("No valid data found.")
        return

    headers = ['file_name', 'timestamp', 'problem_type', 'optimizer', 'method', 
               'execution_time', 'best_value', 'iterations', 'success', 'message', 
               'history_length', 'variance', 'convergence_step', 'eval_success_rate', 
               'best_params', 'trajectory_file']
    
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)

    print(f"Extended aggregated data saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
