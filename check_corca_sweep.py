import os
import glob
import json

results_dir = "/home/ycl/AICO-Intellig/results/massive_pde_database_v1"
files = glob.glob(os.path.join(results_dir, "trajectory_*.json"))
corca_files = []
for f in files:
    try:
        with open(f, 'r') as fp:
            data = json.load(fp)
            if data.get('problem_type') == 'corca_state':
                corca_files.append(data)
    except:
        pass

print(f"Total corca_state trajectories found: {len(corca_files)}")
if len(corca_files) > 0:
    success_count = sum(1 for d in corca_files if d.get('feasible', False))
    print(f"Successful corca_state trajectories: {success_count}/{len(corca_files)}")
