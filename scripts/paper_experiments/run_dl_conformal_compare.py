import os
import json
import numpy as np
import pandas as pd
import math
from scipy.stats import wilcoxon
import sys
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("EVALUATING DEEP LEARNING (POINT CLOUD) UNDER STRICT RISK-AWARE METRICS")
print("="*80)

# We load the results we generated in run_deep_ela_eng.py previously.
# Since we didn't save the exact instance-level predictions for DL, I will re-run 
# the simplified DL model logic quickly and calculate CVaR, Win/Loss Rate, and Mean Gain.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold

OUT_DIRS = [
    "/home/ycl/AICO-Intellig/results/ex_advection2d_v1_full_200_m100_t3600_r1_20260709_231148",
    "/home/ycl/AICO-Intellig/results/ex_blackscholes2d_v1_full_200_m100_t3600_r1_20260709_104116",
    "/home/ycl/AICO-Intellig/results/ex_heat_time_v1_full_200_m100_t3600_r1_",
    "/home/ycl/AICO-Intellig/results/heat_v1_full_180_m30_t90_r3",
    "/home/ycl/AICO-Intellig/results/iaea_v1_full_180_m30_t90_r3_20260708_102705"
]

instances = []
MAX_DIM = 15
N_POINTS = 20

for out_dir in OUT_DIRS:
    if not os.path.exists(out_dir): continue
    for shard in os.listdir(out_dir):
        shard_dir = os.path.join(out_dir, shard)
        if not os.path.isdir(shard_dir) or not shard.startswith("shard_"): continue
        
        run_sum_path = os.path.join(shard_dir, "run_summary.jsonl")
        pi_path = os.path.join(shard_dir, "problem_instance.jsonl")
        event_path = os.path.join(shard_dir, "iteration_event.jsonl")
        
        if not (os.path.exists(run_sum_path) and os.path.exists(event_path) and os.path.exists(pi_path)):
            continue
            
        pi_dict = {}
        for line in open(pi_path):
            pi = json.loads(line)
            pi_dict[pi["instance_id"]] = pi
            
        nm_runs = {}
        labels = {}
        actual_aucs = {}
        for line in open(run_sum_path):
            rs = json.loads(line)
            iid = rs["instance_id"]
            algo = f"{rs['backend_lib']}-{rs['method']}"
            if algo == "SciPy-Nelder-Mead":
                nm_runs[rs["run_id"]] = iid
            
            if iid not in labels: labels[iid] = {}
            if iid not in actual_aucs: actual_aucs[iid] = {}
            
            auc = rs.get("anytime_auc")
            if auc is not None:
                auc_score = math.log1p(min(1e6, max(0.0, float(auc))))
                labels[iid][algo] = auc_score
                actual_aucs[iid][algo] = auc_score
            
        run_events = {}
        for line in open(event_path):
            ev = json.loads(line)
            rid = ev["run_id"]
            if rid in nm_runs and ev["eval_index"] <= N_POINTS:
                run_events.setdefault(rid, []).append(ev)
                
        for rid, iid in nm_runs.items():
            evs = run_events.get(rid, [])
            evs.sort(key=lambda x: x["eval_index"])
            if len(evs) < 10: continue
            
            pc = np.zeros((N_POINTS, MAX_DIM + 1))
            for i, ev in enumerate(evs[:N_POINTS]):
                x_arr = np.array(ev["x_json"])
                dim = min(len(x_arr), MAX_DIM)
                pc[i, :dim] = x_arr[:dim]
                pc[i, -1] = ev.get("loss_value", 1e6)
            
            y_col = pc[:, -1]
            pc[:, -1] = (y_col - np.mean(y_col)) / (np.std(y_col) + 1e-9)
            
            aucs = labels.get(iid, {})
            if "SciPy-Nelder-Mead" not in aucs: continue
            if not aucs: continue
            
            # Lower AUC is better
            best_algo = min(aucs, key=aucs.get)
            
            pi = pi_dict.get(iid, {})
            family = pi.get("family_version", pi.get("problem_type", "unknown"))
            
            instances.append({
                "pc": pc,
                "family": family,
                "best_algo": best_algo,
                "aucs": aucs
            })

print(f"Loaded {len(instances)} engineering instances for DL evaluation.")

X = np.array([inst["pc"] for inst in instances])
families = np.array([inst["family"] for inst in instances])
unique_labels = list(set(inst["best_algo"] for inst in instances))
label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
inv_label_map = {i: lbl for lbl, i in label_map.items()}
y = np.array([label_map[inst["best_algo"]] for inst in instances])

class DeepELA_Simplified(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, 128), nn.ReLU(), nn.BatchNorm1d(128)
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        B, N, D = x.size()
        x = x.view(B * N, D)
        x = self.shared_mlp(x)
        x = x.view(B, N, -1)
        x = torch.max(x, dim=1)[0]
        x = self.global_mlp(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unique_fams = list(set(families))

dl_gains = []

for test_fam in unique_fams:
    train_idx = [i for i, f in enumerate(families) if f != test_fam]
    test_idx = [i for i, f in enumerate(families) if f == test_fam]
    if len(train_idx) == 0 or len(test_idx) == 0: continue
    
    model = DeepELA_Simplified(input_dim=16, num_classes=len(unique_labels)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    X_train, y_train = torch.FloatTensor(X[train_idx]), torch.LongTensor(y[train_idx])
    X_test, y_test = torch.FloatTensor(X[test_idx]), torch.LongTensor(y[test_idx])
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    for epoch in range(30):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(device)).argmax(dim=1).cpu().numpy()
        
    for i, pred_idx in enumerate(preds):
        inst_idx = test_idx[i]
        pred_algo = inv_label_map[pred_idx]
        def_auc = instances[inst_idx]["aucs"]["SciPy-Nelder-Mead"]
        pred_auc = instances[inst_idx]["aucs"].get(pred_algo, def_auc)
        # Gain = Default - Predicted (since lower AUC is better)
        dl_gains.append(def_auc - pred_auc)

def calc_cvar(gains, alpha=0.1):
    gains = np.array(gains)
    threshold = np.percentile(gains, alpha * 100)
    tail_losses = gains[gains <= threshold]
    return np.mean(tail_losses) if len(tail_losses) > 0 else 0.0

def compute_metrics(gains, name):
    gains = np.array(gains)
    uncond_mean = np.mean(gains)
    win_rate = np.mean(gains > 0)
    loss_rate = np.mean(gains < 0)
    switch_rate = np.mean(gains != 0)
    cvar_10 = calc_cvar(gains, 0.1)
    worst_case = np.min(gains) if len(gains) > 0 else 0.0
    
    print(f"[{name}]")
    print(f"  - Mean Net Gain:       {uncond_mean:+.3f}")
    print(f"  - Win Rate:            {win_rate*100:.1f}%")
    print(f"  - Loss Rate (Risk):    {loss_rate*100:.1f}%")
    print(f"  - Switch Rate:         {switch_rate*100:.1f}%")
    print(f"  - CVaR_10 (Tail Risk): {cvar_10:+.3f} (closer to 0 is safer)")
    print(f"  - Worst Case Loss:     {worst_case:+.3f}")
    return gains

print("\n--- RESULTS UNDER RISK-AWARE METRICS ---")
g_dl = compute_metrics(dl_gains, "Deep-ELA (Point Cloud Classification)")

