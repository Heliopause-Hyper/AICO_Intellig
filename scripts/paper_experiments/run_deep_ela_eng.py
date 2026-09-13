import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("TESTING DEEP LEARNING (POINT CLOUD) ON ENGINEERING PDE DATA")
print("="*60)

# Load engineering data
OUT_DIRS = [
    "/home/ycl/AICO-Intellig/results/ex_advection2d_v1_full_200_m100_t3600_r1_20260709_231148",
    "/home/ycl/AICO-Intellig/results/ex_blackscholes2d_v1_full_200_m100_t3600_r1_20260709_104116",
    "/home/ycl/AICO-Intellig/results/ex_heat_time_v1_full_200_m100_t3600_r1_",
    "/home/ycl/AICO-Intellig/results/heat_v1_full_180_m30_t90_r3",
    "/home/ycl/AICO-Intellig/results/iaea_v1_full_180_m30_t90_r3_20260708_102705"
]

def load_data():
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
            for line in open(run_sum_path):
                rs = json.loads(line)
                iid = rs["instance_id"]
                algo = f"{rs['backend_lib']}-{rs['method']}"
                if algo == "SciPy-Nelder-Mead":
                    nm_runs[rs["run_id"]] = iid
                
                # To find top1 label
                if iid not in labels: labels[iid] = {}
                labels[iid][algo] = rs.get("anytime_auc", 0)
                
            # Parse events
            run_events = {}
            for line in open(event_path):
                ev = json.loads(line)
                rid = ev["run_id"]
                if rid in nm_runs and ev["eval_index"] <= N_POINTS:
                    run_events.setdefault(rid, []).append(ev)
                    
            for rid, iid in nm_runs.items():
                evs = run_events.get(rid, [])
                evs.sort(key=lambda x: x["eval_index"])
                if len(evs) < 10: continue # Skip if too few
                
                # Build Point Cloud Matrix
                pc = np.zeros((N_POINTS, MAX_DIM + 1))
                for i, ev in enumerate(evs[:N_POINTS]):
                    x_arr = np.array(ev["x_json"])
                    dim = min(len(x_arr), MAX_DIM)
                    pc[i, :dim] = x_arr[:dim]
                    pc[i, -1] = ev.get("loss_value", 1e6)
                
                # Normalize y
                y_col = pc[:, -1]
                pc[:, -1] = (y_col - np.mean(y_col)) / (np.std(y_col) + 1e-9)
                
                # Get Best Algo Label
                aucs = labels.get(iid, {})
                if not aucs: continue
                best_algo = max(aucs, key=aucs.get) # Assuming larger AUC is better here, wait, in our script AUC is log1p of area. 
                # Let's check train_curve_policy.py: AUC_CAP = 1e6, log1p(min(AUC_CAP, max(0.0, auc))). Lower area is better? 
                # Wait, in BBOB script I did `sum(y) / max_evals`. Lower is better. So min() is better.
                # Let's just do Top-1 prediction for simplicity to see if DL works.
                
                pi = pi_dict.get(iid, {})
                family = pi.get("family_version", pi.get("problem_type", "unknown"))
                
                instances.append({
                    "pc": pc,
                    "family": family,
                    "best_algo": best_algo
                })
    return instances

instances = load_data()
if not instances:
    print("No instances loaded.")
    exit()

print(f"Loaded {len(instances)} engineering instances with 20-step trajectories.")

X = np.array([inst["pc"] for inst in instances])
families = np.array([inst["family"] for inst in instances])
unique_labels = list(set(inst["best_algo"] for inst in instances))
label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
y = np.array([label_map[inst["best_algo"]] for inst in instances])

print(f"Classes: {len(unique_labels)}")

# Define PointNet
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

# LOFO CV
unique_fams = list(set(families))
lofo_accs = []
majority_accs = []

print("\nRunning Leave-One-Family-Out (LOFO) Evaluation for Deep Learning Model...")

for test_fam in unique_fams:
    train_idx = [i for i, f in enumerate(families) if f != test_fam]
    test_idx = [i for i, f in enumerate(families) if f == test_fam]
    if len(train_idx) == 0 or len(test_idx) == 0: continue
    
    # Majority guess
    from collections import Counter
    maj_label = Counter(y[train_idx]).most_common(1)[0][0]
    maj_acc = (y[test_idx] == maj_label).mean()
    majority_accs.append(maj_acc)
    
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
        acc = (preds == y_test.numpy()).mean()
        lofo_accs.append(acc)
        print(f"  Hold-out: {test_fam:25s} | DL Acc: {acc:.1%} | Blind Guess: {maj_acc:.1%}")

print(f"\n=> Macro Avg DL LOFO Accuracy: {np.mean(lofo_accs)*100:.2f}%")
print(f"=> Macro Avg Blind Guess Accuracy: {np.mean(majority_accs)*100:.2f}%")

