import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("REPLICATING FEATURE-FREE DEEP LEARNING (SIMPLIFIED DEEP-ELA/PCT)")
print("="*60)

# 1. BBOB-like Functions
def get_shifted_func(base_func, dim):
    shift = np.random.uniform(-2, 2, size=dim)
    def f(x): return base_func(x - shift)
    return f

def sphere(x): return np.sum(x**2)
def rastrigin(x): return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
def rosenbrock(x): return np.sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
def ackley(x): return np.sum(20 + np.e - 20 * np.exp(-0.2 * np.sqrt(np.mean(x**2))) - np.exp(np.mean(np.cos(2 * np.pi * x))))
def griewank(x): return 1 + np.sum(x**2)/4000 - np.prod(np.cos(x/np.sqrt(np.arange(1, len(x)+1))))

BASE_FUNCTIONS = [sphere, rastrigin, rosenbrock, ackley, griewank]

# 2. Extract Raw Point Cloud (X, y) instead of manual ELA features
def extract_point_cloud(func, dim, n_samples=None):
    if n_samples is None: n_samples = 50 * dim
    sampler = qmc.LatinHypercube(d=dim)
    X = sampler.random(n=n_samples) * 10 - 5
    y = np.array([func(x) for x in X]).reshape(-1, 1)
    
    # Normalize y to avoid exploding gradients in neural net
    y_mean, y_std = np.mean(y), np.std(y) + 1e-9
    y_norm = (y - y_mean) / y_std
    
    # Point cloud matrix: N x (d + 1)
    point_cloud = np.hstack([X, y_norm])
    return point_cloud

def find_best_algo(func, dim, max_evals=100):
    x0 = np.random.uniform(-5, 5, size=dim)
    results = {}
    results["Nelder-Mead"] = minimize(func, x0, method="Nelder-Mead", options={"maxfev": max_evals}).fun
    results["COBYLA"] = minimize(func, x0, method="COBYLA", options={"maxiter": max_evals}).fun
    results["Powell"] = minimize(func, x0, method="Powell", options={"maxfev": max_evals}).fun
    return min(results, key=results.get)

# Generate Dataset
print("Generating Point Cloud Dataset (50*d samples per instance)...")
data_pcs = []
labels = []

np.random.seed(42)
N_INSTANCES = 50
DIM = 5  # Fix dimension to 5 for neural net input consistency in this simplified script
N_POINTS = 50 * DIM

for base_func in BASE_FUNCTIONS:
    for _ in range(N_INSTANCES):
        func = get_shifted_func(base_func, DIM)
        pc = extract_point_cloud(func, DIM, N_POINTS)
        best_algo = find_best_algo(func, DIM)
        
        data_pcs.append(pc)
        labels.append(best_algo)

X = np.array(data_pcs) # Shape: (250, 250, 6)
label_map = {lbl: i for i, lbl in enumerate(np.unique(labels))}
y = np.array([label_map[lbl] for lbl in labels])

print(f"Dataset generated: {len(X)} instances. Shape: {X.shape}")

# 3. Define PointNet-like Architecture (Permutation Invariant Deep Learning)
class DeepELA_Simplified(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        # Shared MLP across all points
        self.shared_mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        # Global MLP after pooling
        self.global_mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x shape: (Batch, N_points, input_dim)
        B, N, D = x.size()
        x = x.view(B * N, D)
        x = self.shared_mlp(x)
        x = x.view(B, N, -1)
        # Max pooling across points (permutation invariant)
        x = torch.max(x, dim=1)[0]
        x = self.global_mlp(x)
        return x

# 4. Train and Evaluate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nTraining Deep Learning Model on {device}...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accs = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    model = DeepELA_Simplified(input_dim=DIM+1, num_classes=len(label_map)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    X_train, y_train = torch.FloatTensor(X[train_idx]), torch.LongTensor(y[train_idx])
    X_test, y_test = torch.FloatTensor(X[test_idx]), torch.LongTensor(y[test_idx])
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Train for 40 epochs
    for epoch in range(40):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    # Evaluate
    model.eval()
    with torch.no_grad():
        X_test = X_test.to(device)
        preds = model(X_test).argmax(dim=1).cpu().numpy()
        acc = (preds == y_test.numpy()).mean()
        fold_accs.append(acc)

print(f"\n--- DEEP LEARNING (RAW POINT CLOUD) TOP-1 PREDICTION ON MATH BENCHMARKS ---")
print(f"5-Fold CV Accuracy: {np.mean(fold_accs)*100:.2f}% (+/- {np.std(fold_accs)*100:.2f}%)")

print("\n" + "="*60)
print("CRITICAL ENGINEERING REALITY CHECK:")
print("="*60)
print(f"To achieve this accuracy, the Deep-ELA model required sampling {N_POINTS} points per instance.")
print(f"In our Engineering PDE scenarios, 1 evaluation = 5 minutes.")
print(f"Time required just to gather features for ONE instance: {N_POINTS * 5 / 60:.1f} HOURS.")
print("This completely exceeds the total optimization budget (30-100 evals).")
print("Conclusion: Deep-ELA is brilliant for cheap math functions, but structurally DEAD ON ARRIVAL for expensive engineering problems.")
