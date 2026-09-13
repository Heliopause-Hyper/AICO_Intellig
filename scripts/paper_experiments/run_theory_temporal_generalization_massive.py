import json
import math
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath("/home/ycl/AICO-Intellig"))

from scripts.paper_experiments.theory_validation_common import compute_gain_metrics  # noqa: E402
from scripts.train_curve_policy import (  # noqa: E402
    AUC_CAP,
    _build_pairwise_features,
    _collect_shards,
    _family_name,
    _instance_features,
    _read_jsonl,
    _safe_float,
)


def _algo_key(run_summary_row):
    return f"{run_summary_row.get('backend_lib') or ''}-{run_summary_row.get('method') or ''}"


def _auc_score(run_summary_row):
    v = _safe_float(run_summary_row.get("anytime_auc"), None)
    if v is None:
        return None
    return float(math.log1p(min(float(AUC_CAP), max(0.0, float(v)))))


def _load_k_checkpoints(out_dirs, k: int):
    run_summaries = {}
    pi_by_instance = {}
    ck_by_run = {}

    for out_dir in out_dirs:
        for shard in _collect_shards(out_dir):
            rp = os.path.join(shard, "run_summary.jsonl")
            pip = os.path.join(shard, "problem_instance.jsonl")
            ckp = os.path.join(shard, "run_checkpoint.jsonl")

            if os.path.exists(rp):
                for row in _read_jsonl(rp):
                    rid = str(row.get("run_id") or "")
                    if rid:
                        run_summaries[rid] = row

            if os.path.exists(pip):
                for row in _read_jsonl(pip):
                    iid = str(row.get("instance_id") or "")
                    if iid:
                        pi_by_instance[iid] = row

            if os.path.exists(ckp):
                for row in _read_jsonl(ckp):
                    rid = str(row.get("run_id") or "")
                    if not rid:
                        continue
                    ei = int(row.get("eval_index") or 0)
                    if ei != int(k):
                        continue
                    ck_by_run[rid] = row

    return run_summaries, pi_by_instance, ck_by_run


def _checkpoint_to_ck(row):
    if row is None:
        return None
    out = dict(row)
    if "prefix_best_area_norm" not in out:
        out["prefix_best_area_norm"] = out.get("auc_regret_prefix")
    return out


def _build_pairwise_rows(out_dirs, k: int, default_algo: str, candidate_pool: set):
    run_summaries, pi_by_instance, ck_by_run = _load_k_checkpoints(out_dirs, int(k))

    group_runs = defaultdict(dict)
    family_by_group = {}
    auc_by_group = defaultdict(dict)

    for rid, srow in run_summaries.items():
        iid = str(srow.get("instance_id") or "")
        if not iid:
            continue
        seed = int(srow.get("seed") or 0)
        gk = f"{iid}__seed{seed}"

        algo = _algo_key(srow)
        group_runs[gk][algo] = {"rid": rid, "summary": srow}

    rows = []
    y_gain = []
    group_keys = []
    groups_out = []

    for gk, algos in group_runs.items():
        if default_algo not in algos:
            continue

        def_run = algos[default_algo]
        def_rid = def_run["rid"]
        def_summary = def_run["summary"]
        def_auc = _auc_score(def_summary)
        if def_auc is None:
            continue

        def_ck_raw = ck_by_run.get(def_rid)
        def_ck = _checkpoint_to_ck(def_ck_raw)
        if def_ck is None:
            continue

        iid = str(def_summary.get("instance_id") or "")
        pi = pi_by_instance.get(iid)
        fam = _family_name(pi, def_summary)
        family_by_group[gk] = fam

        f_inst = _instance_features(pi)

        for algo, payload in algos.items():
            auc = _auc_score(payload["summary"])
            if auc is None:
                continue
            auc_by_group[gk][algo] = float(auc)

        for cand_algo, payload in algos.items():
            if cand_algo == default_algo:
                continue
            if candidate_pool and cand_algo not in candidate_pool:
                continue
            cand_rid = payload["rid"]
            cand_ck_raw = ck_by_run.get(cand_rid)
            cand_ck = _checkpoint_to_ck(cand_ck_raw)
            if cand_ck is None:
                continue
            cand_auc = auc_by_group[gk].get(cand_algo)
            if cand_auc is None:
                continue

            feat = _build_pairwise_features(f_inst, def_ck, cand_ck, cand_algo)
            rows.append(feat)
            y_gain.append(float(def_auc - float(cand_auc)))
            group_keys.append(gk)
            groups_out.append(gk)

    unique_groups = sorted(set(groups_out))
    return rows, y_gain, group_keys, family_by_group, auc_by_group, unique_groups


def _conformal_quantile(residuals, alpha: float) -> float:
    xs = np.asarray(list(residuals), dtype=float)
    if xs.size == 0:
        return 0.0
    n = int(xs.size)
    q_level = min(1.0, (n + 1.0) * (1.0 - float(alpha)) / float(n))
    return float(np.quantile(xs, q_level))


def _fit_gain_regressor(rows, ys):
    df_train = pd.DataFrame(rows).fillna(0.0)
    cat_features = [
        c
        for c in df_train.columns
        if df_train[c].dtype == object or isinstance(df_train[c].iloc[0], str)
    ]
    for c in cat_features:
        df_train[c] = df_train[c].astype(str)
    reg = CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=5,
        l2_leaf_reg=10.0,
        verbose=False,
        random_state=42,
        thread_count=1,
    )
    reg.fit(df_train, ys, cat_features=cat_features)
    return reg, df_train.columns.tolist(), cat_features


def _predict_gain(reg, columns, cat_features, row):
    df = pd.DataFrame([row]).reindex(columns=columns).fillna(0.0)
    for c in cat_features:
        df[c] = df[c].astype(str)
    return float(reg.predict(df)[0])


def _best_candidate_by_pred(reg, columns, cat_features, feats_by_group, gk, candidates):
    preds = {}
    base = feats_by_group[gk]
    for cand in candidates:
        row = dict(base)
        row["cand_algo"] = cand
        preds[cand] = _predict_gain(reg, columns, cat_features, row)
    best_cand = max(preds, key=preds.get)
    return best_cand, float(preds[best_cand])


def _policy_metrics_from_pairs(pairs, tau):
    gains = []
    for pred, true_gain in pairs:
        gains.append(float(true_gain) if float(pred) > float(tau) else 0.0)
    return compute_gain_metrics(gains)


print("=" * 80)
print("THEORY VALIDATION E6: TEMPORAL GENERALIZATION USING MASSIVE ARCHIVE (RUN_CHECKPOINT)")
print("=" * 80)

K = 20
ALPHA = 0.10
LAMBDAS = [0.0, 0.10, 0.20, 0.40]

DEFAULT_ALGO = "SciPy-Nelder-Mead"
POOL9 = {
    "SciPy-DE",
    "SciPy-Powell",
    "SciPy-Nelder-Mead",
    "Optuna-TPE",
    "Optuna-CMA-ES",
    "Hyperopt-TPE",
    "Hyperopt-Random",
    "Nevergrad-NGOpt",
    "Nevergrad-PSO",
}
CANDIDATES = set(POOL9) - {DEFAULT_ALGO}

TRAIN_DIRS = [
    "/home/ycl/AICO-Intellig/results/massive_pde_database_v1",
    "/home/ycl/AICO-Intellig/results/massive_pde_extensions_v1",
    "/home/ycl/AICO-Intellig/results/sns_family_protocol_v3_instances240_more1",
    "/home/ycl/AICO-Intellig/results/heat_family_protocol_v1_instances180",
    "/home/ycl/AICO-Intellig/results/iaea_family_protocol_v1_instances180",
    "/home/ycl/AICO-Intellig/results/sns_family_protocol_v1_instances500",
]

TEST_DIRS = [
    "/home/ycl/AICO-Intellig/results/expansion_20260826/ex_pde_v1_pool9_r3",
    "/home/ycl/AICO-Intellig/results/expansion_20260826/ex_heat_time_v1_pool9_r3",
]

rows_tr, y_tr, gks_tr, fam_tr, aucs_tr, train_groups = _build_pairwise_rows(
    TRAIN_DIRS, K, DEFAULT_ALGO, CANDIDATES
)
rows_te, y_te, gks_te, fam_te, aucs_te, test_groups = _build_pairwise_rows(
    TEST_DIRS, K, DEFAULT_ALGO, CANDIDATES
)

print(f"Train groups: {len(train_groups)}; pairwise rows: {len(rows_tr)}")
print(f"Test groups:  {len(test_groups)}; pairwise rows: {len(rows_te)}")

if not train_groups or not test_groups:
    raise RuntimeError("insufficient groups for temporal generalization experiment")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
proper_train_idx, calib_idx = next(kf.split(train_groups))
proper_train_groups = [train_groups[i] for i in proper_train_idx]
calib_groups = [train_groups[i] for i in calib_idx]

train_idxs = [i for i, gk in enumerate(gks_tr) if gk in proper_train_groups]
X_train = [rows_tr[i] for i in train_idxs]
y_train = [y_tr[i] for i in train_idxs]

reg, columns, cat_features = _fit_gain_regressor(X_train, y_train)

base_feats_by_group = {}
for gk in train_groups:
    idx = next((i for i, key in enumerate(gks_tr) if key == gk), None)
    if idx is not None:
        d = dict(rows_tr[idx])
        d.pop("cand_algo", None)
        base_feats_by_group[gk] = d
for gk in test_groups:
    idx = next((i for i, key in enumerate(gks_te) if key == gk), None)
    if idx is not None:
        d = dict(rows_te[idx])
        d.pop("cand_algo", None)
        base_feats_by_group[gk] = d

calib_pairs = []
calib_residuals = []

for gk in calib_groups:
    aucs = aucs_tr.get(gk) or {}
    candidates = [c for c in CANDIDATES if c in aucs]
    if not candidates or gk not in base_feats_by_group:
        continue
    best_cand, best_pred = _best_candidate_by_pred(
        reg, columns, cat_features, base_feats_by_group, gk, candidates
    )
    true_gain = float(aucs.get(DEFAULT_ALGO)) - float(aucs.get(best_cand))
    calib_pairs.append((float(best_pred), float(true_gain)))
    calib_residuals.append(float(best_pred - true_gain))

q_val = _conformal_quantile(calib_residuals, ALPHA)

preds = sorted(set(float(p) for p, _ in calib_pairs))
grid = []
if preds:
    qs_grid = np.quantile(np.asarray(preds, dtype=float), np.linspace(0.0, 1.0, 41))
    grid = sorted(set(float(x) for x in qs_grid))
grid = sorted(set(grid + [0.0]))

best_mean = None
best_cvar = None
tau_mean = 0.0
tau_cvar = 0.0
for tau in grid:
    m = _policy_metrics_from_pairs(calib_pairs, tau)
    if best_mean is None or m["mean_gain"] > best_mean:
        best_mean = float(m["mean_gain"])
        tau_mean = float(tau)
    if best_cvar is None or m["cvar_10"] > best_cvar:
        best_cvar = float(m["cvar_10"])
        tau_cvar = float(tau)

g_tau0 = []
g_tau_mean = []
g_tau_cvar = []
g_conformal = []
g_cost = {lam: [] for lam in LAMBDAS}
g_cost_util = {lam: [] for lam in LAMBDAS}

per_family = defaultdict(lambda: defaultdict(list))

for gk in test_groups:
    aucs = aucs_te.get(gk) or {}
    candidates = [c for c in CANDIDATES if c in aucs]
    if not candidates or gk not in base_feats_by_group:
        continue
    best_cand, best_pred = _best_candidate_by_pred(
        reg, columns, cat_features, base_feats_by_group, gk, candidates
    )
    true_gain = float(aucs.get(DEFAULT_ALGO)) - float(aucs.get(best_cand))

    fam = fam_te.get(gk) or "unknown"

    g_tau0.append(float(true_gain) if float(best_pred) > 0.0 else 0.0)
    g_tau_mean.append(float(true_gain) if float(best_pred) > float(tau_mean) else 0.0)
    g_tau_cvar.append(float(true_gain) if float(best_pred) > float(tau_cvar) else 0.0)

    lower_bound = float(best_pred) - float(q_val)
    switched = lower_bound > 0.0
    g_conformal.append(float(true_gain) if switched else 0.0)
    per_family["conformal_lb"][fam].append(float(true_gain) if switched else 0.0)

    b = float(base_feats_by_group[gk].get("budget_eval") or 0.0)
    frac = float(K) / float(b) if b > 0 else 0.0
    for lam in LAMBDAS:
        trig = lower_bound > float(lam) * float(frac)
        raw = float(true_gain) if trig else 0.0
        util = float(raw - float(lam) * float(frac)) if trig else 0.0
        g_cost[lam].append(raw)
        g_cost_util[lam].append(util)
        per_family[f"cost_lb_lam{lam}"][fam].append(raw)

summary = {
    "k": int(K),
    "alpha": float(ALPHA),
    "q_1_minus_alpha": float(q_val),
    "tau_mean": float(tau_mean),
    "tau_cvar": float(tau_cvar),
    "train_groups": int(len(train_groups)),
    "test_groups": int(len(test_groups)),
    "policies": {
        "tau0": compute_gain_metrics(g_tau0),
        "tau_star_mean": compute_gain_metrics(g_tau_mean),
        "tau_star_cvar": compute_gain_metrics(g_tau_cvar),
        "conformal_lb": compute_gain_metrics(g_conformal),
    },
    "cost_regularized": {
        str(lam): {
            "raw": compute_gain_metrics(g_cost[lam]),
            "utility": compute_gain_metrics(g_cost_util[lam]),
        }
        for lam in LAMBDAS
    },
    "per_family": {
        pol: {fam: compute_gain_metrics(xs) for fam, xs in fam_map.items()}
        for pol, fam_map in per_family.items()
    },
}

out_dir = "/home/ycl/AICO-Intellig/results/theory_validation_latest"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "temporal_generalization_massive.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
    f.write("\n")

print("")
print("[Temporal Test Policies]")
for name, m in summary["policies"].items():
    print(
        f"{name:>14s}  mean={m['mean_gain']:+.3f}  "
        f"cvar10={m['cvar_10']:+.3f}  "
        f"loss={m['loss_rate']*100:5.1f}%  "
        f"switch={m['switch_rate']*100:5.1f}%"
    )
print("")
for lam in LAMBDAS:
    m = summary["cost_regularized"][str(lam)]["raw"]
    print(
        f"cost_lb lam={lam:>4.2f}  mean={m['mean_gain']:+.3f}  "
        f"cvar10={m['cvar_10']:+.3f}  "
        f"loss={m['loss_rate']*100:5.1f}%  "
        f"switch={m['switch_rate']*100:5.1f}%"
    )
