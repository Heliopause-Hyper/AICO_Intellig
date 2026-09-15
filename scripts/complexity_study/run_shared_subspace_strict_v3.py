from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

ROOT = "/home/ycl/AICO-Intellig"
DEFAULT_OUT = os.path.join(ROOT, "results", "complexity_study", "shared_subspace_strict_v3.json")


def proj_ball(x: np.ndarray, radius: float) -> np.ndarray:
    n = float(np.linalg.norm(x))
    return x if n <= radius or n == 0.0 else x * (radius / n)


def ortho(rng: np.random.Generator, d: int, k: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((d, k)))
    return q[:, :k]


class Task:
    def __init__(self, basis: np.ndarray, hessian: np.ndarray, z_star: np.ndarray, ambient: float, radius: float):
        self.basis = basis
        self.hessian = hessian
        self.z_star = z_star
        self.ambient = ambient
        self.radius = radius
        self.d = int(basis.shape[0])
        self.k = int(basis.shape[1])

    def optimum(self) -> np.ndarray:
        return proj_ball(self.basis @ self.z_star, self.radius)

    def eval(self, x: np.ndarray) -> float:
        x = proj_ball(x, self.radius)
        dx = x - self.optimum()
        z = self.basis.T @ dx
        return 0.5 * float(z.T @ self.hessian @ z) + 0.5 * self.ambient * float(dx.T @ dx)


def make_task(rng: np.random.Generator, basis: np.ndarray, cond: float, ambient: float, radius: float) -> Task:
    k = int(basis.shape[1])
    eigs = np.exp(rng.uniform(0.0, math.log(cond), size=k))
    return Task(basis, np.diag(eigs), rng.uniform(-0.6, 0.6, size=k), ambient, radius)


def coord_grad(task: Task, x: np.ndarray, eps: float):
    g = np.zeros(task.d)
    for i in range(task.d):
        e = np.zeros(task.d)
        e[i] = eps
        g[i] = (task.eval(proj_ball(x + e, task.radius)) - task.eval(proj_ball(x - e, task.radius))) / (2.0 * eps)
    return g, 2 * task.d


def estimate_basis(tasks, k: int, probe_points: int, fd_eps: float, rng: np.random.Generator):
    grads, cost = [], 0
    for task in tasks:
        for _ in range(probe_points):
            x = proj_ball(rng.standard_normal(task.d), task.radius)
            g, c = coord_grad(task, x, fd_eps)
            grads.append(g)
            cost += c
    _, _, vh = np.linalg.svd(np.asarray(grads), full_matrices=False)
    q, _ = np.linalg.qr(vh[:k].T)
    return q[:, :k], cost


def projector_err(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.linalg.norm(u @ u.T - v @ v.T, ord=2))


def fd_descent(task: Task, basis: np.ndarray, x0: np.ndarray, budget: int, fd_eps: float):
    z = basis.T @ proj_ball(x0, task.radius)
    x = proj_ball(basis @ z, task.radius)
    cur = task.eval(x)
    best, trace, evals, step = float(cur), [float(cur)], 1, 0.2
    while evals < budget:
        g = np.zeros(int(basis.shape[1]))
        for j in range(int(basis.shape[1])):
            if evals + 2 > budget:
                break
            dvec = basis[:, j]
            xp = proj_ball(x + fd_eps * dvec, task.radius)
            xn = proj_ball(x - fd_eps * dvec, task.radius)
            fp, fn = float(task.eval(xp)), float(task.eval(xn))
            best = min(best, fp, fn)
            trace.extend([best, best])
            evals += 2
            g[j] = (fp - fn) / (2.0 * fd_eps)
        if evals >= budget:
            break
        z_new = z - step * g
        x_new = proj_ball(basis @ z_new, task.radius)
        val = float(task.eval(x_new))
        best = min(best, val)
        trace.append(best)
        evals += 1
        if val <= cur:
            z, x, cur = z_new, x_new, val
        else:
            step = 0.2 if step < 1e-4 else step * 0.5
    if len(trace) < budget:
        trace.extend([best] * (budget - len(trace)))
    return {"best_value": best, "best_trace": trace[:budget]}


def rand_dir_descent(task: Task, x0: np.ndarray, budget: int, fd_eps: float, rng: np.random.Generator):
    x = proj_ball(x0.copy(), task.radius)
    cur = float(task.eval(x))
    best, trace, evals, step = cur, [cur], 1, 0.2
    while evals < budget:
        v = rng.standard_normal(task.d)
        v /= max(float(np.linalg.norm(v)), 1e-12)
        if evals + 2 > budget:
            break
        xp = proj_ball(x + fd_eps * v, task.radius)
        xn = proj_ball(x - fd_eps * v, task.radius)
        fp, fn = float(task.eval(xp)), float(task.eval(xn))
        g = ((fp - fn) / (2.0 * fd_eps)) * v
        best = min(best, fp, fn)
        trace.extend([best, best])
        evals += 2
        if evals >= budget:
            break
        x_new = proj_ball(x - step * g, task.radius)
        val = float(task.eval(x_new))
        best = min(best, val)
        trace.append(best)
        evals += 1
        if val <= cur:
            x, cur = x_new, val
        else:
            step = 0.2 if step < 1e-4 else step * 0.5
    if len(trace) < budget:
        trace.extend([best] * (budget - len(trace)))
    return {"best_value": best, "best_trace": trace[:budget]}


def first_hit(trace, eps):
    for i, v in enumerate(trace, start=1):
        if float(v) <= eps:
            return i
    return None


def summarize(runs, thresholds, budget: int, history_share: float):
    vals = [float(r["best_value"]) for r in runs]
    out = {"n": len(runs), "mean_simple_regret": float(np.mean(vals)), "thresholds": {}}
    for eps in thresholds:
        hits = [first_hit(r["best_trace"], eps) for r in runs]
        succ = [h for h in hits if h is not None]
        capped = [(h if h is not None else budget) + history_share for h in hits]
        out["thresholds"][str(eps)] = {
            "success_rate": float(len(succ) / max(1, len(hits))),
            "mean_queries_if_hit": float(np.mean(succ)) if succ else None,
            "mean_capped_queries_with_history_share": float(np.mean(capped)),
        }
    return out


def run_suite(args):
    suite = []
    for d in args.d_values:
        rows = {k: [] for k in ["scratch_coordinate_fd", "scratch_random_direction", "history_init_random_direction", "estimated_subspace", "oracle_subspace"]}
        costs, errs = [], []
        for family_seed in range(args.family_seeds):
            rng = np.random.default_rng(8100 + d * 97 + family_seed)
            u_true = ortho(rng, d, args.k)
            history = [make_task(rng, u_true, args.condition_number, args.ambient_curvature, args.radius) for _ in range(args.history_tasks)]
            test = [make_task(rng, u_true, args.condition_number, args.ambient_curvature, args.radius) for _ in range(args.test_tasks)]
            u_est, hist_cost = estimate_basis(history, args.k, args.probe_points, args.fd_eps, rng)
            costs.append(hist_cost)
            errs.append(projector_err(u_true, u_est))
            x0 = proj_ball(rng.standard_normal(d), args.radius)
            hist_init = min(history, key=lambda t: t.eval(x0)).optimum()
            for task_i, task in enumerate(test):
                seed = 990000 + d * 1000 + family_seed * 100 + task_i
                rows["scratch_coordinate_fd"].append(fd_descent(task, np.eye(d), x0, args.budget, args.fd_eps))
                rows["scratch_random_direction"].append(rand_dir_descent(task, x0, args.budget, args.fd_eps, np.random.default_rng(seed)))
                rows["history_init_random_direction"].append(rand_dir_descent(task, hist_init, args.budget, args.fd_eps, np.random.default_rng(seed + 1)))
                rows["estimated_subspace"].append(fd_descent(task, u_est, x0, args.budget, args.fd_eps))
                rows["oracle_subspace"].append(fd_descent(task, u_true, x0, args.budget, args.fd_eps))
        item = {"d": d, "history_query_cost_mean": float(np.mean(costs)), "projector_error_mean": float(np.mean(errs)), "methods": {}}
        for m in args.amortized_m_values:
            item["methods"][f"M={m}"] = {
                name: summarize(rr, args.thresholds, args.budget, (float(np.mean(costs)) / m) if ("history" in name or "subspace" in name) else 0.0)
                for name, rr in rows.items()
            }
        suite.append(item)
    return suite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--d_values", nargs="+", type=int, default=[32, 64])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--family_seeds", type=int, default=10)
    ap.add_argument("--history_tasks", type=int, default=4)
    ap.add_argument("--test_tasks", type=int, default=12)
    ap.add_argument("--probe_points", type=int, default=2)
    ap.add_argument("--budget", type=int, default=180)
    ap.add_argument("--condition_number", type=float, default=12.0)
    ap.add_argument("--ambient_curvature", type=float, default=0.02)
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--fd_eps", type=float, default=1e-3)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.1, 0.02, 0.005])
    ap.add_argument("--amortized_m_values", nargs="+", type=int, default=[1, 5, 20])
    args = ap.parse_args()
    out = {"config": vars(args), "suite": run_suite(args)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(os.path.abspath(args.out), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print("=" * 80)
    print("STRICT SHARED-SUBSPACE V3")
    print("=" * 80)
    for row in out["suite"]:
        m = row["methods"]["M=5"]
        print(f"d={row['d']} coord={m['scratch_coordinate_fd']['mean_simple_regret']:.4f} rand={m['scratch_random_direction']['mean_simple_regret']:.4f} hist_init={m['history_init_random_direction']['mean_simple_regret']:.4f} est={m['estimated_subspace']['mean_simple_regret']:.4f} oracle={m['oracle_subspace']['mean_simple_regret']:.4f}")
    print(f"Wrote: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
