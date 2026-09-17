from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

ROOT = "/home/ycl/AICO-Intellig"
DEFAULT_OUT = os.path.join(ROOT, "results", "complexity_study", "shared_subspace_strict_v4.json")


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


def collect_history_observations(tasks, k: int, probe_points: int, fd_eps: float, rng: np.random.Generator):
    grads, best_centers, cost = [], [], 0
    for task in tasks:
        best_x, best_val = None, float("inf")
        for _ in range(probe_points):
            x = proj_ball(rng.standard_normal(task.d), task.radius)
            value = float(task.eval(x))
            cost += 1
            if value < best_val:
                best_x, best_val = x.copy(), value
            g, c = coord_grad(task, x, fd_eps)
            grads.append(g)
            cost += c
        best_centers.append(best_x)
    _, _, vh = np.linalg.svd(np.asarray(grads), full_matrices=False)
    q, _ = np.linalg.qr(vh[:k].T)
    warm = proj_ball(np.mean(np.asarray(best_centers), axis=0), tasks[0].radius)
    return q[:, :k], cost, warm


def projector_err(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.linalg.norm(u @ u.T - v @ v.T, ord=2))


def fd_descent(task: Task, basis: np.ndarray, x0: np.ndarray, budget: int, fd_eps: float, step_size: float = 0.2):
    anchor = proj_ball(x0.copy(), task.radius)
    z = np.zeros(int(basis.shape[1]))
    x = anchor.copy()
    cur = float(task.eval(x))
    initial_value = cur
    best, trace, evals, step = cur, [cur], 1, step_size
    while evals < budget:
        g = np.zeros(int(basis.shape[1]))
        for j in range(int(basis.shape[1])):
            if evals + 2 > budget:
                break
            dvec = basis[:, j]
            xp = proj_ball(x + fd_eps * dvec, task.radius)
            xn = proj_ball(x - fd_eps * dvec, task.radius)
            fp = float(task.eval(xp))
            best = min(best, fp)
            trace.append(best)
            evals += 1
            fn = float(task.eval(xn))
            best = min(best, fn)
            trace.append(best)
            evals += 1
            g[j] = (fp - fn) / (2.0 * fd_eps)
        if evals >= budget:
            break
        z_new = z - step * g
        x_new = proj_ball(anchor + basis @ z_new, task.radius)
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
    return {"best_value": best, "best_trace": trace[:budget], "initial_value": initial_value}


def rand_dir_descent(task: Task, x0: np.ndarray, budget: int, fd_eps: float, rng: np.random.Generator, step_size: float):
    x = proj_ball(x0.copy(), task.radius)
    cur = float(task.eval(x))
    initial_value = cur
    best, trace, evals, step = cur, [cur], 1, step_size
    while evals < budget:
        v = rng.standard_normal(task.d)
        v /= max(float(np.linalg.norm(v)), 1e-12)
        if evals + 2 > budget:
            break
        xp = proj_ball(x + fd_eps * v, task.radius)
        xn = proj_ball(x - fd_eps * v, task.radius)
        fp = float(task.eval(xp))
        best = min(best, fp)
        trace.append(best)
        evals += 1
        fn = float(task.eval(xn))
        best = min(best, fn)
        trace.append(best)
        evals += 1
        g = ((fp - fn) / (2.0 * fd_eps)) * v
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
            step = step_size if step < 1e-4 else step * 0.5
    if len(trace) < budget:
        trace.extend([best] * (budget - len(trace)))
    return {"best_value": best, "best_trace": trace[:budget], "initial_value": initial_value}


def first_hit(trace, eps):
    for i, v in enumerate(trace, start=1):
        if float(v) <= eps:
            return i
    return None


def summarize(runs, thresholds, budget: int, history_share: float):
    vals = [float(r["best_value"]) for r in runs]
    out = {
        "n": len(runs),
        "mean_simple_regret": float(np.mean(vals)),
        "std_simple_regret": float(np.std(vals)),
        "thresholds": {},
    }
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


def tune_random_direction_step(cases, candidates, fd_eps: float, mode: str):
    scores = {}
    for step in candidates:
        vals = []
        for idx, case in enumerate(cases):
            x_init = case["x0"] if mode == "scratch" else case["history_probe_init"]
            run = rand_dir_descent(
                case["task"],
                x_init,
                budget=case["budget"],
                fd_eps=fd_eps,
                rng=np.random.default_rng(case["seed"] + 10000 * idx + int(1000 * step)),
                step_size=step,
            )
            vals.append(run["best_value"])
        scores[str(step)] = float(np.mean(vals))
    best_step = min(candidates, key=lambda s: scores[str(s)])
    return float(best_step), scores


def task_record(method_name: str, run: dict, thresholds, family_seed: int, task_idx: int, common_start_value: float, start_type: str, step_size=None):
    return {
        "method": method_name,
        "family_seed": int(family_seed),
        "task_idx": int(task_idx),
        "start_type": start_type,
        "common_start_value": float(common_start_value),
        "initial_value": float(run["initial_value"]),
        "best_value": float(run["best_value"]),
        "threshold_hits": {str(eps): first_hit(run["best_trace"], eps) for eps in thresholds},
        "step_size": None if step_size is None else float(step_size),
    }


def run_suite(args):
    suite = []
    for d in args.d_values:
        families, costs, errs = [], [], []
        for family_seed in range(args.family_seeds):
            rng = np.random.default_rng(8100 + d * 97 + family_seed)
            u_true = ortho(rng, d, args.k)
            history = [make_task(rng, u_true, args.condition_number, args.ambient_curvature, args.radius) for _ in range(args.history_tasks)]
            validation = [make_task(rng, u_true, args.condition_number, args.ambient_curvature, args.radius) for _ in range(args.validation_tasks)]
            test = [make_task(rng, u_true, args.condition_number, args.ambient_curvature, args.radius) for _ in range(args.test_tasks)]
            u_est, hist_cost, history_probe_init = collect_history_observations(history, args.k, args.probe_points, args.fd_eps, rng)
            x0_validation = [proj_ball(rng.standard_normal(d), args.radius) for _ in validation]
            x0_test = [proj_ball(rng.standard_normal(d), args.radius) for _ in test]
            families.append({
                "family_seed": family_seed,
                "u_true": u_true,
                "u_est": u_est,
                "history": history,
                "validation": validation,
                "test": test,
                "x0_validation": x0_validation,
                "x0_test": x0_test,
                "history_probe_init": history_probe_init,
            })
            costs.append(hist_cost)
            errs.append(projector_err(u_true, u_est))

        validation_cases = []
        for fam in families:
            for idx, task in enumerate(fam["validation"]):
                validation_cases.append({
                    "task": task,
                    "x0": fam["x0_validation"][idx],
                    "history_probe_init": fam["history_probe_init"],
                    "budget": args.budget,
                    "seed": 440000 + d * 1000 + fam["family_seed"] * 100 + idx,
                })
        scratch_step, scratch_scores = tune_random_direction_step(validation_cases, args.random_direction_steps, args.fd_eps, mode="scratch")
        history_step, history_scores = tune_random_direction_step(validation_cases, args.random_direction_steps, args.fd_eps, mode="history")

        rows = {k: [] for k in ["scratch_coordinate_fd", "scratch_random_direction", "history_probe_init_random_direction", "estimated_subspace", "oracle_subspace"]}
        task_records = []
        for fam in families:
            for task_idx, task in enumerate(fam["test"]):
                x0 = fam["x0_test"][task_idx]
                common_start_value = float(task.eval(x0))
                base_seed = 990000 + d * 1000 + fam["family_seed"] * 100 + task_idx
                run_coord = fd_descent(task, np.eye(d), x0, args.budget, args.fd_eps)
                run_rand = rand_dir_descent(task, x0, args.budget, args.fd_eps, np.random.default_rng(base_seed), scratch_step)
                run_hist = rand_dir_descent(task, fam["history_probe_init"], args.budget, args.fd_eps, np.random.default_rng(base_seed + 1), history_step)
                run_est = fd_descent(task, fam["u_est"], x0, args.budget, args.fd_eps)
                run_oracle = fd_descent(task, fam["u_true"], x0, args.budget, args.fd_eps)
                rows["scratch_coordinate_fd"].append(run_coord)
                rows["scratch_random_direction"].append(run_rand)
                rows["history_probe_init_random_direction"].append(run_hist)
                rows["estimated_subspace"].append(run_est)
                rows["oracle_subspace"].append(run_oracle)
                task_records.extend([
                    task_record("scratch_coordinate_fd", run_coord, args.thresholds, fam["family_seed"], task_idx, common_start_value, "common_start"),
                    task_record("scratch_random_direction", run_rand, args.thresholds, fam["family_seed"], task_idx, common_start_value, "common_start", scratch_step),
                    task_record("history_probe_init_random_direction", run_hist, args.thresholds, fam["family_seed"], task_idx, common_start_value, "history_probe_init", history_step),
                    task_record("estimated_subspace", run_est, args.thresholds, fam["family_seed"], task_idx, common_start_value, "common_start"),
                    task_record("oracle_subspace", run_oracle, args.thresholds, fam["family_seed"], task_idx, common_start_value, "common_start"),
                ])

        item = {
            "d": int(d),
            "history_query_cost_mean": float(np.mean(costs)),
            "projector_error_mean": float(np.mean(errs)),
            "validation": {
                "scratch_random_direction": {"best_step": scratch_step, "candidate_mean_regret": scratch_scores},
                "history_probe_init_random_direction": {"best_step": history_step, "candidate_mean_regret": history_scores},
            },
            "methods": {},
            "task_records": task_records,
        }
        for m in args.amortized_m_values:
            item["methods"][f"M={m}"] = {
                "scratch_coordinate_fd": summarize(rows["scratch_coordinate_fd"], args.thresholds, args.budget, 0.0),
                "scratch_random_direction": summarize(rows["scratch_random_direction"], args.thresholds, args.budget, 0.0),
                "history_probe_init_random_direction": summarize(rows["history_probe_init_random_direction"], args.thresholds, args.budget, float(np.mean(costs)) / m),
                "estimated_subspace": summarize(rows["estimated_subspace"], args.thresholds, args.budget, float(np.mean(costs)) / m),
                "oracle_subspace": summarize(rows["oracle_subspace"], args.thresholds, args.budget, 0.0),
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
    ap.add_argument("--validation_tasks", type=int, default=6)
    ap.add_argument("--test_tasks", type=int, default=12)
    ap.add_argument("--probe_points", type=int, default=2)
    ap.add_argument("--budget", type=int, default=180)
    ap.add_argument("--condition_number", type=float, default=12.0)
    ap.add_argument("--ambient_curvature", type=float, default=0.02)
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--fd_eps", type=float, default=1e-3)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.1, 0.02, 0.005])
    ap.add_argument("--amortized_m_values", nargs="+", type=int, default=[1, 5, 20])
    ap.add_argument("--random_direction_steps", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.4, 0.8, 1.6])
    args = ap.parse_args()
    out = {"config": vars(args), "suite": run_suite(args)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(os.path.abspath(args.out), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print("=" * 80)
    print("STRICT SHARED-SUBSPACE V4")
    print("=" * 80)
    for row in out["suite"]:
        m = row["methods"]["M=5"]
        print(
            f"d={row['d']} coord={m['scratch_coordinate_fd']['mean_simple_regret']:.4f} "
            f"rand={m['scratch_random_direction']['mean_simple_regret']:.4f} "
            f"hist_probe={m['history_probe_init_random_direction']['mean_simple_regret']:.4f} "
            f"est={m['estimated_subspace']['mean_simple_regret']:.4f} "
            f"oracle={m['oracle_subspace']['mean_simple_regret']:.4f}"
        )
    print(f"Wrote: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
