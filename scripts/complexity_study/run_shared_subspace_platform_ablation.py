from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import Dict, Iterable, List

import numpy as np


REPO_ROOT = "/home/ycl/AICO-Intellig"
DEFAULT_OUT = os.path.join(REPO_ROOT, "results", "complexity_study", "shared_subspace_platform_ablation.json")


def _load_strict_v4_module():
    path = os.path.join(REPO_ROOT, "scripts", "complexity_study", "run_shared_subspace_strict_v4.py")
    spec = importlib.util.spec_from_file_location("strict_v4", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load strict_v4 module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STRICT_V4 = _load_strict_v4_module()


def _fd_descent_free(task, basis: np.ndarray, budget: int, fd_eps: float, rng: np.random.Generator, step_size: float = 0.2):
    p = int(basis.shape[1])
    z = rng.uniform(-0.5, 0.5, size=p)
    x = STRICT_V4.proj_ball(basis @ z, task.radius)
    cur = float(task.eval(x))
    best = cur
    trace = [best]
    evals = 1
    step = float(step_size)

    while evals < int(budget):
        g = np.zeros(p, dtype=float)
        for j in range(p):
            if evals + 2 > int(budget):
                break
            dvec = basis[:, j]
            xp = STRICT_V4.proj_ball(x + float(fd_eps) * dvec, task.radius)
            xn = STRICT_V4.proj_ball(x - float(fd_eps) * dvec, task.radius)
            fp = float(task.eval(xp))
            best = min(best, fp)
            trace.append(best)
            evals += 1
            fn = float(task.eval(xn))
            best = min(best, fn)
            trace.append(best)
            evals += 1
            g[j] = (fp - fn) / (2.0 * float(fd_eps))
        if evals >= int(budget):
            break
        z_new = np.clip(z - step * g, -task.radius, task.radius)
        x_new = STRICT_V4.proj_ball(basis @ z_new, task.radius)
        val = float(task.eval(x_new))
        best = min(best, val)
        trace.append(best)
        evals += 1
        if val <= cur:
            z = z_new
            x = x_new
            cur = val
        else:
            step = 0.2 if step < 1e-4 else step * 0.5

    if len(trace) < int(budget):
        trace.extend([best] * (int(budget) - len(trace)))

    return {
        "best_value": float(best),
        "best_trace": trace[: int(budget)],
        "initial_value": float(trace[0]),
    }


def _run_condition(
    d: int,
    k: int,
    family_seeds: int,
    history_tasks: int,
    test_tasks: int,
    probe_points: int,
    budget: int,
    condition_number: float,
    ambient_curvature: float,
    radius: float,
    fd_eps: float,
    thresholds: Iterable[float],
) -> Dict[str, object]:
    projector_errors: List[float] = []
    history_costs: List[int] = []
    rows = {
        "oracle_anchor": [],
        "oracle_free": [],
        "estimated_anchor": [],
        "estimated_free": [],
    }

    for family_seed in range(int(family_seeds)):
        rng = np.random.default_rng(
            9100
            + 97 * int(d)
            + 29 * int(history_tasks)
            + 43 * int(probe_points)
            + 53 * int(budget)
            + int(round(1000.0 * float(fd_eps)))
            + int(round(100.0 * float(radius)))
            + family_seed
        )
        u_true = STRICT_V4.ortho(rng, int(d), int(k))
        history = [
            STRICT_V4.make_task(rng, u_true, float(condition_number), float(ambient_curvature), float(radius))
            for _ in range(int(history_tasks))
        ]
        test = [
            STRICT_V4.make_task(rng, u_true, float(condition_number), float(ambient_curvature), float(radius))
            for _ in range(int(test_tasks))
        ]
        u_est, hist_cost, _ = STRICT_V4.collect_history_observations(
            history,
            int(k),
            int(probe_points),
            float(fd_eps),
            rng,
        )
        projector_errors.append(STRICT_V4.projector_err(u_true, u_est))
        history_costs.append(int(hist_cost))
        starts = [STRICT_V4.proj_ball(rng.standard_normal(int(d)), float(radius)) for _ in range(int(test_tasks))]

        for task_idx, task in enumerate(test):
            seed_base = 1200000 + 1000 * family_seed + 10 * task_idx
            x0 = starts[task_idx]
            rows["oracle_anchor"].append(STRICT_V4.fd_descent(task, u_true, x0, int(budget), float(fd_eps)))
            rows["estimated_anchor"].append(STRICT_V4.fd_descent(task, u_est, x0, int(budget), float(fd_eps)))
            rows["oracle_free"].append(
                _fd_descent_free(task, u_true, int(budget), float(fd_eps), np.random.default_rng(seed_base + 1))
            )
            rows["estimated_free"].append(
                _fd_descent_free(task, u_est, int(budget), float(fd_eps), np.random.default_rng(seed_base + 2))
            )

    return {
        "d": int(d),
        "k": int(k),
        "family_seeds": int(family_seeds),
        "history_tasks_per_family": int(history_tasks),
        "test_tasks_per_family": int(test_tasks),
        "probe_points_per_task": int(probe_points),
        "budget": int(budget),
        "radius": float(radius),
        "fd_eps": float(fd_eps),
        "history_query_cost_mean": float(np.mean(history_costs)),
        "projector_error_mean": float(np.mean(projector_errors)),
        "oracle_anchor": STRICT_V4.summarize(rows["oracle_anchor"], thresholds, int(budget), 0.0),
        "oracle_free": STRICT_V4.summarize(rows["oracle_free"], thresholds, int(budget), 0.0),
        "estimated_anchor": STRICT_V4.summarize(rows["estimated_anchor"], thresholds, int(budget), 0.0),
        "estimated_free": STRICT_V4.summarize(rows["estimated_free"], thresholds, int(budget), 0.0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run platform ablations for strict shared-subspace optimization.")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--d_values", nargs="+", type=int, default=[64, 256])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--family_seeds", type=int, default=24)
    ap.add_argument("--history_tasks", type=int, default=8)
    ap.add_argument("--test_tasks", type=int, default=30)
    ap.add_argument("--probe_points", type=int, default=2)
    ap.add_argument("--budgets", nargs="+", type=int, default=[180, 360, 720])
    ap.add_argument("--radii", nargs="+", type=float, default=[0.5, 1.0, 1.5])
    ap.add_argument("--fd_eps_values", nargs="+", type=float, default=[1e-2, 1e-3, 1e-4])
    ap.add_argument("--condition_number", type=float, default=12.0)
    ap.add_argument("--ambient_curvature", type=float, default=0.01)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.1, 0.02, 0.005])
    args = ap.parse_args()

    rows = []
    for d in args.d_values:
        for budget in args.budgets:
            for radius in args.radii:
                for fd_eps in args.fd_eps_values:
                    rows.append(
                        _run_condition(
                            d=int(d),
                            k=int(args.k),
                            family_seeds=int(args.family_seeds),
                            history_tasks=int(args.history_tasks),
                            test_tasks=int(args.test_tasks),
                            probe_points=int(args.probe_points),
                            budget=int(budget),
                            condition_number=float(args.condition_number),
                            ambient_curvature=float(args.ambient_curvature),
                            radius=float(radius),
                            fd_eps=float(fd_eps),
                            thresholds=args.thresholds,
                        )
                    )

    payload = {
        "config": {
            "d_values": [int(x) for x in args.d_values],
            "k": int(args.k),
            "family_seeds": int(args.family_seeds),
            "history_tasks": int(args.history_tasks),
            "test_tasks": int(args.test_tasks),
            "probe_points": int(args.probe_points),
            "budgets": [int(x) for x in args.budgets],
            "radii": [float(x) for x in args.radii],
            "fd_eps_values": [float(x) for x in args.fd_eps_values],
            "condition_number": float(args.condition_number),
            "ambient_curvature": float(args.ambient_curvature),
            "thresholds": [float(x) for x in args.thresholds],
        },
        "rows": rows,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(os.path.abspath(args.out), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print("=" * 80)
    print("SHARED-SUBSPACE PLATFORM ABLATION")
    print("=" * 80)
    for row in rows[:8]:
        print(
            f"d={row['d']} budget={row['budget']} radius={row['radius']:.2f} fd_eps={row['fd_eps']:.0e} "
            f"est(anchor/free)={row['estimated_anchor']['mean_simple_regret']:.4f}/{row['estimated_free']['mean_simple_regret']:.4f} "
            f"oracle(anchor/free)={row['oracle_anchor']['mean_simple_regret']:.4f}/{row['oracle_free']['mean_simple_regret']:.4f}"
        )
    print("")
    print(f"Wrote: {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
