from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import Dict, Iterable, List

import numpy as np


REPO_ROOT = "/home/ycl/AICO-Intellig"
DEFAULT_OUT = os.path.join(REPO_ROOT, "results", "complexity_study", "shared_subspace_mismatch_noise.json")


def _load_mismatch_v2_module():
    path = os.path.join(REPO_ROOT, "scripts", "complexity_study", "run_shared_subspace_mismatch_v2.py")
    spec = importlib.util.spec_from_file_location("mismatch_v2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load mismatch_v2 module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MISMATCH_V2 = _load_mismatch_v2_module()


def _estimate_basis_from_history_noisy(
    tasks,
    k: int,
    probe_points: int,
    fd_eps: float,
    noise_std: float,
    rng: np.random.Generator,
):
    grads = []
    query_cost = 0
    for task in tasks:
        for _ in range(int(probe_points)):
            x = rng.uniform(-1.0, 1.0, size=task.d)
            grad, cost = MISMATCH_V2._coordinate_gradient(task, x, eps=float(fd_eps))
            if float(noise_std) > 0.0:
                grad = grad + rng.normal(0.0, float(noise_std), size=grad.shape)
            grads.append(grad)
            query_cost += cost
    gram = np.asarray(grads, dtype=float)
    _, _, vh = np.linalg.svd(gram, full_matrices=False)
    est = vh[: int(k)].T
    q, _ = np.linalg.qr(est)
    return q[:, : int(k)], query_cost


def _run_condition(
    d: int,
    k: int,
    angle_deg: float,
    ambient_curvature: float,
    probe_points: int,
    noise_std: float,
    family_seeds: int,
    history_tasks: int,
    test_tasks: int,
    budget: int,
    condition_number: float,
    fd_eps: float,
    thresholds: Iterable[float],
) -> Dict[str, object]:
    scratch_runs: List[Dict[str, object]] = []
    oracle_runs: List[Dict[str, object]] = []
    estimated_runs: List[Dict[str, object]] = []
    projector_errors: List[float] = []
    history_costs: List[int] = []

    for family_seed in range(int(family_seeds)):
        rng = np.random.default_rng(
            15000
            + 17 * int(d)
            + 31 * int(history_tasks)
            + 43 * int(probe_points)
            + int(round(10.0 * float(angle_deg)))
            + int(round(10000.0 * float(ambient_curvature)))
            + int(round(1000.0 * float(noise_std)))
            + family_seed
        )
        u_true = MISMATCH_V2._orthonormal_basis(rng, d=int(d), k=int(k))
        complement = MISMATCH_V2._complement_basis(u_true)
        history_basis = u_true
        test_basis = MISMATCH_V2._rotate_basis(u_true, complement, angle_deg=float(angle_deg))
        history = [
            MISMATCH_V2._make_task(
                rng,
                history_basis,
                condition_number=float(condition_number),
                ambient_curvature=float(ambient_curvature),
            )
            for _ in range(int(history_tasks))
        ]
        test = [
            MISMATCH_V2._make_task(
                rng,
                test_basis,
                condition_number=float(condition_number),
                ambient_curvature=float(ambient_curvature),
            )
            for _ in range(int(test_tasks))
        ]
        u_est, history_cost = _estimate_basis_from_history_noisy(
            history,
            k=int(k),
            probe_points=int(probe_points),
            fd_eps=float(fd_eps),
            noise_std=float(noise_std),
            rng=rng,
        )
        projector_errors.append(MISMATCH_V2._operator_projector_error(test_basis, u_est))
        history_costs.append(int(history_cost))
        full_basis = np.eye(int(d))

        for task_idx, task in enumerate(test):
            seed_base = 1600000 + family_seed * 1000 + task_idx * 10
            scratch_runs.append(
                MISMATCH_V2._fd_descent_in_basis(
                    task,
                    full_basis,
                    budget=int(budget),
                    rng=np.random.default_rng(seed_base),
                    fd_eps=float(fd_eps),
                )
            )
            oracle_runs.append(
                MISMATCH_V2._fd_descent_in_basis(
                    task,
                    test_basis,
                    budget=int(budget),
                    rng=np.random.default_rng(seed_base + 1),
                    fd_eps=float(fd_eps),
                )
            )
            estimated_runs.append(
                MISMATCH_V2._fd_descent_in_basis(
                    task,
                    u_est,
                    budget=int(budget),
                    rng=np.random.default_rng(seed_base + 2),
                    fd_eps=float(fd_eps),
                )
            )

    return {
        "d": int(d),
        "k": int(k),
        "angle_deg": float(angle_deg),
        "ambient_curvature": float(ambient_curvature),
        "probe_points_per_task": int(probe_points),
        "noise_std": float(noise_std),
        "family_seeds": int(family_seeds),
        "history_tasks_per_family": int(history_tasks),
        "test_tasks_per_family": int(test_tasks),
        "budget": int(budget),
        "history_query_cost_mean": float(np.mean(history_costs)),
        "projector_error_mean": float(np.mean(projector_errors)),
        "scratch": MISMATCH_V2._summarize_runs(scratch_runs, thresholds),
        "oracle_subspace": MISMATCH_V2._summarize_runs(oracle_runs, thresholds),
        "estimated_subspace": MISMATCH_V2._summarize_runs(estimated_runs, thresholds),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run mismatch plus noisy-history shared-subspace sweeps.")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--family_seeds", type=int, default=24)
    ap.add_argument("--history_tasks", type=int, default=8)
    ap.add_argument("--test_tasks", type=int, default=30)
    ap.add_argument("--budget", type=int, default=180)
    ap.add_argument("--probe_points", nargs="+", type=int, default=[2, 4])
    ap.add_argument("--angles", nargs="+", type=float, default=[0.0, 15.0, 30.0, 45.0])
    ap.add_argument("--ambient_curvatures", nargs="+", type=float, default=[0.01, 0.05])
    ap.add_argument("--noise_stds", nargs="+", type=float, default=[0.0, 0.01, 0.05, 0.1])
    ap.add_argument("--condition_number", type=float, default=12.0)
    ap.add_argument("--fd_eps", type=float, default=1e-3)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.1, 0.02, 0.005])
    args = ap.parse_args()

    rows = []
    for probe_points in args.probe_points:
        for angle_deg in args.angles:
            for ambient_curvature in args.ambient_curvatures:
                for noise_std in args.noise_stds:
                    rows.append(
                        _run_condition(
                            d=int(args.d),
                            k=int(args.k),
                            angle_deg=float(angle_deg),
                            ambient_curvature=float(ambient_curvature),
                            probe_points=int(probe_points),
                            noise_std=float(noise_std),
                            family_seeds=int(args.family_seeds),
                            history_tasks=int(args.history_tasks),
                            test_tasks=int(args.test_tasks),
                            budget=int(args.budget),
                            condition_number=float(args.condition_number),
                            fd_eps=float(args.fd_eps),
                            thresholds=args.thresholds,
                        )
                    )

    payload = {
        "config": {
            "d": int(args.d),
            "k": int(args.k),
            "family_seeds": int(args.family_seeds),
            "history_tasks": int(args.history_tasks),
            "test_tasks": int(args.test_tasks),
            "budget": int(args.budget),
            "probe_points": [int(x) for x in args.probe_points],
            "angles": [float(x) for x in args.angles],
            "ambient_curvatures": [float(x) for x in args.ambient_curvatures],
            "noise_stds": [float(x) for x in args.noise_stds],
            "condition_number": float(args.condition_number),
            "fd_eps": float(args.fd_eps),
            "thresholds": [float(x) for x in args.thresholds],
            "noise_model": "gaussian additive noise on history coordinate gradients before SVD basis estimation",
        },
        "rows": rows,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(os.path.abspath(args.out), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print("=" * 80)
    print("SHARED-SUBSPACE MISMATCH + NOISE SWEEP")
    print("=" * 80)
    for row in rows[:8]:
        print(
            f"probe={row['probe_points_per_task']} angle={row['angle_deg']:.1f} ambient={row['ambient_curvature']:.3f} "
            f"noise={row['noise_std']:.3f} proj_err={row['projector_error_mean']:.4f} "
            f"regret(est)={row['estimated_subspace']['mean_simple_regret']:.4f}"
        )
    print("")
    print(f"Wrote: {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
