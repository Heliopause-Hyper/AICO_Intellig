from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


REPO_ROOT = "/home/ycl/AICO-Intellig"
DEFAULT_OUT = os.path.join(REPO_ROOT, "results", "complexity_study", "shared_subspace_mismatch_large_v2.json")


@dataclass
class AmbientQuadraticTask:
    basis: np.ndarray
    hessian: np.ndarray
    optimum_latent: np.ndarray
    ambient_curvature: float

    @property
    def d(self) -> int:
        return int(self.basis.shape[0])

    @property
    def k(self) -> int:
        return int(self.basis.shape[1])

    def evaluate(self, x: np.ndarray) -> float:
        optimum = self.basis @ self.optimum_latent
        delta = x - optimum
        proj = self.basis.T @ delta
        ridge = 0.5 * float(proj.T @ self.hessian @ proj)
        # Weak ambient curvature keeps the optimum unique in the full space.
        ambient = 0.5 * float(self.ambient_curvature) * float(delta.T @ delta)
        return ridge + ambient


def _orthonormal_basis(rng: np.random.Generator, d: int, k: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((d, k)))
    return q[:, :k]


def _complement_basis(u: np.ndarray) -> np.ndarray:
    d = int(u.shape[0])
    q, _ = np.linalg.qr(np.concatenate([u, np.eye(d)], axis=1))
    return q[:, u.shape[1] :]


def _rotate_basis(u_true: np.ndarray, complement: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(float(angle_deg)) < 1e-12:
        return u_true.copy()
    theta = math.radians(float(angle_deg))
    mixed = math.cos(theta) * u_true + math.sin(theta) * complement[:, : u_true.shape[1]]
    q, _ = np.linalg.qr(mixed)
    return q[:, : u_true.shape[1]]


def _make_task(
    rng: np.random.Generator,
    basis: np.ndarray,
    condition_number: float,
    ambient_curvature: float,
) -> AmbientQuadraticTask:
    k = int(basis.shape[1])
    eigs = np.exp(rng.uniform(0.0, math.log(condition_number), size=k))
    hessian = np.diag(eigs)
    optimum_latent = rng.uniform(-0.75, 0.75, size=k)
    return AmbientQuadraticTask(
        basis=basis,
        hessian=hessian,
        optimum_latent=optimum_latent,
        ambient_curvature=float(ambient_curvature),
    )


def _coordinate_gradient(task: AmbientQuadraticTask, x: np.ndarray, eps: float) -> Tuple[np.ndarray, int]:
    grad = np.zeros(task.d, dtype=float)
    for idx in range(task.d):
        step = np.zeros(task.d, dtype=float)
        step[idx] = eps
        grad[idx] = (task.evaluate(np.clip(x + step, -1.0, 1.0)) - task.evaluate(np.clip(x - step, -1.0, 1.0))) / (2.0 * eps)
    return grad, 2 * task.d


def _estimate_basis_from_history(
    tasks: Iterable[AmbientQuadraticTask],
    k: int,
    probe_points: int,
    fd_eps: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, int]:
    grads: List[np.ndarray] = []
    query_cost = 0
    for task in tasks:
        for _ in range(int(probe_points)):
            x = rng.uniform(-1.0, 1.0, size=task.d)
            grad, cost = _coordinate_gradient(task, x, eps=fd_eps)
            grads.append(grad)
            query_cost += cost
    gram = np.asarray(grads, dtype=float)
    _, _, vh = np.linalg.svd(gram, full_matrices=False)
    est = vh[:k].T
    q, _ = np.linalg.qr(est)
    return q[:, :k], query_cost


def _operator_projector_error(u_true: np.ndarray, u_est: np.ndarray) -> float:
    proj_true = u_true @ u_true.T
    proj_est = u_est @ u_est.T
    return float(np.linalg.norm(proj_true - proj_est, ord=2))


def _fd_descent_in_basis(
    task: AmbientQuadraticTask,
    basis: np.ndarray,
    budget: int,
    rng: np.random.Generator,
    fd_eps: float,
    step_size: float = 0.20,
) -> Dict[str, object]:
    p = int(basis.shape[1])
    z = rng.uniform(-1.0, 1.0, size=p)
    x = np.clip(basis @ z, -1.0, 1.0)
    current = float(task.evaluate(x))
    best = float(current)
    trace = [best]
    evals = 1

    while evals < int(budget):
        grad = np.zeros(p, dtype=float)
        for j in range(p):
            if evals + 2 > int(budget):
                break
            direction = basis[:, j]
            x_pos = np.clip(x + float(fd_eps) * direction, -1.0, 1.0)
            x_neg = np.clip(x - float(fd_eps) * direction, -1.0, 1.0)
            f_pos = float(task.evaluate(x_pos))
            best = min(best, f_pos)
            trace.append(best)
            evals += 1
            f_neg = float(task.evaluate(x_neg))
            best = min(best, f_neg)
            trace.append(best)
            evals += 1
            grad[j] = (f_pos - f_neg) / (2.0 * float(fd_eps))
        if evals >= int(budget):
            break
        candidate_z = np.clip(z - float(step_size) * grad, -1.0, 1.0)
        candidate_x = np.clip(basis @ candidate_z, -1.0, 1.0)
        candidate_value = float(task.evaluate(candidate_x))
        best = min(best, candidate_value)
        trace.append(best)
        evals += 1
        if candidate_value <= current:
            z = candidate_z
            x = candidate_x
            current = candidate_value
        else:
            step_size *= 0.5
            if step_size < 1e-4:
                step_size = 0.20

    if len(trace) < int(budget):
        trace.extend([best] * (int(budget) - len(trace)))
    return {"best_value": float(best), "best_trace": trace[: int(budget)]}


def _first_hit(trace: List[float], threshold: float) -> int | None:
    for idx, value in enumerate(trace, start=1):
        if float(value) <= float(threshold):
            return int(idx)
    return None


def _summarize_runs(runs: List[Dict[str, object]], thresholds: Iterable[float]) -> Dict[str, object]:
    values = [float(run["best_value"]) for run in runs]
    summary = {
        "n": len(runs),
        "mean_simple_regret": float(np.mean(values)) if values else 0.0,
        "median_simple_regret": float(np.median(values)) if values else 0.0,
        "max_simple_regret": float(np.max(values)) if values else 0.0,
    }
    by_threshold = {}
    for eps in thresholds:
        hits = [_first_hit(run["best_trace"], eps) for run in runs]
        success = [h for h in hits if h is not None]
        by_threshold[str(eps)] = {
            "success_rate": float(len(success) / max(1, len(hits))),
            "mean_queries_if_hit": float(np.mean(success)) if success else None,
        }
    summary["thresholds"] = by_threshold
    return summary


def _run_condition(
    d: int,
    k: int,
    angle_deg: float,
    ambient_curvature: float,
    probe_points: int,
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
            5000
            + 17 * int(d)
            + 31 * int(history_tasks)
            + 43 * int(probe_points)
            + int(round(angle_deg * 10))
            + int(round(ambient_curvature * 10000))
            + family_seed
        )
        u_true = _orthonormal_basis(rng, d=int(d), k=int(k))
        complement = _complement_basis(u_true)
        history_basis = u_true
        test_basis = _rotate_basis(u_true, complement, angle_deg=float(angle_deg))
        history = [
            _make_task(rng, history_basis, condition_number=condition_number, ambient_curvature=float(ambient_curvature))
            for _ in range(int(history_tasks))
        ]
        test = [
            _make_task(rng, test_basis, condition_number=condition_number, ambient_curvature=float(ambient_curvature))
            for _ in range(int(test_tasks))
        ]
        u_est, history_cost = _estimate_basis_from_history(
            history,
            k=int(k),
            probe_points=int(probe_points),
            fd_eps=float(fd_eps),
            rng=rng,
        )
        projector_errors.append(_operator_projector_error(test_basis, u_est))
        history_costs.append(int(history_cost))
        full_basis = np.eye(int(d))

        for task_idx, task in enumerate(test):
            seed_base = 1100000 + family_seed * 1000 + task_idx * 10
            scratch_runs.append(
                _fd_descent_in_basis(task, full_basis, budget=budget, rng=np.random.default_rng(seed_base), fd_eps=fd_eps)
            )
            oracle_runs.append(
                _fd_descent_in_basis(task, test_basis, budget=budget, rng=np.random.default_rng(seed_base + 1), fd_eps=fd_eps)
            )
            estimated_runs.append(
                _fd_descent_in_basis(task, u_est, budget=budget, rng=np.random.default_rng(seed_base + 2), fd_eps=fd_eps)
            )

    return {
        "d": int(d),
        "k": int(k),
        "angle_deg": float(angle_deg),
        "ambient_curvature": float(ambient_curvature),
        "probe_points_per_task": int(probe_points),
        "family_seeds": int(family_seeds),
        "history_tasks_per_family": int(history_tasks),
        "test_tasks_per_family": int(test_tasks),
        "budget": int(budget),
        "history_query_cost_mean": float(np.mean(history_costs)) if history_costs else 0.0,
        "projector_error_mean": float(np.mean(projector_errors)) if projector_errors else 0.0,
        "scratch": _summarize_runs(scratch_runs, thresholds),
        "oracle_subspace": _summarize_runs(oracle_runs, thresholds),
        "estimated_subspace": _summarize_runs(estimated_runs, thresholds),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run shared-subspace mismatch sweep v2.")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--family_seeds", type=int, default=20)
    ap.add_argument("--history_tasks", type=int, default=8)
    ap.add_argument("--test_tasks", type=int, default=30)
    ap.add_argument("--budget", type=int, default=180)
    ap.add_argument("--probe_points", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--angles", nargs="+", type=float, default=[0.0, 5.0, 15.0, 30.0, 45.0])
    ap.add_argument("--ambient_curvatures", nargs="+", type=float, default=[0.005, 0.01, 0.02, 0.05])
    ap.add_argument("--condition_number", type=float, default=12.0)
    ap.add_argument("--fd_eps", type=float, default=1e-3)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.1, 0.02, 0.005])
    args = ap.parse_args()

    rows = []
    for probe_points in args.probe_points:
        for angle_deg in args.angles:
            for ambient_curvature in args.ambient_curvatures:
                rows.append(
                    _run_condition(
                        d=int(args.d),
                        k=int(args.k),
                        angle_deg=float(angle_deg),
                        ambient_curvature=float(ambient_curvature),
                        probe_points=int(probe_points),
                        family_seeds=int(args.family_seeds),
                        history_tasks=int(args.history_tasks),
                        test_tasks=int(args.test_tasks),
                        budget=int(args.budget),
                        condition_number=float(args.condition_number),
                        fd_eps=float(args.fd_eps),
                        thresholds=args.thresholds,
                    )
                )

    summary = {
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
            "condition_number": float(args.condition_number),
            "fd_eps": float(args.fd_eps),
            "thresholds": [float(x) for x in args.thresholds],
        },
        "rows": rows,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(os.path.abspath(args.out), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print("=" * 80)
    print("SHARED-SUBSPACE MISMATCH SWEEP V2")
    print("=" * 80)
    for row in rows[:8]:
        print(
            f"probe={row['probe_points_per_task']} angle={row['angle_deg']:.1f} ambient={row['ambient_curvature']:.3f} "
            f"proj_err={row['projector_error_mean']:.4f} "
            f"regret(est)={row['estimated_subspace']['mean_simple_regret']:.4f}"
        )
    print("")
    print(f"Wrote: {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
