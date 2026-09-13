import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _optimizer_count() -> int:
    sys.path.append(_repo_root())
    from src.problem_config import get_optimizer_config

    cfg = get_optimizer_config()
    n = 0
    for _, meta in cfg.items():
        n += len(meta.get("methods", []) or [])
    return int(n)


def _make_config(
    instances_dataset_path: str,
    max_iterations: int,
    time_limit_s: float,
    repeats: int,
    out_dir: str,
    shard_count: int,
    shard_index: int,
    resume: bool,
) -> Dict[str, Any]:
    seeds = list(range(int(repeats)))
    return {
        "resume": bool(resume),
        "stable_instance_ids": True,
        "instances_from_dataset_path": str(instances_dataset_path),
        "instances_sample_seed": 0,
        "instances_sample_clean": 0,
        "instances_sample_hard": 0,
        "budget": {"max_iterations": int(max_iterations), "time_limit_s": float(time_limit_s)},
        "seeds": seeds,
        "auto": True,
        "candidates": [],
        "validate_instances": False,
        "sns_family_probe_endpoints": False,
        "shard_count": int(shard_count),
        "shard_index": int(shard_index),
        "meta": {
            "script": "run_sns_family_full_sweep.py",
            "created_at": datetime.now().isoformat(),
            "optimizer_total_config": _optimizer_count(),
        },
    }


def _run_one_shard(cfg: Dict[str, Any], out_dir: str) -> int:
    os.makedirs(out_dir, exist_ok=True)
    cfg_path = os.path.join(out_dir, "full_sweep_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    script_path = os.path.join(_repo_root(), "scripts", "generate_classification_data.py")
    cmd = [sys.executable, script_path, "--config", cfg_path, "--out_dir", out_dir]
    print("CMD:", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--instances_dataset",
        default=os.path.join(_repo_root(), "results", "sns_family_protocol_v1_instances150", "dataset.jsonl"),
    )
    p.add_argument("--out_dir", default="")
    p.add_argument("--max_iterations", type=int, default=30)
    p.add_argument("--time_limit_s", type=float, default=60.0)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()

    if not args.out_dir:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = os.path.join(_repo_root(), "results", f"sns_v1_full_sweep_{stamp}")

    instances_dataset = os.path.abspath(str(args.instances_dataset))
    if not os.path.exists(instances_dataset):
        raise SystemExit(f"instances_dataset not found: {instances_dataset}")

    workers = max(1, int(args.workers))
    if workers == 1:
        cfg = _make_config(
            instances_dataset_path=instances_dataset,
            max_iterations=args.max_iterations,
            time_limit_s=args.time_limit_s,
            repeats=args.repeats,
            out_dir=args.out_dir,
            shard_count=1,
            shard_index=0,
            resume=bool(args.resume),
        )
        return int(_run_one_shard(cfg, args.out_dir))

    base_dir = os.path.abspath(args.out_dir)
    os.makedirs(base_dir, exist_ok=True)
    procs: List[subprocess.Popen] = []
    for shard_index in range(workers):
        shard_dir = os.path.join(base_dir, f"shard_{shard_index:02d}")
        cfg = _make_config(
            instances_dataset_path=instances_dataset,
            max_iterations=args.max_iterations,
            time_limit_s=args.time_limit_s,
            repeats=args.repeats,
            out_dir=shard_dir,
            shard_count=workers,
            shard_index=shard_index,
            resume=bool(args.resume),
        )
        os.makedirs(shard_dir, exist_ok=True)
        cfg_path = os.path.join(shard_dir, "full_sweep_config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        script_path = os.path.join(_repo_root(), "scripts", "generate_classification_data.py")
        cmd = [sys.executable, script_path, "--config", cfg_path, "--out_dir", shard_dir]
        print("CMD:", " ".join(cmd), flush=True)
        procs.append(subprocess.Popen(cmd))

    exit_code = 0
    for proc in procs:
        rc = proc.wait()
        if rc != 0:
            exit_code = int(rc)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

