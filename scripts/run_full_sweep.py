import argparse
import json
import os
import subprocess
import sys
import time
from typing import List, Dict, Tuple


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _parse_int_list(s: str) -> List[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_str_list(s: str) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _load_all_candidates() -> List[Dict[str, str]]:
    sys.path.append(_repo_root())
    from src.problem_config import get_optimizer_config

    cfg = get_optimizer_config()
    out: List[Dict[str, str]] = []
    for optimizer, meta in cfg.items():
        for method in meta.get("methods", []) or []:
            out.append({"optimizer": optimizer, "method": method})
    return out


def _pick_candidate_pool(pool: str) -> List[Dict[str, str]]:
    pool = str(pool).strip()
    all_cands = _load_all_candidates()
    all_set = {(c["optimizer"], c["method"]) for c in all_cands}

    if pool in {"all", ""}:
        return all_cands

    if pool == "9":
        preferred = [
            ("SciPy", "DE"),
            ("SciPy", "Powell"),
            ("SciPy", "Nelder-Mead"),
            ("Optuna", "TPE"),
            ("Optuna", "CMA-ES"),
            ("Hyperopt", "TPE"),
            ("Hyperopt", "Random"),
            ("Nevergrad", "NGOpt"),
            ("Nevergrad", "PSO"),
        ]
        out = [{"optimizer": o, "method": m} for (o, m) in preferred if (o, m) in all_set]
        if len(out) < 9:
            for c in all_cands:
                if (c["optimizer"], c["method"]) not in {(x["optimizer"], x["method"]) for x in out}:
                    out.append(c)
                if len(out) >= 9:
                    break
        return out

    raise ValueError(f"未知算法池: {pool}（可选: all, 9）")


def _merge_jsonl_dedup_by_instance_id(out_path: str, in_paths: List[str]):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    seen = set()
    with open(out_path, "w", encoding="utf-8") as out_f:
        for p in in_paths:
            if not os.path.exists(p):
                continue
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        iid = row.get("instance_id")
                        if not iid:
                            continue
                        if iid in seen:
                            continue
                        seen.add(iid)
                        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    except Exception:
                        continue


def _merge_runs_jsonl(out_path: str, in_paths: List[str]):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out_f:
        for p in in_paths:
            if not os.path.exists(p):
                continue
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out_f.write(line)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=os.path.join(_repo_root(), "results", "training_all"))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--max_iterations", type=int, default=20)
    parser.add_argument("--time_limit_s", type=float, default=90.0)
    parser.add_argument("--instances_per_problem", type=int, default=12)
    parser.add_argument("--opt_param_sizes", default="1,2,3")
    parser.add_argument("--include_problem_types", default="")
    parser.add_argument("--exclude_problem_types", default="colorbar")
    parser.add_argument("--include_corca", action="store_true", default=False)
    parser.add_argument("--max_total_instances", type=int, default=0)
    parser.add_argument("--auto_seed", type=int, default=0)
    parser.add_argument("--algo_pool", default="all")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--merge_only", action="store_true", default=False)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    config_path = os.path.join(args.out_dir, "config_used.json")

    cfg = {
        "auto": True,
        "resume": True,
        "stable_instance_ids": True,
        "budget": {"max_iterations": int(args.max_iterations), "time_limit_s": float(args.time_limit_s)},
        "seeds": _parse_int_list(args.seeds),
        "candidates": _pick_candidate_pool(args.algo_pool),
        "auto_seed": int(args.auto_seed),
        "auto_instances_per_problem": int(args.instances_per_problem),
        "auto_opt_param_sizes": _parse_int_list(args.opt_param_sizes),
        "auto_targets": {
            "keff": 1.02,
            "pressuredrop_target": 600.0,
            "u_mean_target": 12.0,
        },
        "auto_include_problem_types": _parse_str_list(args.include_problem_types) or None,
        "auto_exclude_problem_types": _parse_str_list(args.exclude_problem_types),
        "auto_include_corca": bool(args.include_corca),
        "auto_max_total_instances": int(args.max_total_instances) if int(args.max_total_instances) > 0 else None,
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    t0 = time.time()
    jobs = max(1, int(args.jobs))
    shard_dirs = [os.path.join(args.out_dir, f"shard_{i:02d}") for i in range(jobs)]

    if not args.merge_only:
        if jobs == 1:
            cmd = [
                sys.executable,
                os.path.join(_repo_root(), "scripts", "generate_classification_data.py"),
                "--config",
                config_path,
                "--out_dir",
                args.out_dir,
            ]
            subprocess.run(cmd, check=True)
            shard_dirs = [args.out_dir]
        else:
            procs: List[Tuple[int, subprocess.Popen]] = []
            for i, shard_dir in enumerate(shard_dirs):
                os.makedirs(shard_dir, exist_ok=True)
                shard_cfg = dict(cfg)
                shard_cfg["resume"] = True
                shard_cfg["stable_instance_ids"] = True
                shard_cfg["shard_count"] = jobs
                shard_cfg["shard_index"] = i
                shard_cfg_path = os.path.join(shard_dir, "config_used.json")
                with open(shard_cfg_path, "w", encoding="utf-8") as f:
                    json.dump(shard_cfg, f, ensure_ascii=False, indent=2)
                cmd = [
                    sys.executable,
                    os.path.join(_repo_root(), "scripts", "generate_classification_data.py"),
                    "--config",
                    shard_cfg_path,
                    "--out_dir",
                    shard_dir,
                ]
                procs.append((i, subprocess.Popen(cmd)))

            failed = []
            for i, p in procs:
                rc = p.wait()
                if rc != 0:
                    failed.append((i, rc))
            if failed:
                raise RuntimeError(f"部分分片失败: {failed}")
    dt = time.time() - t0

    dataset_paths = [os.path.join(d, "dataset.jsonl") for d in shard_dirs]
    runs_paths = [os.path.join(d, "runs.jsonl") for d in shard_dirs]
    _merge_jsonl_dedup_by_instance_id(os.path.join(args.out_dir, "dataset.jsonl"), dataset_paths)
    _merge_runs_jsonl(os.path.join(args.out_dir, "runs.jsonl"), runs_paths)

    with open(os.path.join(args.out_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "seconds": dt,
                "config_path": config_path,
                "jobs": jobs,
                "shards": shard_dirs,
                "merge_only": bool(args.merge_only),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
