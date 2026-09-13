import argparse
import glob
import json
import os
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Tuple


PDE_ELLIPTIC = "elliptic"
PDE_PARABOLIC = "parabolic"
PDE_HYPERBOLIC = "hyperbolic"
PDE_UNKNOWN = "unknown"


def _read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _infer_pde_type(problem_type: str, template_name: str) -> str:
    pt = str(problem_type or "").lower()
    tn = str(template_name or "").lower()
    s = f"{pt}::{tn}"
    if "advection" in s:
        return PDE_HYPERBOLIC
    if "blackscholes" in s or "heat_time" in s:
        return PDE_PARABOLIC

    if "sns" in s:
        return PDE_ELLIPTIC
    if "thmf" in s or "thermal_fins" in s or "heat_v" in s or "heat-cond" in s:
        return PDE_ELLIPTIC
    if "iaea" in s or "neutron_diffusion" in s:
        return PDE_ELLIPTIC

    return PDE_UNKNOWN


def _infer_problem_type_from_instance_id(instance_id: str) -> str:
    s = str(instance_id or "")
    if not s.startswith("inst_"):
        return ""
    parts = s.split("_")
    if len(parts) < 3:
        return ""
    return "_".join(parts[1:-1])


def _backfill_row(row: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    changed = False
    typed = row.get("typed_config_json")

    pt = str(row.get("problem_type") or "")
    tn = str(row.get("template_name") or "")
    if isinstance(typed, dict):
        if not pt:
            pt = str(typed.get("problem_type") or "")
        if not tn:
            tn = str(typed.get("template_name") or "")
    if not pt:
        pt = _infer_problem_type_from_instance_id(str(row.get("instance_id") or ""))

    inferred = _infer_pde_type(pt, tn)
    if inferred != PDE_UNKNOWN:
        if str(row.get("pde_type") or "").strip().lower() in {"", PDE_UNKNOWN}:
            row["pde_type"] = inferred
            changed = True
        if isinstance(typed, dict):
            if str(typed.get("pde_type") or "").strip().lower() in {"", PDE_UNKNOWN}:
                typed["pde_type"] = inferred
                changed = True
        elif typed is None:
            pass

    return row, changed


def _rewrite_jsonl_inplace(path: str, dry_run: bool) -> Dict[str, int]:
    total = 0
    changed = 0
    skipped = 0
    unknown = 0

    parent = os.path.dirname(os.path.abspath(path))
    tmp_fd = None
    tmp_path = None
    if not dry_run:
        tmp_fd, tmp_path = tempfile.mkstemp(prefix=".tmp_backfill_", suffix=".jsonl", dir=parent)
        tmp_f = os.fdopen(tmp_fd, "w", encoding="utf-8")
    else:
        tmp_f = None

    try:
        for row in _read_jsonl(path):
            total += 1
            typed = row.get("typed_config_json")

            pt = str(row.get("problem_type") or "")
            tn = str(row.get("template_name") or "")
            if isinstance(typed, dict):
                if not pt:
                    pt = str(typed.get("problem_type") or "")
                if not tn:
                    tn = str(typed.get("template_name") or "")
            if not pt:
                pt = _infer_problem_type_from_instance_id(str(row.get("instance_id") or ""))
            inferred = _infer_pde_type(pt, tn)
            if inferred == PDE_UNKNOWN:
                unknown += 1
            new_row, ch = _backfill_row(row)
            if ch:
                changed += 1
            if tmp_f is not None:
                tmp_f.write(json.dumps(new_row, ensure_ascii=False) + "\n")
        if tmp_f is not None:
            tmp_f.flush()
            os.fsync(tmp_f.fileno())
            tmp_f.close()
            tmp_f = None
        if not dry_run and tmp_path is not None:
            os.replace(tmp_path, path)
            tmp_path = None
    finally:
        if tmp_f is not None:
            try:
                tmp_f.close()
            except Exception:
                pass
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    return {"total": total, "changed": changed, "unknown_infer": unknown, "skipped": skipped}


def _collect_targets(root: str, include_patterns: List[str], exclude_patterns: List[str]) -> List[str]:
    root = os.path.abspath(root)
    out: List[str] = []
    for pat in include_patterns:
        out.extend(glob.glob(os.path.join(root, pat), recursive=True))
    out = [os.path.abspath(p) for p in out if os.path.isfile(p)]
    if exclude_patterns:
        ex = []
        for pat in exclude_patterns:
            ex.extend(glob.glob(os.path.join(root, pat), recursive=True))
        ex_set = {os.path.abspath(p) for p in ex}
        out = [p for p in out if p not in ex_set]
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/ycl/AICO-Intellig/results")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--only_results", action="store_true", default=True)
    ap.add_argument("--include", nargs="*", default=["**/problem_instance.jsonl", "**/dataset.jsonl"])
    ap.add_argument("--exclude", nargs="*", default=["**/algo_landscape_*/**", "**/algo_landscape_latest/**"])
    args = ap.parse_args()

    dry_run = not bool(args.apply)
    targets = _collect_targets(str(args.root), list(args.include), list(args.exclude))
    if not targets:
        print("no targets")
        return 0

    total_files = 0
    total_rows = 0
    total_changed = 0
    total_unknown = 0

    for fp in targets:
        stats = _rewrite_jsonl_inplace(fp, dry_run=dry_run)
        total_files += 1
        total_rows += int(stats["total"])
        total_changed += int(stats["changed"])
        total_unknown += int(stats["unknown_infer"])
        print(f"{fp} rows={stats['total']} changed={stats['changed']} unknown_infer={stats['unknown_infer']}")

    print(
        json.dumps(
            {
                "dry_run": bool(dry_run),
                "files": int(total_files),
                "rows": int(total_rows),
                "changed_rows": int(total_changed),
                "unknown_infer_rows": int(total_unknown),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
