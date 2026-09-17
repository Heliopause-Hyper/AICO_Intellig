from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List


REPO_ROOT = "/home/ycl/AICO-Intellig"
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "results", "corca_family_lab")
DEFAULT_BURNUP = os.path.join(REPO_ROOT, "folderA", "preciseFZ", "result", "SimuOutBurnup.out")
DEFAULT_XENON = os.path.join(REPO_ROOT, "folderA", "preciseFZ", "result", "SimuOutXenon.out")


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        return handle.read()


def _extract_scalar_block(text: str, name: str) -> float | None:
    pattern = re.compile(
        rf"@{re.escape(name)}_BEG\s*([^\n\r@]+?)\s*@{re.escape(name)}_END",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return None
    raw = match.group(1).strip().splitlines()[0].strip()
    try:
        return float(raw)
    except Exception:
        return None


def _extract_grid_block(text: str, name: str) -> List[List[float]]:
    pattern = re.compile(
        rf"@{re.escape(name)}_BEG\s*(.*?)\s*@{re.escape(name)}_END",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return []
    rows: List[List[float]] = []
    for line in match.group(1).strip().splitlines():
        vals = []
        for token in line.split():
            try:
                vals.append(float(token))
            except Exception:
                vals = []
                break
        if vals:
            rows.append(vals)
    return rows


def _grid_stats(grid: List[List[float]]) -> Dict[str, float | int]:
    flat = [value for row in grid for value in row]
    if not flat:
        return {"rows": 0, "cols": 0, "min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "rows": len(grid),
        "cols": len(grid[0]) if grid else 0,
        "min": float(min(flat)),
        "max": float(max(flat)),
        "mean": float(sum(flat) / len(flat)),
    }


def parse_burnup_output(path: str) -> Dict[str, object]:
    text = _read_text(path)
    burnup_2d = _extract_grid_block(text, "Burnup_Distribution_2D")
    burnup_deep = _extract_scalar_block(text, "Burnup_Deep")
    keff = _extract_scalar_block(text, "Effective_Increment_Factor")
    fq = _extract_scalar_block(text, "FQ")
    return {
        "source_path": os.path.abspath(path),
        "kind": "burnup",
        "burnup_deep": burnup_deep,
        "keff": keff,
        "FQ": fq,
        "burnup_2d_stats": _grid_stats(burnup_2d),
    }


def parse_xenon_output(path: str) -> Dict[str, object]:
    text = _read_text(path)
    return {
        "source_path": os.path.abspath(path),
        "kind": "xenon",
        "AO": _extract_scalar_block(text, "AO"),
        "DI": _extract_scalar_block(text, "DI"),
        "boron_concentration": _extract_scalar_block(text, "Boron_Concentration"),
        "keff": _extract_scalar_block(text, "Effective_Increment_Factor"),
        "rod_position": _extract_scalar_block(text, "Rod_Position"),
        "FDH": _extract_scalar_block(text, "FDH"),
        "FQ": _extract_scalar_block(text, "FQ"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Parse CORCA lab outputs into lightweight JSON summaries.")
    ap.add_argument("--burnup_out", default=DEFAULT_BURNUP)
    ap.add_argument("--xenon_out", default=DEFAULT_XENON)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    summaries: Dict[str, object] = {}
    if os.path.exists(args.burnup_out):
        summaries["burnup"] = parse_burnup_output(args.burnup_out)
    if os.path.exists(args.xenon_out):
        summaries["xenon"] = parse_xenon_output(args.xenon_out)

    out_path = os.path.join(out_dir, "corca_output_summary.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(summaries, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Wrote: {out_path}")
    for name, summary in summaries.items():
        print(f"[{name}] {json.dumps(summary, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
