from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List


REPO_ROOT = "/home/ycl/AICO-Intellig"
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "results", "corca_family_lab")


def _state_groups() -> List[Dict[str, object]]:
    return [
        {
            "name": "thermal_triplet",
            "variables": ["Pp:", "Prk:", "Tin:"],
            "note": "Current stable low-risk subset already used by existing CORCA smoke runs.",
        },
        {
            "name": "rod_bank_quadrants",
            "variables": [
                "pos1: R, K 04 ",
                "pos2: R, F 04 ",
                "pos7: R, K 07 ",
                "pos8: R, F 07 ",
                "pos15: R, K 12 ",
                "pos16: R, F 12 ",
            ],
            "note": "First expansion target for grouped rod control without opening the full 16-rod space.",
        },
        {
            "name": "boron_plus_banks",
            "variables": ["bore_ppm", "Pp:", "Prk:", "Tin:"],
            "note": "Couples chemistry and thermal controls for shared-subspace stress tests.",
        },
    ]


def _transient_groups() -> List[Dict[str, object]]:
    return [
        {
            "problem_type": "corca_evol",
            "template_name": "burnup",
            "groups": [
                ["sequence_count", "burnup_steps_1", "rod_1"],
                ["burnup_steps_2", "rod_2"],
                ["burnup_steps_3", "rod_3"],
            ],
            "targets": ["keff", "FQ", "burnup_finished", "burnup_deep"],
            "regimes": [
                "short_100_step_baseline",
                "balanced_150",
                "front_loaded_150",
                "anchor_150",
            ],
            "notes": [
                "Anchor-style 100+50 schedules reached the current best burnup score under the lab objective.",
                "Terminal burnup and deep burnup should both be tracked to avoid favoring truncated schedules.",
            ],
        },
        {
            "problem_type": "corca_xenon",
            "template_name": "xenon",
            "groups": [
                ["xenon_steps", "power_1", "rod_1"],
                ["step_2", "power_2", "rod_2", "change_state_2"],
                ["step_3", "power_3", "rod_3", "change_state_3"],
                ["step_4", "power_4", "rod_4", "change_state_4"],
            ],
            "targets": ["keff", "AO", "DI", "FDH", "FQ", "boron_concentration"],
            "regimes": [
                "low_power_hold",
                "drop_schedule",
                "soft_plateau",
                "recovery",
            ],
            "notes": [
                "Current xenon best cases stay in the low-power high-rod corner.",
                "Boron concentration remains the dominant penalty term and should stay explicit in the protocol.",
            ],
        },
    ]


def build_protocol() -> Dict[str, object]:
    return {
        "lab_name": "corca_family_lab",
        "design_goal": "Expand CORCA tasks through isolated lab protocols without touching the main generation flow.",
        "guardrails": [
            "No edits to scripts/generate_classification_data.py at lab bootstrap stage.",
            "No edits to src/optimizer.py at lab bootstrap stage.",
            "New templates stay under templates/corcasim_lab/.",
            "All new artifacts stay under results/corca_family_lab/.",
        ],
        "families": [
            {
                "family_id": "corca_state_family_v1",
                "problem_type": "corca_state",
                "template_name": "default",
                "objective_pool": ["keff", "FQ", "FDH", "AO", "deltai", "bore"],
                "parameter_groups": _state_groups(),
                "regime_axes": ["power", "pressure", "temperature", "boron", "rod_bank_group"],
            },
            {
                "family_id": "corca_transient_family_v1",
                "problem_type": ["corca_evol", "corca_xenon"],
                "template_name": ["burnup", "xenon"],
                "objective_pool": ["keff", "FQ", "burnup_finished", "burnup_deep", "AO", "DI", "FDH", "boron_concentration"],
                "parameter_groups": _transient_groups(),
                "regime_axes": ["horizon", "rod_schedule", "power_schedule", "terminal_target"],
            },
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build isolated CORCA family lab protocol manifest.")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    payload = build_protocol()
    out_path = os.path.join(out_dir, "corca_family_protocol_v1.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Wrote: {out_path}")
    print(json.dumps({"families": [row["family_id"] for row in payload["families"]]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
