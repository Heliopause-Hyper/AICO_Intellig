from __future__ import annotations

from typing import Dict, List


def _case(label: str, base: Dict[str, float], rods: Dict[str, float] | None = None) -> Dict[str, object]:
    payload: Dict[str, object] = {"_case_label": label}
    payload.update(base)
    if rods:
        payload.update(rods)
    return payload


def state_family_candidates() -> List[Dict[str, object]]:
    thermal_regimes = [
        (
            "nominal",
            {"Pp:": 15.5, "Prk:": 50.0, "Tin:": 292.1, "bore_ppm": 1000.0},
        ),
        (
            "cool_low_boron",
            {"Pp:": 15.2, "Prk:": 48.0, "Tin:": 290.5, "bore_ppm": 850.0},
        ),
        (
            "warm_high_boron",
            {"Pp:": 15.8, "Prk:": 55.0, "Tin:": 294.0, "bore_ppm": 1150.0},
        ),
    ]

    rod_patterns = [
        ("baseline", {}),
        (
            "bank12_pair_mid",
            {
                "pos15: R, K 12 ": 180.0,
                "pos16: R, F 12 ": 180.0,
            },
        ),
        (
            "bank07_pair_mid",
            {
                "pos7: R, K 07 ": 185.0,
                "pos8: R, F 07 ": 185.0,
            },
        ),
        (
            "outer_quadrant_soft_pull",
            {
                "pos1: R, K 04 ": 195.0,
                "pos2: R, F 04 ": 195.0,
                "pos15: R, K 12 ": 195.0,
                "pos16: R, F 12 ": 195.0,
            },
        ),
        (
            "staggered_bank0712",
            {
                "pos7: R, K 07 ": 205.0,
                "pos8: R, F 07 ": 165.0,
                "pos15: R, K 12 ": 205.0,
                "pos16: R, F 12 ": 165.0,
            },
        ),
    ]

    selected_layout = [
        ("nominal", "baseline"),
        ("nominal", "bank12_pair_mid"),
        ("nominal", "bank07_pair_mid"),
        ("nominal", "outer_quadrant_soft_pull"),
        ("cool_low_boron", "baseline"),
        ("cool_low_boron", "bank12_pair_mid"),
        ("cool_low_boron", "staggered_bank0712"),
        ("warm_high_boron", "baseline"),
        ("warm_high_boron", "bank07_pair_mid"),
        ("warm_high_boron", "outer_quadrant_soft_pull"),
        ("warm_high_boron", "staggered_bank0712"),
    ]

    regime_map = {name: params for name, params in thermal_regimes}
    rod_map = {name: params for name, params in rod_patterns}

    cases: List[Dict[str, object]] = []
    for regime_name, rod_name in selected_layout:
        label = f"{regime_name}__{rod_name}"
        cases.append(_case(label, regime_map[regime_name], rod_map[rod_name]))
    return cases
