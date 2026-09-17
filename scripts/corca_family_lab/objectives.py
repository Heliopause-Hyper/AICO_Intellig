from __future__ import annotations

from typing import Dict, Tuple


def _safe_float(metrics: Dict[str, object], key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _positive_part(value: float) -> float:
    return float(value) if value > 0 else 0.0


def _missing_keys(metrics: Dict[str, object], names: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple(name for name in names if metrics.get(name) is None)


def score_state_metrics(
    metrics: Dict[str, object],
    target_keff: float = 1.0,
    fq_cap: float = 3.0,
    fdh_cap: float = 1.5,
    ao_target: float = 0.0,
    bore_center: float = 1000.0,
) -> Tuple[float, Dict[str, float]]:
    missing = _missing_keys(metrics, ("keff", "FQ", "FDH", "AO"))
    if missing:
        return 1e9, {"missing_required_metrics": float(len(missing))}
    pieces = {
        "keff_error": 100.0 * abs(_safe_float(metrics, "keff", target_keff) - target_keff),
        "fq_over": 5.0 * _positive_part(_safe_float(metrics, "FQ", 0.0) - fq_cap),
        "fdh_over": 5.0 * _positive_part(_safe_float(metrics, "FDH", 0.0) - fdh_cap),
        "ao_abs": abs(_safe_float(metrics, "AO", ao_target) - ao_target) / 20.0,
        "bore_shift": abs(_safe_float(metrics, "bore", bore_center) - bore_center) / 1000.0,
    }
    return float(sum(pieces.values())), pieces


def score_burnup_metrics(
    metrics: Dict[str, object],
    target_keff: float = 1.0,
    fq_cap: float = 3.0,
    burnup_target: float = 150.0,
    burnup_peak_cap: float = 320.0,
) -> Tuple[float, Dict[str, float]]:
    missing = _missing_keys(metrics, ("keff", "FQ", "burnup_deep", "burnup_finished"))
    if missing:
        return 1e9, {"missing_required_metrics": float(len(missing))}
    grid = metrics.get("burnup_2d_stats") or {}
    pieces = {
        "keff_error": 100.0 * abs(_safe_float(metrics, "keff", target_keff) - target_keff),
        "fq_over": 5.0 * _positive_part(_safe_float(metrics, "FQ", 0.0) - fq_cap),
        "burnup_terminal": abs(_safe_float(metrics, "burnup_finished", burnup_target) - burnup_target) / 100.0,
        "burnup_deep": abs(_safe_float(metrics, "burnup_deep", burnup_target) - burnup_target) / 100.0,
        "burnup_peak": _positive_part(_safe_float(grid, "max", 0.0) - burnup_peak_cap) / 50.0,
    }
    return float(sum(pieces.values())), pieces


def score_xenon_metrics(
    metrics: Dict[str, object],
    target_keff: float = 1.0,
    ao_target: float = 0.0,
    di_target: float = 0.0,
    fq_cap: float = 3.0,
    fdh_cap: float = 1.5,
    boron_center: float = 1000.0,
) -> Tuple[float, Dict[str, float]]:
    missing = _missing_keys(metrics, ("keff", "AO", "DI", "boron_concentration"))
    if missing:
        return 1e9, {"missing_required_metrics": float(len(missing))}
    pieces = {
        "keff_error": 100.0 * abs(_safe_float(metrics, "keff", target_keff) - target_keff),
        "ao_abs": abs(_safe_float(metrics, "AO", ao_target) - ao_target) / 25.0,
        "di_abs": abs(_safe_float(metrics, "DI", di_target) - di_target) / 25.0,
        "fq_over": 5.0 * _positive_part(_safe_float(metrics, "FQ", 0.0) - fq_cap),
        "fdh_over": 5.0 * _positive_part(_safe_float(metrics, "FDH", 0.0) - fdh_cap),
        "boron_shift": abs(_safe_float(metrics, "boron_concentration", boron_center) - boron_center) / 1000.0,
    }
    return float(sum(pieces.values())), pieces


def score_family(problem_type: str, metrics: Dict[str, object]) -> Tuple[float, Dict[str, float]]:
    if problem_type == "corca_state":
        return score_state_metrics(metrics)
    if problem_type == "corca_evol":
        return score_burnup_metrics(metrics)
    if problem_type == "corca_xenon":
        return score_xenon_metrics(metrics)
    raise ValueError(f"Unsupported problem_type: {problem_type}")
