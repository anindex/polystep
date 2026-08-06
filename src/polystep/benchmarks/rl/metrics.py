"""Metric helpers for RL experiment results."""

from __future__ import annotations

import math
from typing import Any, Dict


def normalize_score(value: float, random_return: float, reference_return: float) -> float:
    """Normalize a return so random is 0 and a reference policy is 1."""

    denom = reference_return - random_return
    if abs(denom) < 1e-12:
        return 0.0
    return float((value - random_return) / denom)


def build_rl_metrics(
    *,
    final_return: float,
    best_return: float,
    normalized_score: float,
    wall_time_seconds: float,
    peak_gpu_memory_mb: float,
    function_evals: int,
    total_steps: int,
    rl_env_steps: int,
    best_normalized_score: float | None = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Build metrics compatible with ``experiments.runners.common.save_result``; returns return fields plus accuracy aliases."""

    best_norm = float(best_normalized_score) if best_normalized_score is not None else float(normalized_score)
    metrics: Dict[str, Any] = {
        "final_accuracy": float(normalized_score),
        "best_accuracy": best_norm,
        "wall_time_seconds": float(wall_time_seconds),
        "peak_gpu_memory_mb": float(peak_gpu_memory_mb),
        # None, not int(): gradient baselines evaluate no candidates, and JSON has no NaN.
        "function_evals": (
            None
            if function_evals is None or (isinstance(function_evals, float) and math.isnan(function_evals))
            else int(function_evals)
        ),
        "total_steps": int(total_steps),
        "final_return": float(final_return),
        "best_return": float(best_return),
        "normalized_score": float(normalized_score),
        "rl_env_steps": int(rl_env_steps),
    }
    metrics.update(extra)
    return metrics
