"""Independent metrics for the five public quadratic confirmation tasks."""

from __future__ import annotations

import math
from typing import Any


def optimum(problem: dict[str, Any]) -> list[float]:
    target = [float(value) + float(offset) for value, offset in zip(problem["target"], problem.get("offset", []))]
    return [min(float(high), max(float(low), value)) for value, (low, high) in zip(target, problem["bounds"])]


def metrics(problem: dict[str, Any], point: Any) -> dict[str, float]:
    expected = optimum(problem)
    values = [float(value) for value in point]
    if len(values) != len(expected) or not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite or wrong-dimensional point")
    residual = math.sqrt(sum((value - target) ** 2 for value, target in zip(values, expected)))
    regret = 0.5 * residual * residual
    feasibility = max(max(float(low) - value, 0.0, value - float(high))
                      for value, (low, high) in zip(values, problem["bounds"]))
    return {"lower_residual": residual, "kkt_ni_gap": residual, "regret": regret,
            "primal_feasibility": feasibility}
