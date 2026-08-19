"""Target-neutral engineering baseline used only for evaluator self-tests.

This is not a model-generated proposal and must not be reported as a real
paper rediscovery result.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def solve(
    case_id: str,
    problem: dict[str, Any],
    seed: int,
    oracle_budget: int,
    hyperparameters: dict[str, Any],
) -> dict[str, Any]:
    del seed, hyperparameters
    if case_id == "realpilot_flatness_penalty":
        a = float(problem["a"])
        target = np.asarray(problem["upper_target"], dtype=np.float64)
        grid = np.linspace(0.1, 2.0, 2000)
        manifold = np.sqrt(a * grid)
        best_point = None
        best_value = float("inf")
        for sign in (-1.0, 1.0):
            y = sign * manifold
            values = 0.5 * ((grid - target[0]) ** 2 + (y - target[1]) ** 2)
            index = int(np.argmin(values))
            if float(values[index]) < best_value:
                best_value = float(values[index])
                best_point = [float(grid[index]), float(y[index])]
        return {"point": best_point, "oracle_calls": min(4000, oracle_budget)}

    if case_id == "realpilot_nonconvex_simple":
        a = np.asarray(problem["a"], dtype=np.float64)
        target = np.asarray(problem["upper_target"], dtype=np.float64)
        coupling = float(problem["coupling"])
        roots = np.sqrt(a)
        best_point = None
        best_value = float("inf")
        for first in (-1.0, 1.0):
            for second in (-1.0, 1.0):
                point = roots * np.array([first, second])
                value = 0.5 * float(np.sum((point - target) ** 2)) + coupling * math.sin(float(point.sum()))
                if value < best_value:
                    best_value, best_point = value, point.tolist()
        return {"point": best_point, "oracle_calls": 4}

    if case_id == "realpilot_linear_coupling":
        matrix = np.asarray(problem["lower_matrix"], dtype=np.float64)
        offset = np.asarray(problem["lower_offset"], dtype=np.float64)
        constraint = np.asarray(problem["constraint_matrix"], dtype=np.float64)
        bound = np.asarray(problem["constraint_bound"], dtype=np.float64)
        target = np.asarray(problem["upper_target"], dtype=np.float64)
        grid = np.linspace(-1.6, 1.6, 61)
        best_point = None
        best_value = float("inf")
        calls = 0
        for first in grid:
            for second in grid:
                x = np.array([first, second])
                y = matrix @ x + offset
                point = np.concatenate([x, y])
                calls += 1
                if np.max(constraint @ point - bound) <= 1e-10:
                    value = 0.5 * float(np.sum((point - target) ** 2))
                    if value < best_value:
                        best_value, best_point = value, point.tolist()
        if best_point is None:
            raise RuntimeError("no feasible baseline point")
        return {"point": best_point, "oracle_calls": min(calls, oracle_budget)}

    raise ValueError(f"unsupported case: {case_id}")
