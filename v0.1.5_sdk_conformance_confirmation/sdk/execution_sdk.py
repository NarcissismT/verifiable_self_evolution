"""Frozen, target-neutral SDK exposed to v0.1.5 candidates.

The wrapper owns seed initialization and result validation.  Candidates only
need the public ``Budget`` and ``project_box`` helpers; the validation helpers
remain present in the card for compatibility but are intentionally not
required of student code.
"""

from __future__ import annotations

import math
import random
from typing import Iterable


SDK_VERSION = "v0.1.5-sdk-v1"


def initialize_seed(seed: int) -> None:
    """Compatibility helper; trusted wrapper calls the complete seed setup."""
    random.seed(int(seed))


def finite_vector(values: Iterable[float], dimension: int) -> list[float]:
    vector = [float(value) for value in values]
    if len(vector) != int(dimension) or not all(math.isfinite(value) for value in vector):
        raise ValueError("point must be a finite vector with the declared dimension")
    return vector


def project_box(values: Iterable[float], bounds: Iterable[Iterable[float]]) -> list[float]:
    vector = list(values)
    limits = [list(pair) for pair in bounds]
    if len(vector) != len(limits):
        raise ValueError("point and bounds dimensions differ")
    projected: list[float] = []
    for value, pair in zip(vector, limits):
        if len(pair) != 2:
            raise ValueError("each bound must contain lower and upper values")
        low, high = (float(pair[0]), float(pair[1]))
        if not math.isfinite(low) or not math.isfinite(high) or low > high:
            raise ValueError("bounds must be finite and ordered")
        projected.append(min(high, max(low, float(value))))
    return projected


class Budget:
    def __init__(self, limit: int):
        self.limit = int(limit)
        self.used = 0
        if self.limit < 0:
            raise ValueError("budget must be nonnegative")

    def consume(self, amount: int = 1) -> None:
        amount = int(amount)
        if amount < 0 or self.used + amount > self.limit:
            raise RuntimeError("oracle budget exceeded")
        self.used += amount

    @property
    def remaining(self) -> int:
        return self.limit - self.used
