"""Frozen target-neutral execution SDK for v0.1.4 bridge candidates."""

from __future__ import annotations

import math
import random
from typing import Iterable


SDK_VERSION = "v0.1.4-sdk-v1"


def initialize_seed(seed: int) -> None:
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
    return [min(float(high), max(float(low), float(value))) for value, (low, high) in zip(vector, limits)]


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
