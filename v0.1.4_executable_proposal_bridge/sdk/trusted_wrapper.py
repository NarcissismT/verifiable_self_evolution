#!/usr/bin/env python3
"""Trusted JSON and error boundary around an untrusted bridge candidate."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import random
import sys
from typing import Any


def load_candidate(path: Path, sdk_root: Path) -> Any:
    sys.path.insert(0, str(sdk_root.resolve()))
    spec = importlib.util.spec_from_file_location("vse_bridge_candidate", path.resolve())
    if spec is None or spec.loader is None:
        raise RuntimeError("candidate module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "solve", None)):
        raise ValueError("candidate must define callable solve(problem, seed, budget)")
    return module


def validate_result(result: Any, dimension: int, budget: int) -> dict[str, Any]:
    if not isinstance(result, dict) or set(result) != {"point", "oracle_calls"}:
        raise ValueError("solve result keys must be exactly point and oracle_calls")
    point = result["point"]
    calls = result["oracle_calls"]
    if not isinstance(point, list) or len(point) != dimension:
        raise ValueError("point has the wrong dimension")
    vector = [float(value) for value in point]
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("point contains a nonfinite value")
    if not isinstance(calls, int) or isinstance(calls, bool) or not 0 <= calls <= budget:
        raise ValueError("oracle_calls is outside the execution budget")
    return {"point": vector, "oracle_calls": calls}


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: trusted_wrapper.py CANDIDATE SDK_ROOT")
    payload = json.load(sys.stdin)
    problem = dict(payload["problem"])
    seed = int(payload["seed"])
    budget = int(payload["budget"])
    random.seed(seed)
    try:
        candidate = load_candidate(Path(sys.argv[1]), Path(sys.argv[2]))
        result = candidate.solve(problem, seed, budget)
        sealed = {
            "ok": True,
            "result": validate_result(result, int(problem["dimension"]), budget),
            "error_category": "",
            "error_message": "",
        }
    except Exception as error:
        sealed = {
            "ok": False,
            "result": {},
            "error_category": type(error).__name__,
            "error_message": str(error)[:2000],
        }
    print(json.dumps(sealed, allow_nan=False, sort_keys=True))
    return 0 if sealed["ok"] else 65


if __name__ == "__main__":
    raise SystemExit(main())
