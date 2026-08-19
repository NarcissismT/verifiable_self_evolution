#!/usr/bin/env python3
"""Target-neutral hidden-instance executor and evaluator for the real pilot.

The proposer sees the student API and public capsule, not this file or the
sealed instance stream before committing a proposal. Run inside Docker with
`--network none` and `VSE_NETWORK_POLICY=none`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
from types import ModuleType
from typing import Any, Callable

import numpy as np


SEEDS = (1031, 2063, 4099, 8191)
SUPPORTED_CASES = (
    "realpilot_flatness_penalty",
    "realpilot_nonconvex_simple",
    "realpilot_linear_coupling",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def finite_vector(value: Any, expected: int) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size != expected or not np.isfinite(vector).all():
        raise ValueError(f"expected {expected} finite coordinates")
    return vector


def make_problem(case_id: str, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    if case_id == "realpilot_flatness_penalty":
        return {
            "kind": "flat_lower_manifold_v1",
            "domain": {"x": [0.1, 2.0], "y": [-2.0, 2.0]},
            "a": float(rng.uniform(0.65, 1.35)),
            "upper_target": [float(rng.uniform(0.25, 1.65)), float(rng.uniform(-1.4, 1.4))],
        }
    if case_id == "realpilot_nonconvex_simple":
        return {
            "kind": "nonconvex_simple_v1",
            "domain": [[-2.0, 2.0], [-2.0, 2.0]],
            "a": rng.uniform(0.45, 1.55, size=2).tolist(),
            "upper_target": rng.uniform(-1.6, 1.6, size=2).tolist(),
            "coupling": float(rng.uniform(0.02, 0.08)),
        }
    if case_id == "realpilot_linear_coupling":
        matrix = rng.uniform(-0.35, 0.35, size=(2, 2))
        offset = rng.uniform(-0.25, 0.25, size=2)
        upper_target = rng.uniform(-1.2, 1.2, size=4)
        # Box constraints plus two genuine x/y coupled inequalities. The final
        # b values give every generated instance a strict feasible point.
        constraint = np.array([
            [1, 0, 0, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, -1, 0, 0],
            [0, 0, 1, 0], [0, 0, -1, 0], [0, 0, 0, 1], [0, 0, 0, -1],
            [0.45, -0.25, 0.55, 0.20], [-0.30, 0.50, 0.15, -0.55],
        ], dtype=np.float64)
        bound = np.array([1.6] * 8 + [1.4, 1.4], dtype=np.float64)
        return {
            "kind": "linear_coupling_v1",
            "domain": [[-1.6, 1.6]] * 4,
            "lower_matrix": matrix.tolist(),
            "lower_offset": offset.tolist(),
            "constraint_matrix": constraint.tolist(),
            "constraint_bound": bound.tolist(),
            "upper_target": upper_target.tolist(),
        }
    raise ValueError(f"unknown case: {case_id}")


def flat_metrics(problem: dict[str, Any], point: Any) -> dict[str, float]:
    vector = finite_vector(point, 2)
    x = float(np.clip(vector[0], 0.1, 2.0))
    y = float(np.clip(vector[1], -2.0, 2.0))
    a = float(problem["a"])
    target = np.asarray(problem["upper_target"], dtype=np.float64)
    lower = abs(y * y - a * x)
    upper = 0.5 * float(np.sum((np.array([x, y]) - target) ** 2))
    grid = np.linspace(0.1, 2.0, 4000)
    manifold = np.sqrt(a * grid)
    candidates = np.concatenate([
        0.5 * ((grid - target[0]) ** 2 + (manifold - target[1]) ** 2),
        0.5 * ((grid - target[0]) ** 2 + (-manifold - target[1]) ** 2),
    ])
    optimum = float(np.min(candidates))
    return {
        "lower_residual": lower,
        "upper_regret": max(0.0, upper - optimum),
        "primal_feasibility": 0.0,
    }


def simple_metrics(problem: dict[str, Any], point: Any) -> dict[str, float]:
    vector = finite_vector(point, 2)
    vector = np.clip(vector, -2.0, 2.0)
    a = np.asarray(problem["a"], dtype=np.float64)
    target = np.asarray(problem["upper_target"], dtype=np.float64)
    coupling = float(problem["coupling"])
    lower_gradient = vector * (vector * vector - a)
    lower = float(np.linalg.norm(lower_gradient, ord=2))
    upper = 0.5 * float(np.sum((vector - target) ** 2)) + coupling * math.sin(float(vector.sum()))
    roots = np.sqrt(a)
    candidates = []
    for first in (-1.0, 1.0):
        for second in (-1.0, 1.0):
            z = roots * np.array([first, second])
            candidates.append(0.5 * float(np.sum((z - target) ** 2)) + coupling * math.sin(float(z.sum())))
    return {
        "lower_residual": lower,
        "upper_regret": max(0.0, upper - min(candidates)),
        "primal_feasibility": 0.0,
    }


def linear_metrics(problem: dict[str, Any], point: Any) -> dict[str, float]:
    vector = finite_vector(point, 4)
    vector = np.clip(vector, -1.6, 1.6)
    x, y = vector[:2], vector[2:]
    matrix = np.asarray(problem["lower_matrix"], dtype=np.float64)
    offset = np.asarray(problem["lower_offset"], dtype=np.float64)
    constraint = np.asarray(problem["constraint_matrix"], dtype=np.float64)
    bound = np.asarray(problem["constraint_bound"], dtype=np.float64)
    target = np.asarray(problem["upper_target"], dtype=np.float64)
    lower = float(np.linalg.norm(y - (matrix @ x + offset), ord=2))
    feasibility = float(np.max(np.maximum(constraint @ vector - bound, 0.0)))
    upper = 0.5 * float(np.sum((vector - target) ** 2))
    grid = np.linspace(-1.6, 1.6, 61)
    optimum = float("inf")
    for x0 in grid:
        for x1 in grid:
            gx = np.array([x0, x1])
            gy = matrix @ gx + offset
            candidate = np.concatenate([gx, gy])
            if np.max(constraint @ candidate - bound) <= 1e-10:
                optimum = min(optimum, 0.5 * float(np.sum((candidate - target) ** 2)))
    if not math.isfinite(optimum):
        raise RuntimeError("generated linear instance has no feasible grid point")
    return {
        "lower_residual": lower,
        "upper_regret": max(0.0, upper - optimum),
        "primal_feasibility": feasibility,
    }


METRICS: dict[str, Callable[[dict[str, Any], Any], dict[str, float]]] = {
    "realpilot_flatness_penalty": flat_metrics,
    "realpilot_nonconvex_simple": simple_metrics,
    "realpilot_linear_coupling": linear_metrics,
}


def load_student(path: Path) -> ModuleType:
    resolved = path.resolve()
    spec = importlib.util.spec_from_file_location("vse_student_submission", resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load student implementation: {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "solve", None)):
        raise RuntimeError("student implementation must export callable solve")
    return module


class TimeoutError(RuntimeError):
    pass


def timeout_handler(_signum: int, _frame: Any) -> None:
    raise TimeoutError("student solve exceeded 30 seconds")


def execute(case_id: str, proposal_path: Path, output_path: Path) -> dict[str, Any]:
    if os.environ.get("VSE_NETWORK_POLICY") != "none":
        raise RuntimeError("trusted evaluator requires VSE_NETWORK_POLICY=none")
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "case_id", "model_digest", "algorithm_family",
        "hypothesis", "implementation", "implementation_sha256", "seeds",
        "oracle_budget",
    }
    allowed = required | {"hyperparameters"}
    if set(proposal) - allowed or required - set(proposal):
        raise ValueError("proposal schema field mismatch")
    if proposal.get("case_id") != case_id or case_id not in SUPPORTED_CASES:
        raise ValueError("proposal/case mismatch")
    model_digest = str(proposal.get("model_digest", ""))
    if not (len(model_digest) == 64 and all(char in "0123456789abcdef" for char in model_digest)):
        raise ValueError("proposal model_digest must be a lowercase SHA-256")
    implementation = (proposal_path.parent / str(proposal["implementation"])).resolve()
    implementation.relative_to(proposal_path.parent.resolve())
    implementation_digest = hashlib.sha256(implementation.read_bytes()).hexdigest()
    if proposal.get("implementation_sha256") != implementation_digest:
        raise ValueError("proposal implementation digest mismatch")
    student = load_student(implementation)
    seeds = [int(value) for value in proposal["seeds"]]
    if tuple(seeds) != SEEDS:
        raise ValueError(f"proposal seeds must be exactly {SEEDS}")
    budget = int(proposal["oracle_budget"])
    if budget != 4000:
        raise ValueError("real pilot oracle budget must be 4000")
    hyperparameters = dict(proposal.get("hyperparameters", {}))
    records: list[dict[str, Any]] = []
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    try:
        for seed in seeds:
            problem = make_problem(case_id, seed)
            signal.alarm(30)
            result = student.solve(case_id, problem, seed, budget, hyperparameters)
            signal.alarm(0)
            if not isinstance(result, dict):
                raise ValueError("solve must return a dictionary")
            oracle_calls = int(result["oracle_calls"])
            metrics = METRICS[case_id](problem, result["point"])
            records.append({
                "seed": seed,
                "point": [float(value) for value in finite_vector(result["point"], 4 if case_id == "realpilot_linear_coupling" else 2)],
                "oracle_calls": oracle_calls,
                "metrics": metrics,
                "within_budget": 0 <= oracle_calls <= budget,
            })
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    payload = {
        "schema_version": 1,
        "case_id": case_id,
        "proposal_digest": content_hash(proposal),
        "model_digest": model_digest,
        "records": records,
        "network_policy": "none",
        "execution_digest": "",
    }
    payload["execution_digest"] = content_hash(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def evaluate(execution_path: Path, output_path: Path) -> dict[str, Any]:
    if os.environ.get("VSE_NETWORK_POLICY") != "none":
        raise RuntimeError("trusted evaluator requires VSE_NETWORK_POLICY=none")
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    declared = str(execution.get("execution_digest", ""))
    unsealed = dict(execution)
    unsealed["execution_digest"] = ""
    if content_hash(unsealed) != declared:
        raise ValueError("execution output digest mismatch")
    case_id = str(execution["case_id"])
    thresholds = {
        "realpilot_flatness_penalty": {"lower_residual": 0.03, "upper_regret": 0.08, "primal_feasibility": 1e-12},
        "realpilot_nonconvex_simple": {"lower_residual": 0.08, "upper_regret": 0.08, "primal_feasibility": 1e-12},
        "realpilot_linear_coupling": {"lower_residual": 0.03, "upper_regret": 0.10, "primal_feasibility": 1e-8},
    }[case_id]
    failures: list[str] = []
    scores: list[float] = []
    records = execution.get("records", [])
    if len(records) != len(SEEDS):
        failures.append("missing_seed_records")
    for record in records:
        seed = record.get("seed", "unknown")
        if not record.get("within_budget"):
            failures.append(f"oracle_budget:{seed}")
        metrics = record.get("metrics", {})
        scaled = 0.0
        for metric, threshold in thresholds.items():
            value = float(metrics.get(metric, float("inf")))
            if not math.isfinite(value) or value > threshold:
                failures.append(f"{metric}:{seed}")
            scaled += value / max(threshold, 1e-12)
        scores.append(math.exp(-scaled / 3.0))
    hard_pass = not failures
    payload = {
        "schema_version": 1,
        "case_id": case_id,
        "execution_digest": declared,
        "hard_pass": hard_pass,
        "vds_score": float(np.mean(scores)) if hard_pass and scores else 0.0,
        "failures": sorted(set(failures)),
        "thresholds": thresholds,
        "evaluation_digest": "",
    }
    if not hard_pass:
        payload["vds_score"] = 0.0
    payload["evaluation_digest"] = content_hash(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def self_test(reference_module: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="vse-evaluator-test-") as temporary:
        root = Path(temporary)
        for case_id in SUPPORTED_CASES:
            case_dir = root / case_id
            case_dir.mkdir()
            implementation = case_dir / "solution.py"
            implementation.write_bytes(reference_module.read_bytes())
            proposal = {
                "schema_version": 1,
                "case_id": case_id,
                "model_digest": hashlib.sha256(reference_module.read_bytes()).hexdigest(),
                "algorithm_family": "pre-cutoff target-neutral engineering baseline",
                "hypothesis": "A deterministic grid or finite solution-set search should validate the execution and scoring chain.",
                "implementation": "solution.py",
                "implementation_sha256": hashlib.sha256(reference_module.read_bytes()).hexdigest(),
                "hyperparameters": {},
                "seeds": list(SEEDS),
                "oracle_budget": 4000,
            }
            proposal_path = case_dir / "proposal.json"
            proposal_path.write_text(json.dumps(proposal), encoding="utf-8")
            execution_path = case_dir / "execution.json"
            evaluation_path = case_dir / "evaluation.json"
            result = execute(case_id, proposal_path, execution_path)
            scored = evaluate(execution_path, evaluation_path)
            if not scored["hard_pass"]:
                raise RuntimeError(f"reference self-test failed for {case_id}: {scored['failures']}")
            print(json.dumps({"case_id": case_id, "vds_score": scored["vds_score"], "records": len(result["records"])}))


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--case-id", choices=SUPPORTED_CASES, required=True)
    execute_parser.add_argument("--proposal", type=Path, required=True)
    execute_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--execution", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    test_parser = subparsers.add_parser("self-test")
    test_parser.add_argument("--reference-module", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "execute":
        print(json.dumps(execute(args.case_id, args.proposal, args.output), sort_keys=True))
    elif args.command == "evaluate":
        print(json.dumps(evaluate(args.execution, args.output), sort_keys=True))
    else:
        self_test(args.reference_module)
    return 0


if __name__ == "__main__":
    sys.exit(main())
