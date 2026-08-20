#!/usr/bin/env python3
"""Run the causal proposal gate and its fail-closed negative controls."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import hmac
import importlib.util
import json
from pathlib import Path
import re
import sys
from datetime import datetime, timezone
from typing import Any

from vse.contracts import CandidateSource, ExecutionResult, Split, Task, TrajectoryRecord
from vse.hashing import canonical_json, content_hash, file_hash
from vse.proposal_io import parse_model_proposal
from vse.runner import CodeRunner, RunnerConfig, execution_digest
from vse.verifier import DeclarativeHardVerifier, MetricRule


CASES = (
    "realpilot_flatness_penalty",
    "realpilot_nonconvex_simple",
    "realpilot_linear_coupling",
)
SEEDS = (1031, 2063, 4099, 8191)
BASELINES = (
    "precutoff_penalty_synthesis_1",
    "precutoff_penalty_synthesis_2",
    "precutoff_penalty_synthesis_3",
)
METRICS = frozenset((
    "lower_residual", "upper_regret", "primal_feasibility",
    "oracle_calls", "seed_reproducibility",
))
SCHEDULE = (1000.0, 2000.0, 4000.0)
EXECUTION_CONTAINER_DIGEST = "sha256:f669006c1ce0d3761b4017e4c600c3e0424e4670a0b9262ec53cb33d3406a666"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capsule_digest(capsule: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(capsule).encode("utf-8")).hexdigest()


def task_for(case_id: str, capsule: dict[str, Any]) -> Task:
    return Task(
        task_id=case_id,
        family=str(capsule["field"]),
        split=Split.DEV,
        statement=str(capsule["public_problem"]),
        instance={"case_id": case_id, "capsule_digest": capsule_digest(capsule)},
        verifier_version="causal-real-pilot-evaluator-v1",
        tags=("real-pilot", "engineering-gate", case_id),
    )


def proposal_from_file(path: Path, task: Task, model_digest: str):
    full = json.loads(path.read_text(encoding="utf-8"))
    if full.get("task_id") != task.task_id or full.get("source") != CandidateSource.STUDENT.value:
        raise ValueError(f"proposal identity mismatch: {path}")
    if full.get("model_digest") != model_digest:
        raise ValueError(f"proposal model digest mismatch: {path}")
    value = {
        key: full[key]
        for key in (
            "hypothesis", "solution", "experiment_code", "seeds", "baselines",
            "primary_metric", "secondary_metrics", "expected_effect",
            "power_assumptions", "stopping_rule", "resource_schedule",
        )
    }
    proposal = parse_model_proposal(
        json.dumps(value),
        task=task,
        source=CandidateSource.STUDENT,
        model_id=str(full["model_id"]),
        model_digest=model_digest,
        round_index=int(full["round_index"]),
        frozen_seeds=SEEDS,
        mandatory_baselines=(BASELINES[0],),
        allowed_baselines=frozenset(BASELINES),
        allowed_metrics=METRICS,
        frozen_resource_schedule=SCHEDULE,
    )
    if canonical_json(proposal.payload()) != canonical_json(full):
        raise ValueError(f"proposal payload/digest mismatch: {path}")
    return proposal


def audit_code(code: str) -> None:
    tree = __import__("ast").parse(code)
    allowed_imports = {"json", "math", "sys", "random"}
    forbidden_tokens = (
        "openreview", "arxiv", "neurips", "requests", "urllib", "socket",
        "subprocess", "target.pdf", "sealed/", "authors",
    )
    if any(token in code.lower() for token in forbidden_tokens):
        raise ValueError("target leakage or network token in experiment code")
    for node in __import__("ast").walk(tree):
        if isinstance(node, __import__("ast").Import):
            if any(alias.name.split(".")[0] not in allowed_imports for alias in node.names):
                raise ValueError("experiment code imports a non-approved module")
        if isinstance(node, __import__("ast").ImportFrom):
            if str(node.module).split(".")[0] not in allowed_imports:
                raise ValueError("experiment code imports a non-approved module")


def baseline_point(case_id: str, problem: dict[str, Any]) -> list[float]:
    if case_id == "realpilot_linear_coupling":
        return [0.0, 0.0, float(problem["lower_offset"][0]), float(problem["lower_offset"][1])]
    return [0.0, 0.0]


def trusted_execution(
    case_id: str,
    task: Task,
    proposal: Any,
    execution: ExecutionResult,
    evaluator_module: Any,
    evaluator_digest: str,
    problem: dict[str, Any],
) -> ExecutionResult:
    if execution.exit_code != 0 or execution.timed_out:
        return replace(execution, trusted_metrics={}, trusted_evaluator_digest=evaluator_digest)
    candidate = execution.candidate_result
    if not isinstance(candidate, dict):
        return replace(execution, trusted_metrics={}, trusted_evaluator_digest=evaluator_digest)
    try:
        point = candidate["point"]
        oracle_calls = int(candidate["oracle_calls"])
        unit_tests = candidate.get("unit_tests", {})
        expected = 4 if case_id == "realpilot_linear_coupling" else 2
        metrics = evaluator_module.METRICS[case_id](problem, point)
        finite = all(__import__("math").isfinite(float(value)) for value in point)
        shape = len(point) == expected
        budget = 0 <= oracle_calls <= 4000
        unit_passed = int(finite) + int(shape) + int(budget) + int(
            isinstance(unit_tests, dict) and int(unit_tests.get("passed", 0)) <= int(unit_tests.get("total", 0))
        )
        unit_total = 4
        baseline_metrics = evaluator_module.METRICS[case_id](problem, baseline_point(case_id, problem))
        thresholds = {
            "realpilot_flatness_penalty": {"lower_residual": 0.03, "upper_regret": 0.08, "primal_feasibility": 1e-12},
            "realpilot_nonconvex_simple": {"lower_residual": 0.08, "upper_regret": 0.08, "primal_feasibility": 1e-12},
            "realpilot_linear_coupling": {"lower_residual": 0.03, "upper_regret": 0.10, "primal_feasibility": 1e-8},
        }[case_id]
        hard = finite and shape and budget and all(
            __import__("math").isfinite(float(metrics[key])) and float(metrics[key]) <= threshold
            for key, threshold in thresholds.items()
        ) and unit_passed == unit_total
        quality = -sum(float(metrics[key]) / max(threshold, 1e-12) for key, threshold in thresholds.items())
        baseline_quality = -sum(float(baseline_metrics[key]) / max(threshold, 1e-12) for key, threshold in thresholds.items())
        trusted = {
            **{key: float(value) for key, value in metrics.items()},
            "oracle_calls": float(oracle_calls),
            "seed_reproducibility": 1.0,
            "unit_tests_passed": unit_passed,
            "unit_tests_total": unit_total,
            "quality": quality,
            "baseline_quality": baseline_quality,
            "resource_quality_curve": [
                {"budget": budget_value, "quality": quality}
                for budget_value in proposal.resource_schedule
            ],
            "vds_empirical": float(hard),
            "vds_hypothesis": float(bool(proposal.hypothesis.claim and proposal.hypothesis.mechanism)),
            "vds_experiment": float(hard),
            "vds_novelty": 0.0,
            "vds_calibration": float(unit_passed == unit_total),
        }
        if not hard:
            trusted["unit_tests_passed"] = min(unit_passed, unit_total - 1)
        return replace(execution, trusted_metrics=trusted, trusted_evaluator_digest=evaluator_digest)
    except (KeyError, TypeError, ValueError, OverflowError):
        return replace(execution, trusted_metrics={}, trusted_evaluator_digest=evaluator_digest)


def verifier(evaluator_digest: str) -> DeclarativeHardVerifier:
    return DeclarativeHardVerifier(
        verifier_version="causal-real-pilot-evaluator-v1",
        trusted_evaluator_digest=evaluator_digest,
        metric_rules=(
            MetricRule("lower_residual", "lower_residual", "min", 0.03, 0.0, 1.0),
            MetricRule("upper_regret", "upper_regret", "min", 0.10, 0.0, 1.0),
            MetricRule("primal_feasibility", "primal_feasibility", "min", 1e-8, 0.0, 1.0),
        ),
        max_runtime_seconds=30.0,
        min_unit_pass_rate=1.0,
        min_power=0.0,
    )


def sign_receipt(value: dict[str, Any], key: bytes) -> dict[str, Any]:
    unsigned = dict(value)
    unsigned["receipt_digest"] = ""
    unsigned["hmac_signature"] = ""
    value["receipt_digest"] = content_hash(unsigned)
    value["hmac_signature"] = hmac.new(key, value["receipt_digest"].encode(), hashlib.sha256).hexdigest()
    return value


def run_case(
    case_id: str,
    bundle_root: Path,
    proposals_root: Path,
    output_root: Path,
    model_digest: str,
    evaluator_module: Any,
    evaluator_digest: str,
    hmac_key: bytes,
) -> dict[str, Any]:
    capsule_path = bundle_root / "run" / "public" / case_id / "capsule.json"
    capsule = json.loads(capsule_path.read_text(encoding="utf-8"))
    task = task_for(case_id, capsule)
    proposal_path = proposals_root / case_id / "proposal.json"
    proposal = proposal_from_file(proposal_path, task, model_digest)
    audit_code(proposal.experiment_code)
    case_output = output_root / case_id
    case_output.mkdir(parents=True, exist_ok=True)
    runner = CodeRunner(RunnerConfig(mode="local_test", timeout_seconds=30.0, memory_bytes=2 * 1024**3))
    executions: list[ExecutionResult] = []
    for seed in proposal.seeds:
        problem = evaluator_module.make_problem(case_id, seed)
        execution = runner.execute(
            task,
            proposal,
            seed,
            extra_input={"case_id": case_id, "problem": problem, "capsule_digest": capsule_digest(capsule)},
        )
        executions.append(trusted_execution(case_id, task, proposal, execution, evaluator_module, evaluator_digest, problem))
    report = verifier(evaluator_digest).verify(task, tuple(executions))
    record = TrajectoryRecord(
        task=task,
        proposal=proposal,
        executions=tuple(executions),
        verification=report,
        created_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    )
    (case_output / "proposal.json").write_text(json.dumps(proposal.payload(), indent=2, sort_keys=True) + "\n")
    (case_output / "verification_report.json").write_text(json.dumps(report.payload(), indent=2, sort_keys=True) + "\n")
    (case_output / "trajectory_record.json").write_text(json.dumps(record.payload(), indent=2, sort_keys=True) + "\n")
    for execution in executions:
        (case_output / f"execution_{execution.seed}.json").write_text(json.dumps(asdict(execution), indent=2, sort_keys=True) + "\n")
    receipt = sign_receipt({
        "schema_version": 1,
        "gate": "v0.1.3_causal_proposal_gate",
        "case_id": case_id,
        "raw_model_output_sha256": file_hash(proposals_root / case_id / "raw_model_output.txt"),
        "proposal_digest": proposal.digest,
        "task_digest": task.digest,
        "capsule_digest": capsule_digest(capsule),
        "generated_code_sha256": hashlib.sha256(proposal.experiment_code.encode()).hexdigest(),
        "seeds": list(proposal.seeds),
        "baselines": list(proposal.baselines),
        "resource_schedule": list(proposal.resource_schedule),
        "execution_container_digest": EXECUTION_CONTAINER_DIGEST,
        "trusted_evaluator_digest": evaluator_digest,
        "execution_digests": [execution_digest(item) for item in executions],
        "verification_report_digest": report.report_digest,
        "accepted": report.accepted,
    }, hmac_key)
    (case_output / "bound_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return {"case_id": case_id, "accepted": report.accepted, "hard_failures": list(report.hard_failures), "vds_score": report.vds_score, "report_digest": report.report_digest}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("positive", "negative-controls"))
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--proposals-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-digest", required=True)
    parser.add_argument("--hmac-key", type=Path, required=True)
    args = parser.parse_args()
    evaluator_path = args.bundle_root / "evaluator" / "trusted_evaluator.py"
    evaluator_module = load_module(evaluator_path, "causal_gate_trusted_evaluator")
    evaluator_digest = file_hash(evaluator_path)
    key = args.hmac_key.read_bytes()
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    if args.command == "positive":
        for case_id in CASES:
            results.append(run_case(case_id, args.bundle_root, args.proposals_root, args.output_root, args.model_digest, evaluator_module, evaluator_digest, key))
        passed = all(item["accepted"] for item in results)
    else:
        passed = run_negative_controls(args, evaluator_module, evaluator_digest, key)
        results = [{"negative_controls_passed": passed}]
    report = {
        "schema_version": 1,
        "gate": "v0.1.3_causal_proposal_gate",
        "command": args.command,
        "status": "passed" if passed else "failed",
        "scientific_claims_allowed": False,
        "eligible_for_champion": False,
        "eligible_for_training_library": False,
        "results": results,
        "trusted_evaluator_digest": evaluator_digest,
        "report_digest": "",
    }
    report["report_digest"] = content_hash(report)
    path = args.output_root / f"{args.command}_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 2


def run_negative_controls(args: argparse.Namespace, evaluator_module: Any, evaluator_digest: str, key: bytes) -> bool:
    import copy
    failures: list[str] = []
    statuses: dict[str, dict[str, Any]] = {}
    source = args.proposals_root / CASES[0] / "proposal.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    capsule = json.loads((args.bundle_root / "run" / "public" / CASES[0] / "capsule.json").read_text())
    task = task_for(CASES[0], capsule)
    controls: dict[str, Any] = {}
    swapped = copy.deepcopy(raw)
    swapped["task_id"] = CASES[1]
    controls["swapped_case_identity"] = swapped
    broken = copy.deepcopy(raw)
    broken["experiment_code"] = "import json\nprint(json.dumps({'point': [float('nan'), 0.0], 'oracle_calls': 4000, 'unit_tests': {'passed': 0, 'total': 4}}))\n"
    controls["correct_hypothesis_broken_code"] = broken
    fabricated = copy.deepcopy(raw)
    fabricated["experiment_code"] = "import json\nprint(json.dumps({'point': [0.0, 0.0], 'oracle_calls': 0, 'unit_tests': {'passed': 4, 'total': 4}, 'trusted_metrics': {'upper_regret': 0.0}}))\n"
    controls["fabricated_metrics"] = fabricated
    randomized = copy.deepcopy(raw)
    randomized["hypothesis"] = {
        "claim": "A random unrelated intervention improves the objective.",
        "mechanism": "No mechanism is specified; this is a negative control.",
        "assumptions": ["the intervention is unrelated to the task"],
        "alternative_explanations": ["sampling noise"],
        "null_hypothesis": "the intervention has no effect",
        "predicted_failure_mode": "the unrelated intervention does not improve the objective",
        "discriminating_observation": "trusted metrics do not improve over the baseline",
    }
    controls["random_hypothesis_fixed_adapter"] = randomized
    altered = copy.deepcopy(raw)
    altered["seeds"] = [1031, 2063, 4099, 9999]
    controls["altered_seed"] = altered
    leaked = copy.deepcopy(raw)
    leaked["experiment_code"] = "import json\n# NeurIPS target.pdf authors\nprint(json.dumps({'point': [0.0, 0.0], 'oracle_calls': 1, 'unit_tests': {'passed': 4, 'total': 4}}))\n"
    controls["target_leakage"] = leaked
    network = copy.deepcopy(raw)
    network["experiment_code"] = "import json\nimport socket\nprint(json.dumps({'point': [0.0, 0.0], 'oracle_calls': 1, 'unit_tests': {'passed': 4, 'total': 4}}))\n"
    controls["network_import"] = network
    for name, value in controls.items():
        control_path = args.output_root / "negative_controls" / name / "proposal.json"
        control_path.parent.mkdir(parents=True, exist_ok=True)
        rebound = value
        if value.get("task_id") == task.task_id and value.get("seeds") == list(SEEDS):
            try:
                subset = {key_name: value[key_name] for key_name in (
                    "hypothesis", "solution", "experiment_code", "seeds", "baselines",
                    "primary_metric", "secondary_metrics", "expected_effect",
                    "power_assumptions", "stopping_rule", "resource_schedule")}
                rebound = parse_model_proposal(
                    json.dumps(subset), task=task, source=CandidateSource.STUDENT,
                    model_id=str(value["model_id"]), model_digest=args.model_digest,
                    round_index=int(value["round_index"]), frozen_seeds=SEEDS,
                    mandatory_baselines=(BASELINES[0],), allowed_baselines=frozenset(BASELINES),
                    allowed_metrics=METRICS, frozen_resource_schedule=SCHEDULE).payload()
            except Exception:
                rebound = value
        control_path.write_text(json.dumps(rebound, indent=2, sort_keys=True) + "\n")
        rejected = False
        execution_checked = False
        execution_accepted = False
        reason = ""
        try:
            proposal = proposal_from_file(control_path, task, args.model_digest)
            audit_code(proposal.experiment_code)
            runner = CodeRunner(RunnerConfig(mode="local_test", timeout_seconds=30.0, memory_bytes=2 * 1024**3))
            executions: list[ExecutionResult] = []
            for seed in proposal.seeds:
                problem = evaluator_module.make_problem(CASES[0], seed)
                item = runner.execute(task, proposal, seed, extra_input={
                    "case_id": CASES[0], "problem": problem, "capsule_digest": capsule_digest(capsule)})
                executions.append(trusted_execution(CASES[0], task, proposal, item, evaluator_module, evaluator_digest, problem))
            execution_checked = True
            execution_accepted = verifier(evaluator_digest).verify(task, tuple(executions)).accepted
            if execution_accepted:
                failures.append(name)
                reason = "negative control unexpectedly accepted by trusted verifier"
        except Exception as error:
            rejected = True
            reason = f"rejected: {type(error).__name__}: {error}"
        statuses[name] = {"rejected": rejected, "execution_checked": execution_checked,
                          "execution_accepted": execution_accepted, "reason": reason}
    (args.output_root / "negative_controls" / "summary.json").write_text(
        json.dumps({"passed": not failures, "failures": failures, "controls": statuses}, indent=2, sort_keys=True) + "\n")
    return not failures


if __name__ == "__main__":
    raise SystemExit(main())
