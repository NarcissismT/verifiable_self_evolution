from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from .contracts import (
    CandidateSource,
    ExperimentProposal,
    Split,
    StructuredHypothesis,
    Task,
    TrajectoryRecord,
)
from .runner import CodeRunner, RunnerConfig
from .store import TrajectoryRouter
from .verifier import DeclarativeHardVerifier, MetricRule


TOY_VERIFIER_VERSION = "quadratic-box-v1"


def make_tasks(counts: dict[str, int], seed: int = 20260819) -> list[Task]:
    tasks: list[Task] = []
    split_order = (Split.TRAIN, Split.DEV, Split.PROMOTION, Split.HELDOUT, Split.OOD)
    offset = 0
    for split in split_order:
        count = int(counts[split.value])
        for index in range(count):
            task_number = offset + index
            promotion_stratum = (
                "ood" if split is Split.PROMOTION and index >= 8 else "id"
            )
            if split is Split.OOD or promotion_stratum == "ood":
                curvature = 10.0 + (task_number % 5)
                optimum = -7.0 + 0.5 * (task_number % 9)
            else:
                curvature = 1.0 + (task_number % 5)
                optimum = -2.0 + 0.4 * (task_number % 11)
                if abs(optimum) < 0.25:
                    optimum = 0.8
            lower, upper = -8.0, 8.0
            optimum = max(lower + 0.25, min(upper - 0.25, optimum))
            tasks.append(
                Task(
                    task_id=f"quad_{split.value}_{index:04d}",
                    family="quadratic_box_toy",
                    split=split,
                    statement=(
                        "Find a verifiable minimizer of a one-dimensional boxed "
                        "quadratic and distinguish it from the zero baseline."
                    ),
                    instance={
                        "curvature": curvature,
                        "optimum": optimum,
                        "lower": lower,
                        "upper": upper,
                        "baseline_x": 0.0,
                        "task_seed": seed + task_number,
                    },
                    verifier_version=TOY_VERIFIER_VERSION,
                    tags=("synthetic", split.value, promotion_stratum),
                )
            )
        offset += count
    return tasks


def verifier() -> DeclarativeHardVerifier:
    return DeclarativeHardVerifier(
        verifier_version=TOY_VERIFIER_VERSION,
        trusted_evaluator_digest="quadratic-box-trusted-evaluator-v1",
        metric_rules=(
            MetricRule("solution_error", "solution_error", "min", 1e-8, 0.0, 1.0),
            MetricRule("objective_gap", "objective_gap", "min", 1e-10, 0.0, 1.0),
            MetricRule("kkt_residual", "kkt_residual", "min", 1e-8, 0.0, 1.0),
        ),
        max_runtime_seconds=5.0,
        min_unit_pass_rate=1.0,
        min_power=0.8,
        vds_weights={
            "empirical": 0.35,
            "hypothesis": 0.20,
            "experiment": 0.20,
            "novelty": 0.15,
            "calibration": 0.10,
        },
    )


def candidate_code() -> str:
    return r'''
import json
import sys

payload = json.load(sys.stdin)
solution = payload["solution"]
x = float(solution["x"])
print(json.dumps({"x": x, "seed": int(payload["seed"])}))
'''


def attach_trusted_metrics(task: Task, execution):
    import math

    try:
        x = float(execution.candidate_result["x"])
    except (KeyError, TypeError, ValueError):
        return replace(
            execution,
            trusted_metrics={},
            trusted_evaluator_digest="quadratic-box-trusted-evaluator-v1",
        )
    curvature = float(task.instance["curvature"])
    optimum = float(task.instance["optimum"])
    lower = float(task.instance["lower"])
    upper = float(task.instance["upper"])
    baseline_x = float(task.instance["baseline_x"])

    def objective(value: float) -> float:
        return 0.5 * curvature * (value - optimum) ** 2

    gradient = curvature * (x - optimum)
    if lower < x < upper:
        kkt_residual = abs(gradient)
    elif x <= lower:
        kkt_residual = max(0.0, -gradient)
    else:
        kkt_residual = max(0.0, gradient)
    unit_total = 4
    unit_passed = sum(
        (
            int(math.isfinite(x)),
            int(lower <= x <= upper),
            int(objective(x) >= 0.0),
            int(math.isfinite(kkt_residual)),
        )
    )
    return replace(
        execution,
        trusted_metrics={
            "solution_error": abs(x - optimum),
            "objective_gap": max(0.0, objective(x) - objective(optimum)),
            "kkt_residual": kkt_residual,
            "unit_tests_passed": unit_passed,
            "unit_tests_total": unit_total,
            "quality": -objective(x),
            "baseline_quality": -objective(baseline_x),
            "resource_quality_curve": [
                {"budget": budget, "quality": -objective(x)}
                for budget in execution.preregistered_resource_schedule
            ],
            "vds_empirical": float(
                abs(x - optimum) <= 1e-8
            ),
            "vds_hypothesis": 1.0,
            "vds_experiment": 1.0,
            "vds_novelty": 0.5,
            "vds_calibration": 1.0,
        },
        trusted_evaluator_digest="quadratic-box-trusted-evaluator-v1",
    )


def proposal(
    task: Task,
    x: float,
    source: CandidateSource,
    candidate_id: str,
) -> ExperimentProposal:
    return ExperimentProposal(
        candidate_id=candidate_id,
        task_id=task.task_id,
        source=source,
        model_id=source.value,
        model_digest=f"toy-{source.value}",
        hypothesis=StructuredHypothesis(
            claim="The boxed quadratic has a unique stationary minimizer.",
            mechanism="Positive curvature makes the stationary point globally optimal.",
            assumptions=("positive curvature", "closed finite interval"),
            alternative_explanations=(
                "The zero baseline may appear competitive when the optimum is near zero.",
            ),
            null_hypothesis="The proposed point does not improve over the zero baseline.",
            predicted_failure_mode="A nonstationary point fails KKT or objective-gap checks.",
            discriminating_observation=(
                "KKT residual and objective gap separate the candidate from zero."
            ),
        ),
        solution={"x": x},
        experiment_code=candidate_code(),
        seeds=(101, 211, 307),
        baselines=("zero",),
        primary_metric="objective_gap",
        secondary_metrics=("solution_error", "kkt_residual"),
        expected_effect={"direction": "lower", "minimum_delta": 0.1},
        power_assumptions={"alpha": 0.05, "target_power": 0.8},
        stopping_rule="Run exactly the three preregistered seeds.",
        resource_schedule=(1.0, 2.0, 3.0),
        round_index=0,
    )


def run_training_smoke(root: Path, counts: dict[str, int]) -> dict[str, Any]:
    tasks = make_tasks(counts)
    runner = CodeRunner(RunnerConfig(mode="local_test", timeout_seconds=5.0))
    verifier_instance = verifier()
    router = TrajectoryRouter(root)
    outcomes: dict[str, Any] = {"accepted": 0, "rejected": 0, "paths": []}
    for index, task in enumerate(tasks):
        if task.split not in {Split.TRAIN, Split.DEV}:
            continue
        optimum = float(task.instance["optimum"])
        x = optimum if index % 2 == 0 else 0.0
        source = CandidateSource.CHAMPION if index % 2 == 0 else CandidateSource.VARIANT
        candidate = proposal(task, x, source, f"toy-candidate-{index:04d}")
        executions = tuple(
            attach_trusted_metrics(
                task, runner.execute(task, candidate, execution_seed)
            )
            for execution_seed in candidate.seeds
        )
        report = verifier_instance.verify(task, executions)
        record = TrajectoryRecord(
            task=task,
            proposal=candidate,
            executions=executions,
            verification=report,
            created_at_utc="2026-08-19T00:00:00Z",
        )
        destination = router.route(record)
        outcomes["paths"].append(str(destination))
        outcomes["accepted" if report.accepted else "rejected"] += 1
    return outcomes
