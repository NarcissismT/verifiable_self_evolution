from __future__ import annotations

import json
from typing import Any

from .contracts import (
    CandidateSource,
    ExperimentProposal,
    StructuredHypothesis,
    Task,
)
from .hashing import content_hash


TOP_LEVEL_KEYS = {
    "hypothesis",
    "solution",
    "experiment_code",
    "seeds",
    "baselines",
    "primary_metric",
    "secondary_metrics",
    "expected_effect",
    "power_assumptions",
    "stopping_rule",
    "resource_schedule",
}
HYPOTHESIS_KEYS = {
    "claim",
    "mechanism",
    "assumptions",
    "alternative_explanations",
    "null_hypothesis",
    "predicted_failure_mode",
    "discriminating_observation",
}


def _exact_keys(value: dict[str, Any], expected: set[str], location: str) -> None:
    missing = expected - set(value)
    extra = set(value) - expected
    if missing or extra:
        raise ValueError(
            f"{location} schema mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )


def parse_model_proposal(
    raw: str,
    *,
    task: Task,
    source: CandidateSource,
    model_id: str,
    model_digest: str,
    round_index: int,
    frozen_seeds: tuple[int, ...],
    mandatory_baselines: tuple[str, ...],
    allowed_baselines: frozenset[str],
    allowed_metrics: frozenset[str],
    frozen_resource_schedule: tuple[float, ...],
    parent_candidate_ids: tuple[str, ...] = (),
    max_code_bytes: int = 256_000,
) -> ExperimentProposal:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"model proposal is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("model proposal must be a JSON object")
    _exact_keys(value, TOP_LEVEL_KEYS, "proposal")
    hypothesis = value["hypothesis"]
    if not isinstance(hypothesis, dict):
        raise ValueError("hypothesis must be a JSON object")
    _exact_keys(hypothesis, HYPOTHESIS_KEYS, "hypothesis")
    code = value["experiment_code"]
    if not isinstance(code, str) or not code.strip():
        raise ValueError("experiment_code must be a nonempty string")
    if len(code.encode("utf-8")) > max_code_bytes:
        raise ValueError("experiment_code exceeds the frozen size limit")
    proposed_seeds = tuple(int(item) for item in value["seeds"])
    if proposed_seeds != frozen_seeds:
        raise ValueError("model proposal changed the frozen execution seeds")
    proposed_baselines = tuple(str(item) for item in value["baselines"])
    if not set(mandatory_baselines).issubset(proposed_baselines):
        raise ValueError("model proposal omitted a mandatory baseline")
    if not set(proposed_baselines).issubset(allowed_baselines):
        raise ValueError("model proposal requested an unregistered baseline")
    proposed_primary = str(value["primary_metric"])
    proposed_secondary = tuple(str(item) for item in value["secondary_metrics"])
    if not {proposed_primary, *proposed_secondary}.issubset(allowed_metrics):
        raise ValueError("model proposal requested an unregistered metric")
    proposed_schedule = tuple(float(item) for item in value["resource_schedule"])
    if proposed_schedule != frozen_resource_schedule:
        raise ValueError("model proposal changed the frozen resource schedule")

    candidate_id = "candidate-" + content_hash(
        {
            "raw": value,
            "task_digest": task.digest,
            "source": source.value,
            "model_digest": model_digest,
            "round_index": round_index,
        }
    )[:20]
    proposal = ExperimentProposal(
        candidate_id=candidate_id,
        task_id=task.task_id,
        source=source,
        model_id=model_id,
        model_digest=model_digest,
        hypothesis=StructuredHypothesis(
            claim=str(hypothesis["claim"]),
            mechanism=str(hypothesis["mechanism"]),
            assumptions=tuple(str(item) for item in hypothesis["assumptions"]),
            alternative_explanations=tuple(
                str(item) for item in hypothesis["alternative_explanations"]
            ),
            null_hypothesis=str(hypothesis["null_hypothesis"]),
            predicted_failure_mode=str(hypothesis["predicted_failure_mode"]),
            discriminating_observation=str(
                hypothesis["discriminating_observation"]
            ),
        ),
        solution=dict(value["solution"]),
        experiment_code=code,
        seeds=frozen_seeds,
        baselines=proposed_baselines,
        primary_metric=proposed_primary,
        secondary_metrics=proposed_secondary,
        expected_effect=dict(value["expected_effect"]),
        power_assumptions=dict(value["power_assumptions"]),
        stopping_rule=str(value["stopping_rule"]),
        resource_schedule=frozen_resource_schedule,
        round_index=int(round_index),
        parent_candidate_ids=parent_candidate_ids,
    )
    proposal.validate()
    return proposal
