from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .hashing import content_hash


class Split(str, Enum):
    TRAIN = "train"
    DEV = "dev"
    PROMOTION = "promotion"
    HELDOUT = "heldout"
    OOD = "ood"


class CandidateSource(str, Enum):
    STUDENT = "student"
    CHAMPION = "champion"
    LEGACY = "legacy"
    CLOSED_TEACHER = "closed_teacher"
    VARIANT = "variant"


class TeacherPurpose(str, Enum):
    BOOTSTRAP = "bootstrap_generation"
    HARD_STATE_REPAIR = "hard_state_repair"


@dataclass(frozen=True)
class Task:
    task_id: str
    family: str
    split: Split
    statement: str
    instance: dict[str, Any]
    verifier_version: str
    tags: tuple[str, ...] = ()

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        value["split"] = self.split.value
        value["tags"] = list(self.tags)
        return value

    @property
    def digest(self) -> str:
        return content_hash(self.payload())


@dataclass(frozen=True)
class StructuredHypothesis:
    claim: str
    mechanism: str
    assumptions: tuple[str, ...]
    alternative_explanations: tuple[str, ...]
    null_hypothesis: str
    predicted_failure_mode: str
    discriminating_observation: str


@dataclass(frozen=True)
class ExperimentProposal:
    candidate_id: str
    task_id: str
    source: CandidateSource
    model_id: str
    model_digest: str
    hypothesis: StructuredHypothesis
    solution: dict[str, Any]
    experiment_code: str
    seeds: tuple[int, ...]
    baselines: tuple[str, ...]
    primary_metric: str
    secondary_metrics: tuple[str, ...]
    expected_effect: dict[str, Any]
    power_assumptions: dict[str, Any]
    stopping_rule: str
    resource_schedule: tuple[float, ...]
    round_index: int
    parent_candidate_ids: tuple[str, ...] = ()

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        value["source"] = self.source.value
        return value

    def validate(self) -> None:
        if not self.candidate_id or not self.task_id or not self.model_digest:
            raise ValueError("candidate, task, and model identities are required")
        if not self.hypothesis.claim or not self.hypothesis.null_hypothesis:
            raise ValueError("claim and null hypothesis are required")
        if not self.hypothesis.alternative_explanations:
            raise ValueError("at least one alternative explanation is required")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("execution seeds must be nonempty and unique")
        if not self.baselines:
            raise ValueError("at least one frozen baseline is required")
        if not self.primary_metric:
            raise ValueError("primary metric is required")
        if len(self.resource_schedule) < 2:
            raise ValueError("resource-to-quality evaluation needs at least two budgets")
        if tuple(sorted(self.resource_schedule)) != self.resource_schedule:
            raise ValueError("resource schedule must be sorted")
        if any(value <= 0 for value in self.resource_schedule):
            raise ValueError("resource budgets must be positive")

    @property
    def digest(self) -> str:
        self.validate()
        return content_hash(self.payload())


@dataclass(frozen=True)
class Metric:
    name: str
    value: float
    direction: str
    threshold: float | None = None
    unit: str = ""

    def passes(self) -> bool:
        if self.threshold is None:
            return True
        if self.direction == "min":
            return self.value <= self.threshold
        if self.direction == "max":
            return self.value >= self.threshold
        raise ValueError(f"unsupported metric direction: {self.direction}")


@dataclass(frozen=True)
class ExecutionResult:
    candidate_id: str
    task_id: str
    proposal_digest: str
    preregistered_resource_schedule: tuple[float, ...]
    seed: int
    exit_code: int
    timed_out: bool
    runtime_seconds: float
    stdout: str
    stderr: str
    candidate_result: dict[str, Any]
    trusted_metrics: dict[str, Any]
    trusted_evaluator_digest: str
    resource: dict[str, float]


@dataclass(frozen=True)
class VerificationReport:
    candidate_id: str
    task_id: str
    split: Split
    verifier_version: str
    accepted: bool
    hard_failures: tuple[str, ...]
    metrics: tuple[Metric, ...]
    execution_digests: tuple[str, ...]
    quality_score: float
    vds_score: float = 0.0
    vds_components: dict[str, float] = field(default_factory=dict)
    binary_success: bool = False
    report_digest: str = field(default="")

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        value["split"] = self.split.value
        return value

    def sealed(self) -> "VerificationReport":
        value = self.payload()
        value["report_digest"] = ""
        return VerificationReport(**{
            **asdict(self),
            "report_digest": content_hash(value),
        })


@dataclass(frozen=True)
class TrajectoryRecord:
    task: Task
    proposal: ExperimentProposal
    executions: tuple[ExecutionResult, ...]
    verification: VerificationReport
    created_at_utc: str

    def payload(self) -> dict[str, Any]:
        return {
            "task": self.task.payload(),
            "proposal": self.proposal.payload(),
            "executions": [asdict(item) for item in self.executions],
            "verification": self.verification.payload(),
            "created_at_utc": self.created_at_utc,
        }
