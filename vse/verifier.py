from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import fmean, pstdev
from typing import Any, Iterable

from .contracts import ExecutionResult, Metric, Task, VerificationReport
from .runner import execution_digest


@dataclass(frozen=True)
class MetricRule:
    name: str
    result_key: str
    direction: str
    hard_threshold: float
    good_value: float
    bad_value: float
    aggregation: str = "worst"

    def aggregate(self, executions: Iterable[ExecutionResult]) -> float:
        values = [float(item.trusted_metrics[self.result_key]) for item in executions]
        if not values:
            return float("nan")
        if self.aggregation == "mean":
            return fmean(values)
        if self.aggregation != "worst":
            raise ValueError(f"unsupported aggregation: {self.aggregation}")
        return max(values) if self.direction == "min" else min(values)

    def quality(self, value: float) -> float:
        if not math.isfinite(value) or self.good_value == self.bad_value:
            return 0.0
        if self.direction == "min":
            raw = (self.bad_value - value) / (self.bad_value - self.good_value)
        elif self.direction == "max":
            raw = (value - self.bad_value) / (self.good_value - self.bad_value)
        else:
            raise ValueError(f"unsupported direction: {self.direction}")
        return min(1.0, max(0.0, raw))


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def approximate_paired_power(differences: list[float], alpha: float = 0.05) -> float:
    """Normal approximation for a two-sided paired mean test.

    This is a design diagnostic, not a replacement for the frozen formal
    analysis. Constant nonzero differences have power one; constant zero
    differences have power zero.
    """

    if len(differences) < 2:
        return 0.0
    mean = abs(fmean(differences))
    spread = pstdev(differences)
    if spread == 0.0:
        return 1.0 if mean > 0.0 else 0.0
    noncentrality = mean * math.sqrt(len(differences)) / spread
    # 1.95996 is the standard-normal 97.5th percentile.
    critical = 1.959963984540054
    return normal_cdf(-critical - noncentrality) + 1.0 - normal_cdf(
        critical - noncentrality
    )


class DeclarativeHardVerifier:
    def __init__(
        self,
        verifier_version: str,
        trusted_evaluator_digest: str,
        metric_rules: tuple[MetricRule, ...],
        max_runtime_seconds: float,
        min_unit_pass_rate: float = 1.0,
        min_power: float = 0.8,
        vds_weights: dict[str, float] | None = None,
    ):
        self.verifier_version = verifier_version
        self.trusted_evaluator_digest = trusted_evaluator_digest
        self.metric_rules = metric_rules
        self.max_runtime_seconds = max_runtime_seconds
        self.min_unit_pass_rate = min_unit_pass_rate
        self.min_power = min_power
        self.vds_weights = dict(vds_weights or {})
        if self.vds_weights:
            if set(self.vds_weights) != {
                "empirical",
                "hypothesis",
                "experiment",
                "novelty",
                "calibration",
            }:
                raise ValueError("VDS weights must have exactly five named components")
            if not math.isclose(sum(self.vds_weights.values()), 1.0, abs_tol=1e-12):
                raise ValueError("VDS weights must sum to one")

    def verify(
        self, task: Task, executions: tuple[ExecutionResult, ...]
    ) -> VerificationReport:
        failures: list[str] = []
        metrics: list[Metric] = []
        if task.verifier_version != self.verifier_version:
            failures.append("verifier_version_mismatch")
        if not executions:
            failures.append("no_executions")
        proposal_digests = {item.proposal_digest for item in executions}
        if len(proposal_digests) != 1:
            failures.append("proposal_changed_across_seeds")
        for execution in executions:
            if execution.task_id != task.task_id:
                failures.append(f"wrong_task:{execution.seed}")
            if execution.exit_code != 0:
                failures.append(f"execution_failed:{execution.seed}:{execution.exit_code}")
            if execution.timed_out:
                failures.append(f"execution_timed_out:{execution.seed}")
            if execution.runtime_seconds > self.max_runtime_seconds:
                failures.append(f"runtime_budget_exceeded:{execution.seed}")
            if not execution.trusted_evaluator_digest:
                failures.append(f"missing_trusted_evaluator:{execution.seed}")
            elif execution.trusted_evaluator_digest != self.trusted_evaluator_digest:
                failures.append(f"trusted_evaluator_mismatch:{execution.seed}")

        healthy = [item for item in executions if item.exit_code == 0 and not item.timed_out]
        required_keys = {rule.result_key for rule in self.metric_rules}
        required_keys.update(
            {
                "unit_tests_passed",
                "unit_tests_total",
                "quality",
                "baseline_quality",
                "resource_quality_curve",
            }
        )
        if self.vds_weights:
            required_keys.update(f"vds_{name}" for name in self.vds_weights)
        for execution in healthy:
            missing = required_keys - set(execution.trusted_metrics)
            for key in sorted(missing):
                failures.append(f"missing_result:{execution.seed}:{key}")

        if healthy and not any(item for item in failures if item.startswith("missing_result:")):
            passed = sum(
                float(item.trusted_metrics["unit_tests_passed"]) for item in healthy
            )
            total = sum(
                float(item.trusted_metrics["unit_tests_total"]) for item in healthy
            )
            unit_rate = passed / total if total > 0 else 0.0
            unit_metric = Metric(
                "unit_pass_rate", unit_rate, "max", self.min_unit_pass_rate, "fraction"
            )
            metrics.append(unit_metric)
            if not unit_metric.passes():
                failures.append("unit_tests_failed")

            for rule in self.metric_rules:
                value = rule.aggregate(healthy)
                metric = Metric(
                    rule.name,
                    value,
                    rule.direction,
                    rule.hard_threshold,
                )
                metrics.append(metric)
                if not metric.passes():
                    failures.append(f"hard_metric_failed:{rule.name}")

            differences = [
                float(item.trusted_metrics["quality"])
                - float(item.trusted_metrics["baseline_quality"])
                for item in healthy
            ]
            power = approximate_paired_power(differences)
            power_metric = Metric(
                "discrimination_power", power, "max", self.min_power, "fraction"
            )
            metrics.append(power_metric)
            if not power_metric.passes():
                failures.append("insufficient_discrimination_power")

            resource_aucs: list[float] = []
            for item in healthy:
                curve = item.trusted_metrics["resource_quality_curve"]
                if not isinstance(curve, list):
                    failures.append(f"invalid_resource_curve:{item.seed}")
                    continue
                try:
                    budgets = tuple(float(point["budget"]) for point in curve)
                    qualities = tuple(float(point["quality"]) for point in curve)
                except (KeyError, TypeError, ValueError):
                    failures.append(f"invalid_resource_curve:{item.seed}")
                    continue
                if budgets != item.preregistered_resource_schedule:
                    failures.append(f"resource_schedule_mismatch:{item.seed}")
                    continue
                if len(budgets) < 2 or any(
                    right <= left for left, right in zip(budgets, budgets[1:])
                ):
                    failures.append(f"invalid_resource_schedule:{item.seed}")
                    continue
                if not all(math.isfinite(value) for value in qualities):
                    failures.append(f"nonfinite_resource_quality:{item.seed}")
                    continue
                area = sum(
                    0.5 * (qualities[index] + qualities[index + 1])
                    * (budgets[index + 1] - budgets[index])
                    for index in range(len(budgets) - 1)
                )
                resource_aucs.append(area / (budgets[-1] - budgets[0]))
            if resource_aucs:
                metrics.append(
                    Metric(
                        "resource_quality_auc",
                        fmean(resource_aucs),
                        "max",
                        None,
                        "quality",
                    )
                )

        quality_parts: list[float] = []
        metric_by_name = {metric.name: metric for metric in metrics}
        for rule in self.metric_rules:
            if rule.name in metric_by_name:
                quality_parts.append(rule.quality(metric_by_name[rule.name].value))
        quality_score = fmean(quality_parts) if quality_parts else 0.0
        vds_components: dict[str, float] = {}
        vds_score = 0.0
        if self.vds_weights and healthy:
            for name in self.vds_weights:
                values = [float(item.trusted_metrics[f"vds_{name}"]) for item in healthy]
                if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
                    failures.append(f"invalid_vds_component:{name}")
                vds_components[name] = fmean(values)
            if not failures:
                vds_score = sum(
                    self.vds_weights[name] * vds_components[name]
                    for name in self.vds_weights
                )
            metrics.append(Metric("vds", vds_score, "max", None, "score"))
        report = VerificationReport(
            candidate_id=(executions[0].candidate_id if executions else "missing"),
            task_id=task.task_id,
            split=task.split,
            verifier_version=self.verifier_version,
            accepted=not failures,
            hard_failures=tuple(sorted(set(failures))),
            metrics=tuple(metrics),
            execution_digests=tuple(execution_digest(item) for item in executions),
            quality_score=quality_score,
            vds_score=vds_score,
            vds_components=vds_components,
            binary_success=not failures,
        )
        return report.sealed()
