from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from statistics import fmean, median
from typing import Iterable

from .contracts import Split
from .hashing import content_hash


VDS_COMPONENTS = frozenset(
    {"empirical", "hypothesis", "experiment", "novelty", "calibration"}
)
DEFAULT_ROLLOUT_SEEDS = (101, 211, 307, 401)
DEFAULT_ADAPTER_SEEDS = (17, 29, 43)


@dataclass(frozen=True)
class EvaluationCell:
    evaluation_run_id: str
    evaluation_phase: str
    promotion_attempt: int
    adapter_seed: int
    checkpoint_id: str
    checkpoint_digest: str
    task_id: str
    task_digest: str
    split: Split
    stratum: str
    seed: int
    hard_pass: bool
    executable_pass: bool
    fabricated_result: bool
    target_leakage: bool
    vds_score: float
    vds_components: dict[str, float]
    cost_units: float
    runtime_seconds: float
    manifest_digest: str
    evaluator_digest: str
    contamination_audit_digest: str

    @property
    def key(self) -> tuple[str, int, int]:
        return self.task_id, self.seed, self.adapter_seed


@dataclass(frozen=True)
class PromotionPolicy:
    minimum_tasks: int = 12
    promotion_id_tasks: int = 8
    promotion_ood_tasks: int = 4
    minimum_vds_delta: float = 0.05
    alpha: float = 0.10
    bootstrap_replicates: int = 10000
    minimum_hard_pass_rate: float = 0.95
    minimum_executable_pass_rate: float = 0.95
    maximum_component_drop: float = 0.02
    maximum_regressed_task_fraction: float = 0.0
    maximum_cost_ratio: float = 1.10
    minimum_adapter_seed_passes: int = 2
    maximum_promotion_attempts: int = 3
    rollout_seeds: tuple[int, ...] = DEFAULT_ROLLOUT_SEEDS
    adapter_seeds: tuple[int, ...] = DEFAULT_ADAPTER_SEEDS
    expected_manifest_digest: str | None = None
    expected_evaluator_digest: str | None = None
    expected_contamination_digest: str | None = None


@dataclass(frozen=True)
class FinalPolicy:
    minimum_id_tasks: int = 24
    minimum_ood_tasks: int = 16
    alpha: float = 0.05
    bootstrap_replicates: int = 10000
    minimum_id_vds_delta: float = 0.05
    ood_noninferiority_margin: float = -0.03
    minimum_hard_pass_rate: float = 0.95
    minimum_executable_pass_rate: float = 0.95
    rollout_seeds: tuple[int, ...] = DEFAULT_ROLLOUT_SEEDS
    adapter_seeds: tuple[int, ...] = DEFAULT_ADAPTER_SEEDS
    reference_checkpoint_id: str | None = None
    reference_checkpoint_digest: str | None = None
    expected_manifest_digest: str | None = None
    expected_evaluator_digest: str | None = None
    expected_contamination_digest: str | None = None


def promotion_policy_from_config(config: dict) -> PromotionPolicy:
    values = config["promotion"]
    evolution = config["evolution"]
    return PromotionPolicy(
        minimum_tasks=int(values["tasks"]),
        promotion_id_tasks=int(values["id_tasks"]),
        promotion_ood_tasks=int(values["ood_tasks"]),
        minimum_vds_delta=float(values["vds_delta_min"]),
        alpha=float(values["bootstrap_ci_alpha"]),
        bootstrap_replicates=int(values["bootstrap_replicates"]),
        minimum_hard_pass_rate=float(values["hard_pass_rate_min"]),
        minimum_executable_pass_rate=float(values["executable_pass_rate_min"]),
        maximum_component_drop=abs(float(values["component_delta_min"])),
        maximum_regressed_task_fraction=float(values.get("regressed_task_fraction_max", 0.0)),
        maximum_cost_ratio=float(values["cost_ratio_max"]),
        minimum_adapter_seed_passes=int(values["adapter_seed_passes_min"]),
        maximum_promotion_attempts=int(evolution["maximum_promotion_attempts"]),
        rollout_seeds=tuple(int(seed) for seed in config["seeds"]["rollout"]),
        adapter_seeds=tuple(int(seed) for seed in config["seeds"]["adapter"]),
    )


def final_policy_from_config(config: dict) -> FinalPolicy:
    values = config["final"]
    return FinalPolicy(
        minimum_id_tasks=int(values["id_tasks"]),
        minimum_ood_tasks=int(values["ood_tasks"]),
        alpha=float(values["bootstrap_ci_alpha"]),
        bootstrap_replicates=int(values["bootstrap_replicates"]),
        minimum_id_vds_delta=float(values["vds_delta_min"]),
        ood_noninferiority_margin=float(values["ood_vds_ci_lower_min"]),
        minimum_hard_pass_rate=float(values["hard_pass_rate_min"]),
        minimum_executable_pass_rate=float(values["executable_pass_rate_min"]),
        rollout_seeds=tuple(int(seed) for seed in config["seeds"]["rollout"]),
        adapter_seeds=tuple(int(seed) for seed in config["seeds"]["adapter"]),
        reference_checkpoint_id=str(values.get("reference_checkpoint_id", "base")),
        reference_checkpoint_digest=values.get("reference_checkpoint_digest"),
    )


@dataclass(frozen=True)
class SplitDecision:
    split: Split
    task_count: int
    matched_cells: int
    hard_pass_rate: float
    executable_pass_rate: float
    fabricated_result_count: int
    target_leakage_count: int
    regressed_task_fraction: float
    mean_vds_delta: float
    vds_delta_ci_low: float
    vds_delta_ci_high: float
    component_deltas: dict[str, float]
    cost_ratio: float
    adapter_seed_passes: int
    adapter_seed_median_delta: float
    passed: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    decision_kind: str
    candidate_checkpoint_digest: str
    champion_checkpoint_digest: str
    manifest_digest: str
    evaluator_digest: str
    split_decisions: tuple[SplitDecision, ...]
    failures: tuple[str, ...]
    decision_digest: str = ""

    def sealed(self) -> "PromotionDecision":
        value = asdict(self)
        value["decision_digest"] = ""
        return PromotionDecision(
            **{**asdict(self), "decision_digest": content_hash(value)}
        )


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _paper_bootstrap(
    deltas: dict[str, float], replicates: int, alpha: float, seed: int
) -> tuple[float, float, float]:
    task_ids = sorted(deltas)
    if not task_ids:
        return float("nan"), float("nan"), float("nan")
    point = fmean(deltas[task_id] for task_id in task_ids)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(replicates):
        sampled_tasks = [rng.choice(task_ids) for _ in task_ids]
        draws.append(fmean(deltas[task_id] for task_id in sampled_tasks))
    return point, _quantile(draws, alpha / 2.0), _quantile(draws, 1.0 - alpha / 2.0)


def _single_value(name: str, cells: list[EvaluationCell], field: str) -> str:
    values = {str(getattr(cell, field)) for cell in cells}
    if len(values) != 1:
        raise ValueError(f"{name} cells disagree on {field}: {sorted(values)}")
    return next(iter(values))


def _common_checks(
    candidate: list[EvaluationCell],
    champion: list[EvaluationCell],
    *,
    phase: str,
    allowed_splits: set[Split],
    expected_manifest_digest: str | None = None,
    expected_evaluator_digest: str | None = None,
    expected_contamination_digest: str | None = None,
    expected_reference_checkpoint_id: str | None = None,
    expected_reference_checkpoint_digest: str | None = None,
) -> tuple[str, str, list[str]]:
    if not candidate or not champion:
        raise ValueError("candidate and champion receipts are required")
    all_cells = candidate + champion
    if {cell.evaluation_phase for cell in all_cells} != {phase}:
        raise ValueError(f"evaluation input must use phase={phase!r}")
    forbidden = {cell.split for cell in all_cells} - allowed_splits
    if forbidden:
        raise ValueError(f"evaluation input contains forbidden splits: {sorted(forbidden)}")
    manifest_digest = _single_value("all", all_cells, "manifest_digest")
    evaluator_digest = _single_value("all", all_cells, "evaluator_digest")
    _single_value("all", all_cells, "evaluation_run_id")
    contamination_digest = _single_value(
        "all", all_cells, "contamination_audit_digest"
    )
    if not contamination_digest:
        raise ValueError("contamination audit receipt is required")
    if expected_manifest_digest is not None and manifest_digest != expected_manifest_digest:
        raise ValueError("receipt manifest digest does not match frozen manifest")
    if expected_evaluator_digest is not None and evaluator_digest != expected_evaluator_digest:
        raise ValueError("receipt evaluator digest does not match frozen evaluator")
    if expected_contamination_digest is not None and contamination_digest != expected_contamination_digest:
        raise ValueError("receipt contamination digest does not match frozen audit")
    candidate_digest = _single_value("candidate", candidate, "checkpoint_digest")
    champion_digest = _single_value("champion", champion, "checkpoint_digest")
    failures: list[str] = []
    if candidate_digest == champion_digest:
        failures.append("candidate_is_champion")
    if expected_reference_checkpoint_id is not None:
        reference_ids = {cell.checkpoint_id for cell in champion}
        if reference_ids != {expected_reference_checkpoint_id}:
            raise ValueError("final comparator is not the frozen base checkpoint")
    if (
        expected_reference_checkpoint_digest is not None
        and champion_digest != expected_reference_checkpoint_digest
    ):
        raise ValueError("final comparator digest is not the frozen base checkpoint")
    return manifest_digest, evaluator_digest, failures


def _make_split_decision(
    candidate: list[EvaluationCell],
    champion: list[EvaluationCell],
    split: Split,
    *,
    minimum_tasks: int,
    alpha: float,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    minimum_vds_delta: float,
    minimum_hard_pass_rate: float,
    minimum_executable_pass_rate: float,
    maximum_regressed_task_fraction: float,
    expected_rollout_seeds: tuple[int, ...],
    expected_adapter_seeds: tuple[int, ...],
    maximum_component_drop: float | None,
    maximum_cost_ratio: float | None,
    adapter_gate: bool,
    minimum_adapter_seed_passes: int = 2,
    noninferiority_margin: float | None = None,
) -> SplitDecision:
    expected_rollout_seeds = tuple(sorted(set(expected_rollout_seeds)))
    expected_adapter_seeds = tuple(sorted(set(expected_adapter_seeds)))
    if not expected_rollout_seeds or not expected_adapter_seeds:
        raise ValueError("rollout and adapter seed registries must be nonempty")
    all_candidate_keys = [cell.key for cell in candidate if cell.split is split]
    all_champion_keys = [cell.key for cell in champion if cell.split is split]
    if len(all_candidate_keys) != len(set(all_candidate_keys)):
        raise ValueError("candidate contains duplicate evaluation cells")
    if len(all_champion_keys) != len(set(all_champion_keys)):
        raise ValueError("champion contains duplicate evaluation cells")
    cand = {cell.key: cell for cell in candidate if cell.split is split}
    champ = {cell.key: cell for cell in champion if cell.split is split}
    failures: list[str] = []
    if set(cand) != set(champ):
        failures.append("unmatched_evaluation_grid")
    keys = sorted(set(cand) & set(champ))
    task_ids = sorted({task_id for task_id, _, _ in keys})
    if len(task_ids) != minimum_tasks:
        failures.append("task_count_mismatch")
    for key in keys:
        if cand[key].task_digest != champ[key].task_digest:
            failures.append(f"task_digest_mismatch:{key[0]}")
    for task_id in task_ids:
        expected = {
            (task_id, rollout_seed, adapter_seed)
            for adapter_seed in expected_adapter_seeds
            for rollout_seed in expected_rollout_seeds
        }
        if {key for key in cand if key[0] == task_id} != expected:
            failures.append(f"candidate_seed_grid_mismatch:{task_id}")
        if {key for key in champ if key[0] == task_id} != expected:
            failures.append(f"champion_seed_grid_mismatch:{task_id}")
    for key in keys:
        for cells, label in ((cand[key], "candidate"), (champ[key], "champion")):
            if set(cells.vds_components) != VDS_COMPONENTS:
                failures.append(f"{label}_vds_component_schema_mismatch:{key[0]}")
            if any(
                not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
                for value in cells.vds_components.values()
            ):
                failures.append(f"{label}_vds_component_range_mismatch:{key[0]}")
            if not math.isfinite(float(cells.vds_score)) or not 0.0 <= float(cells.vds_score) <= 1.0:
                failures.append(f"{label}_vds_score_range_mismatch:{key[0]}")
    hard_pass_rate = fmean(float(cand[key].hard_pass) for key in keys) if keys else 0.0
    executable_rate = (
        fmean(float(cand[key].executable_pass) for key in keys) if keys else 0.0
    )
    if hard_pass_rate < minimum_hard_pass_rate:
        failures.append("hard_pass_rate_below_gate")
    if executable_rate < minimum_executable_pass_rate:
        failures.append("executable_pass_rate_below_gate")
    fabricated = sum(int(cand[key].fabricated_result) for key in keys)
    leaked = sum(int(cand[key].target_leakage) for key in keys)
    if fabricated:
        failures.append("fabricated_result_detected")
    if leaked:
        failures.append("target_leakage_detected")

    deltas: dict[str, list[float]] = {}
    component_deltas: dict[str, list[float]] = {}
    candidate_cost: list[float] = []
    champion_cost: list[float] = []
    for key in keys:
        deltas.setdefault(key[0], []).append(cand[key].vds_score - champ[key].vds_score)
        candidate_cost.append(cand[key].cost_units)
        champion_cost.append(champ[key].cost_units)
        for name in set(cand[key].vds_components) | set(champ[key].vds_components):
            component_deltas.setdefault(name, []).append(
                cand[key].vds_components.get(name, 0.0)
                - champ[key].vds_components.get(name, 0.0)
            )
    task_deltas = {task_id: fmean(values) for task_id, values in deltas.items()}
    regressed = (
        fmean(float(value < 0.0) for value in task_deltas.values())
        if task_deltas
        else 1.0
    )
    task_delta_values = {task_id: fmean(values) for task_id, values in deltas.items()}
    point, low, high = _paper_bootstrap(
        task_delta_values, bootstrap_replicates, alpha, bootstrap_seed
    )
    if regressed > maximum_regressed_task_fraction:
        failures.append("regressed_task_fraction_gate_failed")
    if noninferiority_margin is None:
        if point < minimum_vds_delta or not math.isfinite(low) or low <= 0.0:
            failures.append("vds_delta_gate_failed")
    elif not math.isfinite(low) or low <= noninferiority_margin:
        failures.append("ood_noninferiority_gate_failed")
    component_means = {
        name: fmean(values) for name, values in component_deltas.items()
    }
    if maximum_component_drop is not None and any(
        value < -maximum_component_drop for value in component_means.values()
    ):
        failures.append("vds_component_regression_gate_failed")
    cost_ratio = (
        fmean(candidate_cost) / fmean(champion_cost)
        if champion_cost and fmean(champion_cost) > 0.0
        else float("inf")
    )
    if maximum_cost_ratio is not None and cost_ratio > maximum_cost_ratio:
        failures.append("cost_ratio_gate_failed")

    adapter_passes = 0
    adapter_deltas: list[float] = []
    if adapter_gate:
        for adapter_seed in expected_adapter_seeds:
            adapter_keys = [key for key in keys if key[2] == adapter_seed]
            if not adapter_keys:
                continue
            adapter_delta = fmean(
                cand[key].vds_score - champ[key].vds_score for key in adapter_keys
            )
            adapter_deltas.append(adapter_delta)
            adapter_ok = all(
                cand[key].hard_pass
                and cand[key].executable_pass
                and not cand[key].fabricated_result
                and not cand[key].target_leakage
                for key in adapter_keys
            )
            if adapter_ok and adapter_delta >= minimum_vds_delta:
                adapter_passes += 1
        if adapter_passes < minimum_adapter_seed_passes or (
            adapter_deltas and median(adapter_deltas) < minimum_vds_delta
        ):
            failures.append("adapter_seed_gate_failed")

    return SplitDecision(
        split=split,
        task_count=len(task_ids),
        matched_cells=len(keys),
        hard_pass_rate=hard_pass_rate,
        executable_pass_rate=executable_rate,
        fabricated_result_count=fabricated,
        target_leakage_count=leaked,
        regressed_task_fraction=regressed,
        mean_vds_delta=point,
        vds_delta_ci_low=low,
        vds_delta_ci_high=high,
        component_deltas=component_means,
        cost_ratio=cost_ratio,
        adapter_seed_passes=adapter_passes,
        adapter_seed_median_delta=median(adapter_deltas) if adapter_deltas else float("nan"),
        passed=not failures,
        failures=tuple(sorted(set(failures))),
    )


def decide_promotion(
    candidate_cells: Iterable[EvaluationCell],
    champion_cells: Iterable[EvaluationCell],
    policy: PromotionPolicy,
    bootstrap_seed: int,
) -> PromotionDecision:
    candidate = list(candidate_cells)
    champion = list(champion_cells)
    manifest_digest, evaluator_digest, failures = _common_checks(
        candidate,
        champion,
        phase="promotion",
        allowed_splits={Split.PROMOTION},
        expected_manifest_digest=policy.expected_manifest_digest,
        expected_evaluator_digest=policy.expected_evaluator_digest,
        expected_contamination_digest=policy.expected_contamination_digest,
    )
    attempts = {cell.promotion_attempt for cell in candidate + champion}
    if len(attempts) != 1 or next(iter(attempts)) > policy.maximum_promotion_attempts:
        failures.append("promotion_attempt_budget_exceeded")
    candidate_strata: dict[str, set[str]] = {}
    champion_strata: dict[str, set[str]] = {}
    for cell in candidate:
        candidate_strata.setdefault(cell.task_id, set()).add(cell.stratum)
    for cell in champion:
        champion_strata.setdefault(cell.task_id, set()).add(cell.stratum)
    if any(len(values) != 1 for values in candidate_strata.values()) or any(
        len(values) != 1 for values in champion_strata.values()
    ):
        failures.append("task_stratum_is_not_unique")
    if candidate_strata != champion_strata:
        failures.append("task_stratum_grid_mismatch")
    task_strata = {task_id: next(iter(values)) for task_id, values in candidate_strata.items() if len(values) == 1}
    id_count = sum(value == "id" for value in task_strata.values())
    ood_count = sum(value == "ood" for value in task_strata.values())
    if id_count != policy.promotion_id_tasks:
        failures.append("promotion_id_task_count_mismatch")
    if ood_count != policy.promotion_ood_tasks:
        failures.append("promotion_ood_task_count_mismatch")
    decision = _make_split_decision(
        candidate,
        champion,
        Split.PROMOTION,
        minimum_tasks=policy.minimum_tasks,
        alpha=policy.alpha,
        bootstrap_replicates=policy.bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        minimum_vds_delta=policy.minimum_vds_delta,
        minimum_hard_pass_rate=policy.minimum_hard_pass_rate,
        minimum_executable_pass_rate=policy.minimum_executable_pass_rate,
        maximum_regressed_task_fraction=policy.maximum_regressed_task_fraction,
        expected_rollout_seeds=policy.rollout_seeds,
        expected_adapter_seeds=policy.adapter_seeds,
        maximum_component_drop=policy.maximum_component_drop,
        maximum_cost_ratio=policy.maximum_cost_ratio,
        adapter_gate=True,
        minimum_adapter_seed_passes=policy.minimum_adapter_seed_passes,
    )
    candidate_ood = {cell.key: cell for cell in candidate if cell.stratum == "ood"}
    champion_ood = {cell.key: cell for cell in champion if cell.stratum == "ood"}
    ood_deltas = [
        candidate_ood[key].vds_score - champion_ood[key].vds_score
        for key in sorted(set(candidate_ood) & set(champion_ood))
    ]
    if ood_deltas and fmean(ood_deltas) < 0.0:
        failures.append("promotion_ood_delta_negative")
    failures.extend(f"promotion:{item}" for item in decision.failures)
    return PromotionDecision(
        promoted=not failures,
        decision_kind="promotion",
        candidate_checkpoint_digest=_single_value("candidate", candidate, "checkpoint_digest"),
        champion_checkpoint_digest=_single_value("champion", champion, "checkpoint_digest"),
        manifest_digest=manifest_digest,
        evaluator_digest=evaluator_digest,
        split_decisions=(decision,),
        failures=tuple(sorted(set(failures))),
    ).sealed()


def decide_final(
    candidate_cells: Iterable[EvaluationCell],
    champion_cells: Iterable[EvaluationCell],
    policy: FinalPolicy,
    bootstrap_seed: int,
) -> PromotionDecision:
    candidate = list(candidate_cells)
    champion = list(champion_cells)
    manifest_digest, evaluator_digest, failures = _common_checks(
        candidate,
        champion,
        phase="final",
        allowed_splits={Split.HELDOUT, Split.OOD},
        expected_manifest_digest=policy.expected_manifest_digest,
        expected_evaluator_digest=policy.expected_evaluator_digest,
        expected_contamination_digest=policy.expected_contamination_digest,
        expected_reference_checkpoint_id=policy.reference_checkpoint_id,
        expected_reference_checkpoint_digest=policy.reference_checkpoint_digest,
    )
    id_decision = _make_split_decision(
        candidate,
        champion,
        Split.HELDOUT,
        minimum_tasks=policy.minimum_id_tasks,
        alpha=policy.alpha,
        bootstrap_replicates=policy.bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        minimum_vds_delta=0.0,
        minimum_hard_pass_rate=policy.minimum_hard_pass_rate,
        minimum_executable_pass_rate=policy.minimum_executable_pass_rate,
        maximum_regressed_task_fraction=0.0,
        expected_rollout_seeds=policy.rollout_seeds,
        expected_adapter_seeds=policy.adapter_seeds,
        maximum_component_drop=None,
        maximum_cost_ratio=None,
        adapter_gate=False,
    )
    ood_decision = _make_split_decision(
        candidate,
        champion,
        Split.OOD,
        minimum_tasks=policy.minimum_ood_tasks,
        alpha=policy.alpha,
        bootstrap_replicates=policy.bootstrap_replicates,
        bootstrap_seed=bootstrap_seed + 1,
        minimum_vds_delta=0.0,
        minimum_hard_pass_rate=policy.minimum_hard_pass_rate,
        minimum_executable_pass_rate=policy.minimum_executable_pass_rate,
        maximum_regressed_task_fraction=0.0,
        expected_rollout_seeds=policy.rollout_seeds,
        expected_adapter_seeds=policy.adapter_seeds,
        maximum_component_drop=None,
        maximum_cost_ratio=None,
        adapter_gate=False,
        noninferiority_margin=policy.ood_noninferiority_margin,
    )
    if id_decision.mean_vds_delta < policy.minimum_id_vds_delta:
        failures.append("final_id_mean_gain_gate_failed")
    if not math.isfinite(id_decision.vds_delta_ci_low) or id_decision.vds_delta_ci_low <= 0.0:
        failures.append("final_id_positive_gain_gate_failed")
    failures.extend(
        f"{item.split.value}:{failure}"
        for item in (id_decision, ood_decision)
        for failure in item.failures
    )
    return PromotionDecision(
        promoted=not failures,
        decision_kind="final",
        candidate_checkpoint_digest=_single_value("candidate", candidate, "checkpoint_digest"),
        champion_checkpoint_digest=_single_value("champion", champion, "checkpoint_digest"),
        manifest_digest=manifest_digest,
        evaluator_digest=evaluator_digest,
        split_decisions=(id_decision, ood_decision),
        failures=tuple(sorted(set(failures))),
    ).sealed()
