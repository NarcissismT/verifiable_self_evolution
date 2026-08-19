from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import timedelta
import random
from typing import Any, Iterable

from .contracts import Split
from .hashing import content_hash
from .paper_capsule import parse_utc


ALLOWED_VENUES = frozenset(
    {"ICLR", "ICML", "NeurIPS", "AISTATS", "UAI", "L4DC"}
)
ID_STRATA = frozenset(
    {
        "constrained_safe_rl",
        "general_sum_equilibrium_marl",
        "bilevel_stackelberg_alignment_optimization",
    }
)
OOD_STRATA = frozenset(
    {
        "ood_constrained_bilevel_differentiable_optimization",
        "ood_safe_control_learning_to_optimize",
    }
)


@dataclass(frozen=True)
class PaperCandidate:
    paper_id: str
    stratum: str
    venue: str
    proceedings_year: int
    official_main_or_proceedings: bool
    public_paper: bool
    public_code: bool
    public_data: bool
    requires_commercial_api: bool
    requires_private_data: bool
    requires_real_robot: bool
    estimated_gpu_hours: float
    estimated_cpu_hours: float
    public_timestamps_utc: dict[str, str]

    @property
    def earliest_public_utc(self) -> str:
        if not self.public_timestamps_utc:
            raise ValueError(f"candidate has no public timestamp: {self.paper_id}")
        return min(
            (parse_utc(value) for value in self.public_timestamps_utc.values())
        ).isoformat().replace("+00:00", "Z")

    def capsule_cutoff_utc(self, days: int = 30) -> str:
        value = parse_utc(self.earliest_public_utc) - timedelta(days=days)
        return value.isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class PaperSelection:
    sampling_seed: int
    candidate_pool_digest: str
    assignments: dict[str, tuple[str, ...]]
    reserved_ids: tuple[str, ...] = ()
    cutoffs_utc: dict[str, str] = field(default_factory=dict)
    strata_by_id: dict[str, str] = field(default_factory=dict)
    excluded_ids: tuple[str, ...] = ()
    replacement_history: tuple[dict[str, str], ...] = ()
    assignment_digest: str = ""

    def sealed(self) -> "PaperSelection":
        value = asdict(self)
        value["assignment_digest"] = ""
        return PaperSelection(
            **{**asdict(self), "assignment_digest": content_hash(value)}
        )


def candidate_exclusion_reasons(
    candidate: PaperCandidate,
    *,
    publication_start_utc: str,
    publication_end_utc: str,
    max_gpu_hours: float,
    max_cpu_hours: float,
) -> tuple[str, ...]:
    failures: list[str] = []
    try:
        timestamp = parse_utc(candidate.earliest_public_utc)
    except ValueError:
        timestamp = None
        failures.append("missing_public_timestamp")
    start = parse_utc(publication_start_utc)
    end = parse_utc(publication_end_utc)
    if timestamp is not None and not start <= timestamp <= end:
        failures.append("outside_publication_window")
    if candidate.venue not in ALLOWED_VENUES:
        failures.append("venue_not_allowed")
    if candidate.proceedings_year not in {2025, 2026}:
        failures.append("proceedings_year_not_allowed")
    if candidate.stratum not in ID_STRATA | OOD_STRATA:
        failures.append("stratum_not_allowed")
    for valid, name in (
        (candidate.official_main_or_proceedings, "not_official_main_or_proceedings"),
        (candidate.public_paper, "paper_not_public"),
        (candidate.public_code, "code_not_public"),
        (candidate.public_data, "data_not_public"),
    ):
        if not valid:
            failures.append(name)
    for forbidden, name in (
        (candidate.requires_commercial_api, "requires_commercial_api"),
        (candidate.requires_private_data, "requires_private_data"),
        (candidate.requires_real_robot, "requires_real_robot"),
    ):
        if forbidden:
            failures.append(name)
    if candidate.estimated_gpu_hours > max_gpu_hours:
        failures.append("gpu_budget_exceeded")
    if candidate.estimated_cpu_hours > max_cpu_hours:
        failures.append("cpu_budget_exceeded")
    return tuple(sorted(failures))


def eligible_candidates(
    candidates: Iterable[PaperCandidate],
    *,
    publication_start_utc: str,
    publication_end_utc: str,
    max_gpu_hours: float,
    max_cpu_hours: float,
) -> tuple[PaperCandidate, ...]:
    selected: list[PaperCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.paper_id in seen:
            raise ValueError(f"duplicate paper candidate: {candidate.paper_id}")
        seen.add(candidate.paper_id)
        if candidate_exclusion_reasons(
            candidate,
            publication_start_utc=publication_start_utc,
            publication_end_utc=publication_end_utc,
            max_gpu_hours=max_gpu_hours,
            max_cpu_hours=max_cpu_hours,
        ):
            continue
        selected.append(candidate)
    return tuple(sorted(selected, key=lambda item: item.paper_id))


def select_frozen_papers(
    candidates: tuple[PaperCandidate, ...],
    quotas: dict[str, dict[str, int]],
    *,
    sampling_seed: int,
    minimum_candidate_pool: int = 161,
    cutoff_days: int = 30,
    reserve_minimum_by_stratum: dict[str, int] | None = None,
) -> PaperSelection:
    if len(candidates) < minimum_candidate_pool:
        raise ValueError(
            f"eligible candidate pool is too small: {len(candidates)} < {minimum_candidate_pool}"
        )
    candidate_pool_digest = content_hash([asdict(item) for item in candidates])
    available: dict[str, list[PaperCandidate]] = {}
    for candidate in candidates:
        available.setdefault(candidate.stratum, []).append(candidate)
    rng = random.Random(sampling_seed)
    for stratum in available:
        available[stratum].sort(key=lambda item: item.paper_id)
        rng.shuffle(available[stratum])

    expected_splits = (
        Split.TRAIN.value,
        Split.DEV.value,
        Split.PROMOTION.value,
        Split.HELDOUT.value,
        Split.OOD.value,
    )
    missing_quotas = set(expected_splits) - set(quotas)
    if missing_quotas:
        raise ValueError(f"selection quotas are missing splits: {sorted(missing_quotas)}")
    assignments: dict[str, tuple[str, ...]] = {}
    used: set[str] = set()
    for split_name in expected_splits:
        assigned: list[str] = []
        for stratum, count in quotas[split_name].items():
            if int(count) < 0:
                raise ValueError(f"negative quota: {split_name}/{stratum}")
            pool = [item for item in available.get(stratum, []) if item.paper_id not in used]
            if len(pool) < int(count):
                raise ValueError(
                    f"not enough candidates for {split_name}/{stratum}: "
                    f"needed={count}, available={len(pool)}"
                )
            chosen = pool[: int(count)]
            assigned.extend(item.paper_id for item in chosen)
            used.update(item.paper_id for item in chosen)
        assignments[split_name] = tuple(sorted(assigned))
    reserve_ids = tuple(
        sorted(item.paper_id for item in candidates if item.paper_id not in used)
    )
    selected_by_id = {item.paper_id: item for item in candidates}
    reserve_minimum_by_stratum = {
        str(key): int(value)
        for key, value in (reserve_minimum_by_stratum or {}).items()
    }
    reserve_by_stratum: dict[str, int] = {}
    for paper_id in reserve_ids:
        reserve_by_stratum[selected_by_id[paper_id].stratum] = (
            reserve_by_stratum.get(selected_by_id[paper_id].stratum, 0) + 1
        )
    for stratum, minimum in reserve_minimum_by_stratum.items():
        if reserve_by_stratum.get(stratum, 0) < minimum:
            raise ValueError(
                f"reserve minimum is not met for {stratum}: "
                f"{reserve_by_stratum.get(stratum, 0)} < {minimum}"
            )
    all_frozen_ids = sorted(used | set(reserve_ids))
    return PaperSelection(
        sampling_seed=sampling_seed,
        candidate_pool_digest=candidate_pool_digest,
        assignments=assignments,
        reserved_ids=reserve_ids,
        cutoffs_utc={
            paper_id: selected_by_id[paper_id].capsule_cutoff_utc(cutoff_days)
            for paper_id in all_frozen_ids
        },
        strata_by_id={paper_id: selected_by_id[paper_id].stratum for paper_id in all_frozen_ids},
    ).sealed()


def replace_from_reserve(
    selection: PaperSelection,
    *,
    split: str,
    failed_paper_id: str,
    replacement_paper_id: str,
) -> PaperSelection:
    """Replace a failed paper without changing its frozen stratum or split quota."""
    if failed_paper_id not in selection.assignments.get(split, ()):
        raise ValueError(f"failed paper is not assigned to split={split}: {failed_paper_id}")
    if replacement_paper_id not in selection.reserved_ids:
        raise ValueError(f"replacement paper is not in sealed reserve: {replacement_paper_id}")
    failed_stratum = selection.strata_by_id.get(failed_paper_id)
    replacement_stratum = selection.strata_by_id.get(replacement_paper_id)
    if not failed_stratum or failed_stratum != replacement_stratum:
        raise ValueError("reserve replacement must preserve the paper stratum")
    assignments = dict(selection.assignments)
    assignments[split] = tuple(
        sorted(
            replacement_paper_id if paper_id == failed_paper_id else paper_id
            for paper_id in assignments[split]
        )
    )
    reserved = tuple(
        paper_id
        for paper_id in selection.reserved_ids
        if paper_id != replacement_paper_id
    )
    cutoffs = dict(selection.cutoffs_utc)
    strata = dict(selection.strata_by_id)
    strata[replacement_paper_id] = failed_stratum
    return PaperSelection(
        sampling_seed=selection.sampling_seed,
        candidate_pool_digest=selection.candidate_pool_digest,
        assignments=assignments,
        reserved_ids=reserved,
        cutoffs_utc=cutoffs,
        strata_by_id=strata,
        excluded_ids=tuple(sorted(set(selection.excluded_ids) | {failed_paper_id})),
        replacement_history=selection.replacement_history
        + (
            {
                "split": split,
                "failed_paper_id": failed_paper_id,
                "replacement_paper_id": replacement_paper_id,
                "stratum": failed_stratum,
            },
        ),
    ).sealed()
