from __future__ import annotations

from dataclasses import asdict, dataclass
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
    assignment_digest: str = ""

    def sealed(self) -> "PaperSelection":
        value = asdict(self)
        value["assignment_digest"] = ""
        return PaperSelection(
            **{**asdict(self), "assignment_digest": content_hash(value)}
        )


def eligible_candidates(
    candidates: Iterable[PaperCandidate],
    *,
    publication_start_utc: str,
    publication_end_utc: str,
    max_gpu_hours: float,
    max_cpu_hours: float,
) -> tuple[PaperCandidate, ...]:
    start = parse_utc(publication_start_utc)
    end = parse_utc(publication_end_utc)
    selected: list[PaperCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.paper_id in seen:
            raise ValueError(f"duplicate paper candidate: {candidate.paper_id}")
        seen.add(candidate.paper_id)
        timestamp = parse_utc(candidate.earliest_public_utc)
        if not start <= timestamp <= end:
            continue
        if candidate.venue not in ALLOWED_VENUES or candidate.proceedings_year not in {2025, 2026}:
            continue
        if candidate.stratum not in ID_STRATA | OOD_STRATA:
            continue
        if not all(
            (
                candidate.official_main_or_proceedings,
                candidate.public_paper,
                candidate.public_code,
                candidate.public_data,
            )
        ):
            continue
        if any(
            (
                candidate.requires_commercial_api,
                candidate.requires_private_data,
                candidate.requires_real_robot,
            )
        ):
            continue
        if candidate.estimated_gpu_hours > max_gpu_hours:
            continue
        if candidate.estimated_cpu_hours > max_cpu_hours:
            continue
        selected.append(candidate)
    return tuple(sorted(selected, key=lambda item: item.paper_id))


def select_frozen_papers(
    candidates: tuple[PaperCandidate, ...],
    quotas: dict[str, dict[str, int]],
    *,
    sampling_seed: int,
    minimum_candidate_pool: int = 70,
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

    assignments: dict[str, tuple[str, ...]] = {}
    used: set[str] = set()
    for split_name in (Split.PROMOTION.value, Split.HELDOUT.value, Split.OOD.value):
        assigned: list[str] = []
        for stratum, count in quotas[split_name].items():
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
    return PaperSelection(
        sampling_seed=sampling_seed,
        candidate_pool_digest=candidate_pool_digest,
        assignments=assignments,
    ).sealed()
