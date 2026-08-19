from __future__ import annotations

from dataclasses import dataclass

from .contracts import CandidateSource, Split, TeacherPurpose
from .teacher_policy import TeacherPolicy, TeacherRequest


@dataclass(frozen=True)
class PoolMember:
    source: CandidateSource
    model_id: str
    variant_id: str = ""


def validate_candidate_pool(
    members: tuple[PoolMember, ...],
    task_id: str,
    split: Split,
    round_index: int,
    observed_machine_failures: int,
    teacher_policy: TeacherPolicy,
) -> None:
    sources = [member.source for member in members]
    if sources.count(CandidateSource.CHAMPION) != 1:
        raise ValueError("candidate pool requires exactly one champion")
    if CandidateSource.LEGACY not in sources:
        raise ValueError("candidate pool requires at least one legacy model")
    variants = [member for member in members if member.source is CandidateSource.VARIANT]
    if len(variants) < 2:
        raise ValueError("candidate pool requires at least two variants")
    for index, member in enumerate(members):
        if member.source is not CandidateSource.CLOSED_TEACHER:
            continue
        purpose = (
            TeacherPurpose.BOOTSTRAP
            if round_index == teacher_policy.bootstrap_round
            else TeacherPurpose.HARD_STATE_REPAIR
        )
        teacher_policy.authorize(
            TeacherRequest(
                request_id=f"pool-{task_id}-{index}",
                task_id=task_id,
                split=split,
                purpose=purpose,
                round_index=round_index,
                observed_machine_failures=observed_machine_failures,
            )
        )

