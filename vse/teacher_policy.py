from __future__ import annotations

from dataclasses import dataclass

from .contracts import Split, TeacherPurpose


@dataclass(frozen=True)
class TeacherRequest:
    request_id: str
    task_id: str
    split: Split
    purpose: TeacherPurpose
    round_index: int
    observed_machine_failures: int
    includes_sealed_target: bool = False


@dataclass(frozen=True)
class TeacherPolicy:
    bootstrap_round: int = 0
    hard_state_min_failures: int = 2

    def authorize(self, request: TeacherRequest) -> None:
        if request.includes_sealed_target:
            raise PermissionError("closed teacher cannot receive sealed target data")
        if request.split in {Split.PROMOTION, Split.HELDOUT, Split.OOD}:
            raise PermissionError(
                "closed teacher cannot access promotion, held-out, or OOD tasks"
            )
        if request.purpose is TeacherPurpose.BOOTSTRAP:
            if request.split is not Split.TRAIN:
                raise PermissionError("bootstrap teacher data is training-only")
            if request.round_index != self.bootstrap_round:
                raise PermissionError("teacher bootstrap is generation-zero only")
            return
        if request.purpose is TeacherPurpose.HARD_STATE_REPAIR:
            if request.split is not Split.TRAIN:
                raise PermissionError("hard-state repair is training-only")
            if request.observed_machine_failures < self.hard_state_min_failures:
                raise PermissionError("hard state is not established by machine failures")
            return
        raise PermissionError(f"unsupported teacher purpose: {request.purpose}")
