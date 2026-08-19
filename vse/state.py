from __future__ import annotations

from dataclasses import asdict, dataclass

from .ledger import RunLedger
from .promotion import PromotionDecision


@dataclass(frozen=True)
class ExperimentState:
    promotion_attempts: int
    champion_group_digest: str
    final_evaluation_recorded: bool
    stopped: bool


class PromotionStateMachine:
    def __init__(
        self,
        ledger: RunLedger,
        *,
        maximum_attempts: int,
        frozen_bindings: dict[str, str],
        initial_champion_group_digest: str,
    ):
        self.ledger = ledger
        self.maximum_attempts = maximum_attempts
        self.frozen_bindings = dict(frozen_bindings)
        self.initial_champion_group_digest = initial_champion_group_digest

    def state(self) -> ExperimentState:
        attempts = 0
        champion = self.initial_champion_group_digest
        final_recorded = False
        stopped = False
        for entry in self.ledger.validate():
            if entry.event_type in {"promotion_decision", "final_decision"}:
                for key, value in self.frozen_bindings.items():
                    if entry.bindings.get(key) != value:
                        raise ValueError(f"ledger binding drift: {key}")
            if entry.event_type == "promotion_decision":
                expected_attempt = attempts + 1
                attempt = int(entry.payload["promotion_attempt"])
                if attempt != expected_attempt:
                    raise ValueError("promotion attempts are not contiguous")
                attempts = attempt
                if bool(entry.payload["promoted"]):
                    champion = str(entry.payload["candidate_group_digest"])
                if attempts >= self.maximum_attempts:
                    stopped = True
            elif entry.event_type == "final_decision":
                if final_recorded:
                    raise ValueError("final evaluation appears more than once")
                final_recorded = True
                stopped = True
        return ExperimentState(
            promotion_attempts=attempts,
            champion_group_digest=champion,
            final_evaluation_recorded=final_recorded,
            stopped=stopped,
        )

    def record_promotion(
        self, decision: PromotionDecision, *, promotion_attempt: int
    ) -> ExperimentState:
        if decision.decision_kind != "promotion":
            raise ValueError("promotion state requires a promotion decision")
        current = self.state()
        if current.final_evaluation_recorded or current.stopped:
            raise ValueError("experiment no longer accepts promotion attempts")
        if promotion_attempt != current.promotion_attempts + 1:
            raise ValueError("promotion attempt must be the next contiguous attempt")
        if promotion_attempt > self.maximum_attempts:
            raise ValueError("promotion attempt budget exceeded")
        if decision.champion_group_digest != current.champion_group_digest:
            raise ValueError("promotion comparator is not the current champion")
        self.ledger.append(
            "promotion_decision",
            {
                "promotion_attempt": promotion_attempt,
                "promoted": decision.promoted,
                "candidate_group_digest": decision.candidate_group_digest,
                "champion_group_digest": decision.champion_group_digest,
                "decision_digest": decision.decision_digest,
            },
            bindings=self.frozen_bindings,
        )
        if self.frozen_bindings.get("freeze_bindings_digest"):
            self.ledger.anchor_head(
                event_type="promotion_decision",
                freeze_bindings_digest=self.frozen_bindings["freeze_bindings_digest"],
            )
        return self.state()

    def record_final(self, decision: PromotionDecision) -> ExperimentState:
        if decision.decision_kind != "final":
            raise ValueError("final state requires a final decision")
        current = self.state()
        if current.final_evaluation_recorded:
            raise ValueError("final evaluation may run only once")
        if decision.candidate_group_digest != current.champion_group_digest:
            raise ValueError("final candidate is not the selected champion")
        self.ledger.append(
            "final_decision",
            {
                "promoted": decision.promoted,
                "candidate_group_digest": decision.candidate_group_digest,
                "reference_group_digest": decision.champion_group_digest,
                "decision_digest": decision.decision_digest,
                "state_before": asdict(current),
            },
            bindings=self.frozen_bindings,
        )
        if self.frozen_bindings.get("freeze_bindings_digest"):
            self.ledger.anchor_head(
                event_type="final_decision",
                freeze_bindings_digest=self.frozen_bindings["freeze_bindings_digest"],
            )
        return self.state()
