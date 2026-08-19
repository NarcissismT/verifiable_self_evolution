from __future__ import annotations

from dataclasses import dataclass

from .paper_capsule import parse_utc


@dataclass(frozen=True)
class ModelProvenance:
    model_id: str
    checkpoint_digest: str
    documented_knowledge_cutoff_utc: str | None
    contamination_audit_digest: str
    contamination_audit_passed: bool

    def permits_capsule(self, capsule_cutoff_utc: str, target_release_utc: str) -> bool:
        if not self.contamination_audit_passed:
            return False
        if self.documented_knowledge_cutoff_utc is None:
            return False
        model_cutoff = parse_utc(self.documented_knowledge_cutoff_utc)
        capsule_cutoff = parse_utc(capsule_cutoff_utc)
        target_release = parse_utc(target_release_utc)
        return model_cutoff <= capsule_cutoff < target_release

