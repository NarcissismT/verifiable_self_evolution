from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .hashing import content_hash, file_hash
from .paper_capsule import PublicPaperCapsule, SealedTarget


@dataclass(frozen=True)
class SemanticReviewReceipt:
    capsule_digest: str
    target_commitment: str
    reviewer_id: str
    evaluator_version: str
    independent: bool
    passed: bool
    categories: tuple[str, ...]
    findings: tuple[str, ...]
    receipt_digest: str = ""

    def sealed(self) -> "SemanticReviewReceipt":
        value = asdict(self)
        value["receipt_digest"] = ""
        return SemanticReviewReceipt(
            **{**asdict(self), "receipt_digest": content_hash(value)}
        )


def load_semantic_review(
    path: Path,
    *,
    capsule: PublicPaperCapsule,
    target: SealedTarget,
) -> SemanticReviewReceipt:
    if file_hash(path) != capsule.semantic_leak_review_digest:
        raise ValueError("semantic review file digest mismatch")
    raw = json.loads(path.read_text())
    receipt = SemanticReviewReceipt(
        capsule_digest=str(raw["capsule_digest"]),
        target_commitment=str(raw["target_commitment"]),
        reviewer_id=str(raw["reviewer_id"]),
        evaluator_version=str(raw["evaluator_version"]),
        independent=bool(raw["independent"]),
        passed=bool(raw["passed"]),
        categories=tuple(str(value) for value in raw["categories"]),
        findings=tuple(str(value) for value in raw["findings"]),
        receipt_digest=str(raw["receipt_digest"]),
    )
    if receipt.sealed().receipt_digest != receipt.receipt_digest:
        raise ValueError("semantic review receipt digest mismatch")
    if receipt.capsule_digest != capsule.pre_review_digest:
        raise ValueError("semantic review capsule binding mismatch")
    if receipt.target_commitment != target.commitment:
        raise ValueError("semantic review target binding mismatch")
    if not receipt.reviewer_id or not receipt.evaluator_version:
        raise ValueError("semantic review provenance is incomplete")
    if not receipt.independent:
        raise ValueError("semantic review is not independent")
    if not receipt.passed:
        raise ValueError("semantic review did not pass")
    if not receipt.categories:
        raise ValueError("semantic review categories are missing")
    return receipt


def write_semantic_review(path: Path, receipt: SemanticReviewReceipt) -> None:
    sealed = receipt.sealed()
    serialized = json.dumps(asdict(sealed), indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != serialized:
        raise FileExistsError(f"refusing to replace semantic review receipt: {path}")
    path.write_text(serialized)
