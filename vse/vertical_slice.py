from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .hashing import content_hash
from .paper_capsule import audit_capsule, capsule_from_mapping, sealed_target_from_mapping
from .semantic_review import load_semantic_review
from .trusted_producer import verify_trusted_receipt


@dataclass(frozen=True)
class VerticalSliceCaseReceipt:
    case_id: str
    capsule_digest: str
    capsule_audit_passed: bool
    generation_receipt_digest: str
    execution_receipt_digest: str
    evaluation_receipt_digest: str
    hard_pass: bool
    vds_score: float
    failures: tuple[str, ...]


@dataclass(frozen=True)
class VerticalSliceReport:
    pilot_id: str
    passed: bool
    case_receipts: tuple[VerticalSliceCaseReceipt, ...]
    failures: tuple[str, ...]
    report_digest: str = ""

    def sealed(self) -> "VerticalSliceReport":
        value = asdict(self)
        value["report_digest"] = ""
        return VerticalSliceReport(
            **{**asdict(self), "report_digest": content_hash(value)}
        )


def _load_bound_receipt(path: Path, capsule_digest: str, kind: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("capsule_digest") != capsule_digest:
        raise ValueError(f"{kind} receipt capsule binding mismatch")
    declared_digest = value.get("receipt_digest", "")
    payload = dict(value)
    payload["receipt_digest"] = ""
    if content_hash(payload) != declared_digest:
        raise ValueError(f"{kind} receipt digest mismatch")
    return value


def _load_trusted_stage(
    *,
    public_root: Path,
    stage_path: str,
    capsule_digest: str,
    stage: str,
    expected_producer_digest: str | None,
    allow_test_runtime: bool,
    trust_key: bytes,
) -> dict[str, Any]:
    path = (public_root / stage_path).resolve()
    path.relative_to(public_root.resolve())
    value = json.loads(path.read_text())
    producer = verify_trusted_receipt(value, trust_key=trust_key)
    if producer.runtime_mode == "local_test" and not allow_test_runtime:
        raise ValueError("local test runtime is not allowed for a real vertical slice")
    if producer.stage != stage or producer.capsule_digest != capsule_digest:
        raise ValueError(f"{stage} trusted producer binding mismatch")
    if not expected_producer_digest:
        raise ValueError(f"{stage} trusted producer digest is not frozen")
    actual = content_hash(value)
    if actual != expected_producer_digest:
        raise ValueError(f"{stage} trusted producer digest mismatch")
    return value


def run_vertical_slice(
    manifest_path: Path,
    *,
    public_root: Path,
    sealed_root: Path,
    trust_key: bytes,
) -> VerticalSliceReport:
    manifest = json.loads(manifest_path.read_text())
    cases = manifest.get("cases", [])
    failures: list[str] = []
    trust_anchor_digest = hashlib.sha256(trust_key).hexdigest()
    if manifest.get("trust_anchor_digest") != trust_anchor_digest:
        failures.append("trust_anchor_digest_mismatch")
    if not 3 <= len(cases) <= 5:
        failures.append("vertical_slice_requires_3_to_5_cases")
    receipts: list[VerticalSliceCaseReceipt] = []
    seen_ids: set[str] = set()
    for case in cases:
        case_id = str(case.get("case_id", ""))
        case_failures: list[str] = []
        if not case_id or case_id in seen_ids:
            case_failures.append("missing_or_duplicate_case_id")
        seen_ids.add(case_id)
        if not case.get("independent_pilot") or not case.get("excluded_from_formal_splits"):
            case_failures.append("pilot_case_is_not_formally_excluded")
        try:
            capsule_path = (public_root / case["capsule_json"]).resolve()
            target_path = (sealed_root / case["target_json"]).resolve()
            capsule_path.relative_to(public_root.resolve())
            target_path.relative_to(sealed_root.resolve())
            capsule = capsule_from_mapping(json.loads(capsule_path.read_text()))
            target = sealed_target_from_mapping(json.loads(target_path.read_text()))
            audit = audit_capsule(capsule, target, public_root, sealed_root)
            case_failures.extend(audit.failures)
            review_path = (public_root / case["semantic_review_receipt"]).resolve()
            review_path.relative_to(public_root.resolve())
            load_semantic_review(review_path, capsule=capsule, target=target)
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            case_failures.append(f"capsule_or_semantic_review_error:{type(error).__name__}")
            audit = None
            capsule = None
        capsule_digest = capsule.digest if capsule is not None else ""
        stage_values: dict[str, dict[str, Any]] = {}
        for kind in ("generation", "execution", "evaluation"):
            try:
                receipt_path = (public_root / case[f"{kind}_receipt"]).resolve()
                receipt_path.relative_to(public_root.resolve())
                stage_values[kind] = _load_bound_receipt(
                    receipt_path, capsule_digest, kind
                )
                producer_path = case.get(f"{kind}_producer_receipt")
                if not producer_path:
                    raise ValueError("trusted producer receipt path is missing")
                producer = _load_trusted_stage(
                    public_root=public_root,
                    stage_path=str(producer_path),
                    capsule_digest=capsule_digest,
                    stage=kind,
                    expected_producer_digest=case.get(f"{kind}_producer_receipt_digest"),
                    allow_test_runtime=bool(manifest.get("trusted_test_runtime", False)),
                    trust_key=trust_key,
                )
                if stage_values[kind].get("producer_receipt_digest") != content_hash(
                    producer
                ):
                    case_failures.append(f"{kind}_producer_receipt_binding_mismatch")
                if stage_values[kind].get("producer_execution_digest") != producer.get(
                    "execution_digest"
                ):
                    case_failures.append(f"{kind}_producer_execution_binding_mismatch")
                if stage_values[kind].get("producer_stdout_digest") != producer.get(
                    "stdout_digest"
                ):
                    case_failures.append(f"{kind}_producer_output_binding_mismatch")
                if stage_values[kind].get("runtime_container_digest") != producer.get(
                    "container_digest"
                ):
                    case_failures.append(f"{kind}_producer_container_binding_mismatch")
                if kind in {"generation", "execution"} and producer.get(
                    "proposal_digest"
                ) != stage_values[kind].get("proposal_digest"):
                    case_failures.append(f"{kind}_producer_proposal_binding_mismatch")
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                case_failures.append(f"{kind}_receipt_error:{type(error).__name__}")
        generation = stage_values.get("generation", {})
        execution = stage_values.get("execution", {})
        evaluation = stage_values.get("evaluation", {})
        if not generation.get("model_digest") or not generation.get("proposal_digest"):
            case_failures.append("generation_provenance_incomplete")
        if execution:
            if execution.get("proposal_digest") != generation.get("proposal_digest"):
                case_failures.append("execution_proposal_binding_mismatch")
            if execution.get("network_policy") != "none":
                case_failures.append("execution_network_not_disabled")
            if not execution.get("container_digest"):
                case_failures.append("execution_container_digest_missing")
        if evaluation:
            if evaluation.get("execution_digest") != execution.get("execution_digest"):
                case_failures.append("evaluation_execution_binding_mismatch")
            if not evaluation.get("trusted_evaluator_digest"):
                case_failures.append("trusted_evaluator_digest_missing")
        try:
            vds_score = float(evaluation.get("vds_score", 0.0))
        except (TypeError, ValueError):
            vds_score = float("nan")
        if not math.isfinite(vds_score) or not 0.0 <= vds_score <= 1.0:
            case_failures.append("invalid_vds_score")
        hard_pass = bool(evaluation.get("hard_pass", False))
        if not hard_pass and vds_score != 0.0:
            case_failures.append("hard_failure_did_not_zero_vds")
        receipt = VerticalSliceCaseReceipt(
            case_id=case_id,
            capsule_digest=capsule_digest,
            capsule_audit_passed=bool(audit and audit.passed),
            generation_receipt_digest=str(generation.get("receipt_digest", "")),
            execution_receipt_digest=str(execution.get("receipt_digest", "")),
            evaluation_receipt_digest=str(evaluation.get("receipt_digest", "")),
            hard_pass=hard_pass,
            vds_score=vds_score,
            failures=tuple(sorted(set(case_failures))),
        )
        receipts.append(receipt)
        failures.extend(f"{case_id}:{item}" for item in receipt.failures)
    return VerticalSliceReport(
        pilot_id=str(manifest.get("pilot_id", "missing")),
        passed=not failures,
        case_receipts=tuple(receipts),
        failures=tuple(sorted(set(failures))),
    ).sealed()


def write_vertical_slice_report(path: Path, report: VerticalSliceReport) -> None:
    serialized = json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != serialized:
        raise FileExistsError(f"refusing to replace vertical slice report: {path}")
    path.write_text(serialized)
