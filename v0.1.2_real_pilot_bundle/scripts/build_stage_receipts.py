#!/usr/bin/env python3
"""Bind model proposals, outputs and HMAC producer receipts into VSE stage receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CASES = (
    "realpilot_flatness_penalty",
    "realpilot_nonconvex_simple",
    "realpilot_linear_coupling",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    serialized = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != serialized:
        raise FileExistsError(f"refusing to replace stage artifact: {path}")
    path.write_text(serialized, encoding="utf-8")


def seal_receipt(value: dict[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["receipt_digest"] = ""
    payload["receipt_digest"] = content_hash(payload)
    return payload


def producer_bindings(producer: dict[str, Any]) -> dict[str, Any]:
    return {
        "producer_receipt_digest": content_hash(producer),
        "producer_execution_digest": producer["execution_digest"],
        "producer_stdout_digest": producer["stdout_digest"],
        "runtime_container_digest": producer["container_digest"],
        "producer_artifact_digest": producer.get("artifact_digest", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--trust-key", type=Path, required=True)
    parser.add_argument("--model-digest-file", type=Path, required=True)
    parser.add_argument("--generation-image-digest-file", type=Path, required=True)
    args = parser.parse_args()
    try:
        from vse.trusted_producer import verify_trusted_receipt
    except ImportError as error:
        raise SystemExit("run from the verifiable_self_evolution repository root") from error

    bundle_root = args.bundle_root.resolve()
    run_root = args.run_root.resolve()
    trust_key = args.trust_key.read_bytes()
    model_digest = args.model_digest_file.read_text(encoding="utf-8").strip()
    generation_image_digest = args.generation_image_digest_file.read_text(
        encoding="utf-8"
    ).strip()
    if len(model_digest) != 64 or any(char not in "0123456789abcdef" for char in model_digest):
        raise SystemExit("model digest file must contain one lowercase SHA-256")
    if (
        not generation_image_digest.startswith("sha256:")
        or len(generation_image_digest) != 71
        or any(
            char not in "0123456789abcdef"
            for char in generation_image_digest[7:]
        )
    ):
        raise SystemExit("generation image digest must be sha256:<64 hex>")
    evaluator_digest = file_hash(bundle_root / "evaluator" / "trusted_evaluator.py")
    evaluator_review = json.loads((run_root / "evaluator_review_receipt.json").read_text(encoding="utf-8"))
    evaluator_review_unsealed = dict(evaluator_review)
    declared_evaluator_review = str(evaluator_review_unsealed.get("receipt_digest", ""))
    evaluator_review_unsealed["receipt_digest"] = ""
    if content_hash(evaluator_review_unsealed) != declared_evaluator_review:
        raise SystemExit("evaluator review receipt digest mismatch")
    if not evaluator_review.get("passed") or evaluator_review.get("evaluator_digest") != evaluator_digest:
        raise SystemExit("independent evaluator review did not pass or is bound to different source")
    independence_fields = (
        "independent_of_target_authors",
        "independent_of_capsule_curator",
        "independent_of_student_proposer",
    )
    if not all(evaluator_review.get(field) is True for field in independence_fields):
        raise SystemExit("evaluator review independence attestations are incomplete")
    trust_anchor_digest = hashlib.sha256(trust_key).hexdigest()
    if evaluator_review.get("trusted_producer_key_digest") != trust_anchor_digest:
        raise SystemExit("evaluator review is not bound to the trusted producer key")
    manifest_cases: list[dict[str, Any]] = []

    for case_id in CASES:
        public_case = run_root / "public" / case_id
        capsule = json.loads((public_case / "capsule.json").read_text(encoding="utf-8"))
        capsule_digest = content_hash(capsule)
        proposal = json.loads((run_root / "proposals" / case_id / "proposal.json").read_text(encoding="utf-8"))
        proposal_path = run_root / "proposals" / case_id / "proposal.json"
        proposal_digest = content_hash(proposal)
        if proposal.get("model_digest") != model_digest:
            raise SystemExit(f"proposal model digest mismatch: {case_id}")
        implementation = (
            proposal_path.parent / str(proposal.get("implementation", ""))
        ).resolve()
        implementation.relative_to(proposal_path.parent.resolve())
        if not implementation.is_file() or proposal.get(
            "implementation_sha256"
        ) != file_hash(implementation):
            raise SystemExit(f"proposal implementation binding mismatch: {case_id}")
        producers: dict[str, dict[str, Any]] = {}
        for stage in ("generation", "execution", "evaluation"):
            path = public_case / f"{stage}_producer_receipt.json"
            raw = json.loads(path.read_text(encoding="utf-8"))
            verified = verify_trusted_receipt(raw, trust_key=trust_key)
            if verified.stage != stage or verified.capsule_digest != capsule_digest:
                raise SystemExit(f"trusted producer binding mismatch: {case_id}/{stage}")
            producers[stage] = raw
        if producers["generation"].get("proposal_digest") != proposal_digest:
            raise SystemExit(f"generation proposal binding mismatch: {case_id}")
        if producers["generation"].get("container_digest") != generation_image_digest:
            raise SystemExit(f"generation container mismatch: {case_id}")
        if producers["execution"].get("proposal_digest") != proposal_digest:
            raise SystemExit(f"execution proposal binding mismatch: {case_id}")
        if producers["execution"].get("container_digest") != evaluator_review.get("container_digest"):
            raise SystemExit(f"execution container/evaluator review mismatch: {case_id}")
        if producers["generation"].get("artifact_digest") != file_hash(proposal_path):
            raise SystemExit(f"generation artifact/proposal mismatch: {case_id}")
        if producers["evaluation"].get("container_digest") != evaluator_review.get("container_digest"):
            raise SystemExit(f"evaluation container/evaluator review mismatch: {case_id}")

        execution = json.loads((public_case / "execution_output.json").read_text(encoding="utf-8"))
        execution_unsealed = dict(execution)
        declared_execution = str(execution_unsealed.pop("execution_digest", ""))
        execution_unsealed["execution_digest"] = ""
        if content_hash(execution_unsealed) != declared_execution:
            raise SystemExit(f"execution digest mismatch: {case_id}")
        evaluation = json.loads((public_case / "evaluation_output.json").read_text(encoding="utf-8"))
        if evaluation.get("execution_digest") != declared_execution:
            raise SystemExit(f"evaluation/execution binding mismatch: {case_id}")
        declared_evaluation = str(evaluation.get("evaluation_digest", ""))
        evaluation_unsealed = dict(evaluation)
        evaluation_unsealed["evaluation_digest"] = ""
        if content_hash(evaluation_unsealed) != declared_evaluation:
            raise SystemExit(f"evaluation digest mismatch: {case_id}")
        if producers["execution"].get("artifact_digest") != file_hash(public_case / "execution_output.json"):
            raise SystemExit(f"execution artifact/output mismatch: {case_id}")
        if producers["evaluation"].get("artifact_digest") != file_hash(public_case / "evaluation_output.json"):
            raise SystemExit(f"evaluation artifact/output mismatch: {case_id}")

        generation_receipt = seal_receipt({
            "capsule_digest": capsule_digest,
            "model_digest": model_digest,
            "proposal_digest": proposal_digest,
            **producer_bindings(producers["generation"]),
            "receipt_digest": "",
        })
        execution_receipt = seal_receipt({
            "capsule_digest": capsule_digest,
            "proposal_digest": proposal_digest,
            "network_policy": "none",
            "container_digest": producers["execution"]["container_digest"],
            "execution_digest": declared_execution,
            **producer_bindings(producers["execution"]),
            "receipt_digest": "",
        })
        evaluation_receipt = seal_receipt({
            "capsule_digest": capsule_digest,
            "execution_digest": declared_execution,
            "trusted_evaluator_digest": evaluator_digest,
            "evaluation_digest": declared_evaluation,
            "hard_pass": bool(evaluation["hard_pass"]),
            "vds_score": float(evaluation["vds_score"]),
            **producer_bindings(producers["evaluation"]),
            "receipt_digest": "",
        })
        for stage, receipt in (
            ("generation", generation_receipt),
            ("execution", execution_receipt),
            ("evaluation", evaluation_receipt),
        ):
            write_json(public_case / f"{stage}_receipt.json", receipt)

        manifest_case: dict[str, Any] = {
            "case_id": case_id,
            "independent_pilot": True,
            "excluded_from_formal_splits": True,
            "capsule_json": f"{case_id}/capsule.json",
            "target_json": f"{case_id}/target.json",
            "semantic_review_receipt": f"{case_id}/semantic_review_receipt.json",
            "generation_artifact": f"proposals/{case_id}/proposal.json",
        }
        for stage in ("generation", "execution", "evaluation"):
            manifest_case[f"{stage}_receipt"] = f"{case_id}/{stage}_receipt.json"
            manifest_case[f"{stage}_producer_receipt"] = f"{case_id}/{stage}_producer_receipt.json"
            manifest_case[f"{stage}_producer_receipt_digest"] = content_hash(producers[stage])
        manifest_case["execution_artifact"] = f"{case_id}/execution_output.json"
        manifest_case["evaluation_artifact"] = f"{case_id}/evaluation_output.json"
        manifest_cases.append(manifest_case)

    manifest = {
        "schema_version": 1,
        "pilot_id": "bilevel_real_vertical_slice_v0_1_2",
        "trust_anchor_digest": trust_anchor_digest,
        "trusted_test_runtime": False,
        "stage_container_digests": {
            "generation": generation_image_digest,
            "execution": evaluator_review["container_digest"],
            "evaluation": evaluator_review["container_digest"],
        },
        "cases": manifest_cases,
    }
    write_json(run_root / "public" / "vertical_slice_manifest.json", manifest)
    print(json.dumps({"status": "stage_receipts_built", "manifest": str(run_root / "public" / "vertical_slice_manifest.json")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
