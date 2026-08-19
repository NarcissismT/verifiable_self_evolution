#!/usr/bin/env python3
"""Independent custodian receipt for evaluator source and immutable image."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--image-digest-file", type=Path, required=True)
    parser.add_argument("--trust-anchor-digest-file", type=Path, required=True)
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--evaluator-version", required=True)
    parser.add_argument("--decision", choices=("pass", "fail"), required=True)
    parser.add_argument("--findings-file", type=Path, required=True)
    parser.add_argument("--attest-independent-of-target", action="store_true")
    parser.add_argument("--attest-independent-of-capsule", action="store_true")
    parser.add_argument("--attest-independent-of-student", action="store_true")
    args = parser.parse_args()
    if not all((args.attest_independent_of_target, args.attest_independent_of_capsule, args.attest_independent_of_student)):
        raise SystemExit("all evaluator-independence attestations are required")
    if not re.fullmatch(r"[A-Za-z0-9_.:@/-]{4,160}", args.reviewer_id):
        raise SystemExit("reviewer-id must be a stable provenance identifier")
    image_digest = args.image_digest_file.read_text(encoding="utf-8").strip()
    if (
        not image_digest.startswith("sha256:")
        or len(image_digest) != 71
        or any(char not in "0123456789abcdef" for char in image_digest[7:])
    ):
        raise SystemExit("image digest must be sha256:<64 hex>")
    trust_anchor_digest = args.trust_anchor_digest_file.read_text(
        encoding="utf-8"
    ).strip()
    if not re.fullmatch(r"[0-9a-f]{64}", trust_anchor_digest):
        raise SystemExit("trust anchor digest must be one lowercase SHA-256")
    evaluator = args.bundle_root.resolve() / "evaluator" / "trusted_evaluator.py"
    findings = [line.strip() for line in args.findings_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.decision == "fail" and not findings:
        raise SystemExit("failed review must include a finding")
    receipt = {
        "schema_version": 1,
        "evaluator_digest": file_hash(evaluator),
        "container_digest": image_digest,
        "trusted_producer_key_digest": trust_anchor_digest,
        "reviewer_id": args.reviewer_id,
        "evaluator_version": args.evaluator_version,
        "independent_of_target_authors": True,
        "independent_of_capsule_curator": True,
        "independent_of_student_proposer": True,
        "passed": args.decision == "pass",
        "reviewed_properties": [
            "target_neutrality", "hidden_instance_determinism", "hard_failure_zeroes_vds",
            "resource_budget", "finite_metric_checks", "network_disabled_runtime", "reference_self_test",
        ],
        "findings": findings,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = content_hash(receipt)
    output = args.run_root.resolve() / "evaluator_review_receipt.json"
    serialized = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != serialized:
        raise FileExistsError(f"refusing to replace evaluator receipt: {output}")
    output.write_text(serialized, encoding="utf-8")
    print(json.dumps({"path": str(output), "passed": receipt["passed"], "receipt_digest": receipt["receipt_digest"]}, indent=2))
    return 0 if receipt["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
