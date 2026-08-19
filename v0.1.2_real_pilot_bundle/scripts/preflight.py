#!/usr/bin/env python3
"""Fail-closed preflight for pre-review and ready-to-run phases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CASES = (
    "realpilot_flatness_penalty",
    "realpilot_nonconvex_simple",
    "realpilot_linear_coupling",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("pre-review", "ready"), required=True)
    parser.add_argument("--trust-key", type=Path)
    parser.add_argument("--allow-provisional-cutoff", action="store_true")
    args = parser.parse_args()
    try:
        from vse.paper_capsule import audit_capsule, capsule_from_mapping, sealed_target_from_mapping
        from vse.semantic_review import load_semantic_review
        from vse.vertical_slice import run_vertical_slice
    except ImportError as error:
        raise SystemExit("run from the verifiable_self_evolution repository root") from error

    run_root = args.run_root.resolve()
    bundle_root = args.bundle_root.resolve()
    public_root = run_root / "public"
    sealed_root = run_root / "sealed"
    failures: list[str] = []
    passed_cases: list[str] = []
    if any((public_root / name).exists() for name in ("admin", "sealed", "targets_admin.json")):
        failures.append("secret_material_present_under_public_root")
    hydration = json.loads((run_root / "hydration_manifest.json").read_text(encoding="utf-8"))
    if hydration.get("cutoff_provisional") and not args.allow_provisional_cutoff:
        failures.append("provisional_cutoff_requires_explicit_flag")
    hydrated_ids = {item["case_id"] for item in hydration.get("cases", [])}
    if hydrated_ids != set(CASES):
        failures.append("hydration_case_set_mismatch")

    for case_id in CASES:
        try:
            capsule_path = public_root / case_id / "capsule.json"
            target_path = sealed_root / case_id / "target.json"
            capsule = capsule_from_mapping(json.loads(capsule_path.read_text(encoding="utf-8")))
            target = sealed_target_from_mapping(json.loads(target_path.read_text(encoding="utf-8")))
            audit = audit_capsule(capsule, target, public_root, sealed_root)
            if args.phase == "pre-review":
                unexpected = [item for item in audit.failures if item != "missing_semantic_leak_review"]
                if unexpected:
                    failures.extend(f"{case_id}:{item}" for item in unexpected)
                elif audit.failures != ("missing_semantic_leak_review",):
                    failures.append(f"{case_id}:pre_review_expected_exactly_one_missing_review")
                else:
                    passed_cases.append(case_id)
            else:
                if not audit.passed:
                    failures.extend(f"{case_id}:{item}" for item in audit.failures)
                review_path = public_root / case_id / "semantic_review_receipt.json"
                load_semantic_review(review_path, capsule=capsule, target=target)
                required = [
                    "generation_receipt.json", "execution_receipt.json", "evaluation_receipt.json",
                    "generation_producer_receipt.json", "execution_producer_receipt.json",
                    "evaluation_producer_receipt.json", "execution_output.json", "evaluation_output.json",
                ]
                for name in required:
                    if not (public_root / case_id / name).is_file():
                        failures.append(f"{case_id}:missing:{name}")
                if audit.passed:
                    passed_cases.append(case_id)
        except Exception as error:  # fail closed and report the exact class
            failures.append(f"{case_id}:{type(error).__name__}:{error}")

    if args.phase == "ready":
        try:
            evaluator_review = json.loads((run_root / "evaluator_review_receipt.json").read_text(encoding="utf-8"))
            unsealed = dict(evaluator_review)
            declared = str(unsealed.get("receipt_digest", ""))
            unsealed["receipt_digest"] = ""
            from vse.hashing import content_hash, file_hash
            if content_hash(unsealed) != declared:
                failures.append("evaluator_review_receipt_digest_mismatch")
            if not evaluator_review.get("passed"):
                failures.append("evaluator_review_did_not_pass")
            if evaluator_review.get("evaluator_digest") != file_hash(bundle_root / "evaluator" / "trusted_evaluator.py"):
                failures.append("evaluator_review_source_binding_mismatch")
            independence_fields = (
                "independent_of_target_authors",
                "independent_of_capsule_curator",
                "independent_of_student_proposer",
            )
            if not all(
                evaluator_review.get(field) is True
                for field in independence_fields
            ):
                failures.append("evaluator_review_independence_incomplete")
            image_digest = str(evaluator_review.get("container_digest", ""))
            if (
                len(image_digest) != 71
                or not image_digest.startswith("sha256:")
                or any(char not in "0123456789abcdef" for char in image_digest[7:])
            ):
                failures.append("evaluator_review_container_digest_invalid")
            if args.trust_key is not None and evaluator_review.get(
                "trusted_producer_key_digest"
            ) != hashlib.sha256(args.trust_key.read_bytes()).hexdigest():
                failures.append("evaluator_review_trust_anchor_mismatch")
        except Exception as error:
            failures.append(f"evaluator_review:{type(error).__name__}:{error}")
        manifest = public_root / "vertical_slice_manifest.json"
        if not manifest.is_file():
            failures.append("vertical_slice_manifest_missing")
        elif args.trust_key is None:
            failures.append("ready_phase_requires_trust_key")
        else:
            try:
                report = run_vertical_slice(
                    manifest,
                    public_root=public_root,
                    sealed_root=sealed_root,
                    trust_key=args.trust_key.read_bytes(),
                )
                if not report.passed:
                    failures.extend(report.failures)
            except Exception as error:
                failures.append(f"vertical_slice_dry_validation:{type(error).__name__}:{error}")

    result = {
        "phase": args.phase,
        "passed": not failures,
        "passed_cases": sorted(set(passed_cases)),
        "failures": sorted(set(failures)),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
