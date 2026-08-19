#!/usr/bin/env python3
"""Create a repository-compatible receipt after a human independent review."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


CATEGORIES = [
    "title_and_identifier",
    "algorithm_name_and_acronym",
    "semantic_paraphrase",
    "citation_graph",
    "code_fingerprint",
    "numeric_result_fingerprint",
    "task_wording_and_evaluator",
]


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
    serialized = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != serialized:
        raise FileExistsError(f"refusing to replace review artifact: {path}")
    path.write_text(serialized, encoding="utf-8")


def normalized(value: str) -> str:
    return "".join(char.lower() for char in value if char.isalnum())


def deterministic_findings(capsule: dict[str, Any], target: dict[str, Any], public_root: Path) -> list[str]:
    parts = [capsule["public_problem"], canonical_json(capsule["research_context"])]
    for artifact in capsule["artifacts"]:
        parts.extend([artifact["title"], artifact["provenance_url"]])
        text_path = (public_root / artifact["search_text_path"]).resolve()
        text_path.relative_to(public_root.resolve())
        parts.append(text_path.read_text(encoding="utf-8", errors="ignore"))
    raw = "\n".join(parts)
    compact = normalized(raw)
    findings: list[str] = []
    for value in [target["title"], target["target_id"], *target.get("identifiers", [])]:
        token = normalized(str(value))
        if len(token) >= 6 and token in compact:
            findings.append(f"deterministic_identifier_match:{value}")
    for value in [*target.get("algorithm_names", []), *target.get("forbidden_terms", [])]:
        token = normalized(str(value))
        if token and token in compact:
            findings.append(f"deterministic_forbidden_term_match:{value}")
    for value in target.get("distinctive_phrases", []):
        token = normalized(str(value))
        if token and token in compact:
            findings.append(f"deterministic_distinctive_phrase_match:{content_hash(value)[:12]}")
    for value in target.get("code_fingerprints", []):
        token = normalized(str(value))
        if token and token in compact:
            findings.append(f"deterministic_code_match:{content_hash(value)[:12]}")
    no_space = "".join(raw.split())
    for value in target.get("numeric_fingerprints", []):
        if str(value) and "".join(str(value).split()) in no_space:
            findings.append(f"deterministic_numeric_match:{content_hash(str(value))[:12]}")
    return sorted(set(findings))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--evaluator-version", required=True)
    parser.add_argument("--decision", choices=("pass", "fail"), required=True)
    parser.add_argument("--findings-file", type=Path, required=True)
    parser.add_argument("--attest-independent-of-target", action="store_true")
    parser.add_argument("--attest-independent-of-capsule", action="store_true")
    parser.add_argument("--attest-independent-of-evaluator", action="store_true")
    args = parser.parse_args()

    attestations = [
        args.attest_independent_of_target,
        args.attest_independent_of_capsule,
        args.attest_independent_of_evaluator,
    ]
    if not all(attestations):
        raise SystemExit("all three explicit independence attestations are required")
    if not re.fullmatch(r"[A-Za-z0-9_.:@/-]{4,160}", args.reviewer_id):
        raise SystemExit("reviewer-id must be a stable 4-160 character provenance identifier")
    if not args.evaluator_version.strip():
        raise SystemExit("evaluator-version is required")
    run_root = args.run_root.resolve()
    public_root = run_root / "public"
    capsule_path = public_root / args.case_id / "capsule.json"
    target_path = run_root / "sealed" / args.case_id / "target.json"
    capsule = json.loads(capsule_path.read_text(encoding="utf-8"))
    target = json.loads(target_path.read_text(encoding="utf-8"))
    if capsule["capsule_id"] != args.case_id or target["capsule_id"] != args.case_id:
        raise SystemExit("case/capsule/target identity mismatch")
    if capsule.get("semantic_leak_review_digest"):
        raise SystemExit("capsule already has a semantic review; refusing replacement")

    deterministic = deterministic_findings(capsule, target, public_root)
    human_findings = [
        line.strip() for line in args.findings_file.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if args.decision == "pass" and deterministic:
        raise SystemExit("cannot pass: deterministic leakage findings: " + "; ".join(deterministic))
    if args.decision == "fail" and not human_findings and not deterministic:
        raise SystemExit("a failed review must record at least one finding")

    pre_review = dict(capsule)
    pre_review["semantic_leak_review_digest"] = ""
    receipt = {
        "capsule_digest": content_hash(pre_review),
        "target_commitment": content_hash(target),
        "reviewer_id": args.reviewer_id,
        "evaluator_version": args.evaluator_version,
        "independent": True,
        "passed": args.decision == "pass",
        "categories": CATEGORIES,
        "findings": [*deterministic, *human_findings],
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = content_hash(receipt)
    receipt_path = public_root / args.case_id / "semantic_review_receipt.json"
    write_json(receipt_path, receipt)
    capsule["semantic_leak_review_digest"] = file_hash(receipt_path)
    serialized = json.dumps(capsule, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    capsule_path.write_text(serialized, encoding="utf-8")
    print(json.dumps({
        "case_id": args.case_id,
        "passed": receipt["passed"],
        "receipt_path": str(receipt_path),
        "receipt_file_sha256": file_hash(receipt_path),
        "post_review_capsule_digest": content_hash(capsule),
    }, indent=2))
    return 0 if receipt["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
