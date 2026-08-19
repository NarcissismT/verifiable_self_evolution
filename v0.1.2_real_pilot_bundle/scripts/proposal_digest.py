#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("proposal", type=Path)
    parser.add_argument("--expected-model-digest")
    args = parser.parse_args()
    value = json.loads(args.proposal.read_text(encoding="utf-8"))
    required = {
        "schema_version", "case_id", "model_digest", "algorithm_family",
        "hypothesis", "implementation", "implementation_sha256", "seeds", "oracle_budget",
    }
    allowed = required | {"hyperparameters"}
    missing = sorted(required - set(value))
    if missing:
        raise SystemExit("missing proposal fields: " + ", ".join(missing))
    extras = sorted(set(value) - allowed)
    if extras:
        raise SystemExit("unexpected proposal fields: " + ", ".join(extras))
    if value["schema_version"] != 1:
        raise SystemExit("unsupported proposal schema")
    if value["case_id"] not in {
        "realpilot_flatness_penalty",
        "realpilot_nonconvex_simple",
        "realpilot_linear_coupling",
    }:
        raise SystemExit("unsupported real pilot case_id")
    if len(str(value["hypothesis"]).strip()) < 20:
        raise SystemExit("hypothesis is too short")
    model_digest = str(value["model_digest"])
    if len(model_digest) != 64 or any(char not in "0123456789abcdef" for char in model_digest):
        raise SystemExit("model_digest must be lowercase SHA-256")
    if args.expected_model_digest and model_digest != args.expected_model_digest:
        raise SystemExit("proposal/model digest mismatch")
    if value["seeds"] != [1031, 2063, 4099, 8191] or value["oracle_budget"] != 4000:
        raise SystemExit("frozen seed grid or oracle budget mismatch")
    implementation = (args.proposal.parent / value["implementation"]).resolve()
    implementation.relative_to(args.proposal.parent.resolve())
    if not implementation.is_file():
        raise SystemExit(f"implementation missing: {implementation}")
    implementation_digest = hashlib.sha256(implementation.read_bytes()).hexdigest()
    if value["implementation_sha256"] != implementation_digest:
        raise SystemExit("proposal/implementation digest mismatch")
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    print(digest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
