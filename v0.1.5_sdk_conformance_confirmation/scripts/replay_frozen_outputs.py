#!/usr/bin/env python3
"""Replay raw model outputs through the frozen protocol compiler, no model call."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from generate_sdk_proposals import (
    FROZEN_PROTOCOL_FIELDS,
    PROPOSAL_KEYS,
    extract_object,
    frozen_view,
    hydrate_frozen_protocol_fields,
    normalize_code_encoding,
    proposal_binding,
    proposal_with_code,
    public_runtime_check,
    task_for,
    canonical,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--model-digest", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.case_manifest.read_text())
    records = []
    for case in manifest["cases"]:
        source = args.raw_root / case["case_id"] / "raw_initial_output.txt"
        root = args.output_root / case["case_id"]
        root.mkdir(parents=True, exist_ok=True)
        raw = source.read_text()
        (root / "raw_initial_output.txt").write_text(raw)
        proposal = None
        error = ""
        raw_complete = False
        hydrated_fields: list[str] = []
        normalized = False
        try:
            raw_value = extract_object(raw, ("hypothesis", "solution", "experiment_code"))
            raw_complete = set(raw_value) == PROPOSAL_KEYS
            value, hydrated_fields = hydrate_frozen_protocol_fields(raw_value)
            code, normalized = normalize_code_encoding(str(value["experiment_code"]))
            proposal = proposal_with_code(value, task_for(case), args.model_digest, code, ())
            (root / "proposal.json").write_text(json.dumps(proposal.payload(), indent=2, sort_keys=True) + "\n")
            (root / "candidate.py").write_text(code)
            (root / "proposal_binding.json").write_text(json.dumps(proposal_binding(proposal), indent=2, sort_keys=True) + "\n")
            chain = {"schema_version": 1, "initial_digest": proposal.digest, "final_digest": proposal.digest,
                     "repairs": [], "initial_code_encoding_normalized": normalized,
                     "frozen_fields_digest": hashlib.sha256(canonical(frozen_view(proposal)).encode()).hexdigest(),
                     "frozen_fields": ["hypothesis", "solution", "seeds", "baselines", "primary_metric",
                                       "secondary_metrics", "expected_effect", "power_assumptions",
                                       "stopping_rule", "resource_schedule"]}
            (root / "proposal_chain.json").write_text(json.dumps(chain, indent=2, sort_keys=True) + "\n")
            public_runtime_check(code, case)
        except (ValueError, TypeError, KeyError, SyntaxError, subprocess.SubprocessError) as caught:
            error = str(caught)[:1000]
        executable = proposal is not None and not error
        records.append({"case_id": case["case_id"], "parser_valid": proposal is not None,
                        "raw_schema_complete": raw_complete,
                        "hydrated_frozen_protocol_fields": hydrated_fields,
                        "initial_code_encoding_normalized": normalized,
                        "initial_execution_valid": executable, "public_runtime_valid": executable,
                        "repairs_used": 0, "proposal_digest": proposal.digest if proposal else "",
                        "initial_error": error, "final_error": error})
    report = {"schema_version": 1, "gate": "v0.1.5_sdk_conformance_confirmation",
              "replay_of_frozen_raw_outputs": True, "model_calls_during_replay": 0,
              "maximum_code_repairs": 2, "cases": records,
              "raw_schema_complete_rate": sum(int(item["raw_schema_complete"]) for item in records) / len(records),
              "parser_valid_rate": sum(int(item["parser_valid"]) for item in records) / len(records),
              "initial_execution_rate": sum(int(item["initial_execution_valid"]) for item in records) / len(records),
              "execution_rate_after_repair": sum(int(item["public_runtime_valid"]) for item in records) / len(records),
              "frozen_protocol_field_hydration": sorted(FROZEN_PROTOCOL_FIELDS),
              "hypothesis_frozen_during_repair": True, "scientific_hard_pass_launch_gate": False}
    (args.output_root / "generation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["parser_valid_rate"] == 1.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
