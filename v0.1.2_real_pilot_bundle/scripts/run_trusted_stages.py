#!/usr/bin/env python3
"""Run execution/evaluation in the immutable network-disabled evaluator image."""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--image-digest-file", type=Path, required=True)
    parser.add_argument("--trust-key", type=Path, required=True)
    args = parser.parse_args()
    try:
        from vse.trusted_producer import run_trusted_process
    except ImportError as error:
        raise SystemExit("run from the verifiable_self_evolution repository root") from error

    run_root = args.run_root.resolve()
    image_digest = args.image_digest_file.read_text(encoding="utf-8").strip()
    if not image_digest.startswith("sha256:") or len(image_digest) != 71:
        raise SystemExit("image digest file must contain a Docker sha256: image ID")
    trust_key = args.trust_key.read_bytes()
    if len(trust_key) < 32:
        raise SystemExit("trust key must contain at least 32 bytes")

    for case_id in CASES:
        public_case = (run_root / "public" / case_id).resolve()
        proposal_dir = (run_root / "proposals" / case_id).resolve()
        proposal_path = proposal_dir / "proposal.json"
        proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
        if proposal.get("case_id") != case_id:
            raise SystemExit(f"proposal case mismatch: {case_id}")
        proposal_digest = content_hash(proposal)
        capsule = json.loads((public_case / "capsule.json").read_text(encoding="utf-8"))
        capsule_digest = content_hash(capsule)
        generation_producer = public_case / "generation_producer_receipt.json"
        if not generation_producer.is_file():
            raise SystemExit(
                f"missing real generation producer receipt for {case_id}; "
                "run a deterministic two-pass model generation first"
            )
        generation_value = json.loads(generation_producer.read_text(encoding="utf-8"))
        if generation_value.get("proposal_digest") != proposal_digest:
            raise SystemExit(f"generation receipt/proposal mismatch: {case_id}")

        common = [
            "docker", "run", "--rm", "--network", "none", "--read-only",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "128", "--memory", "2g", "--cpus", "2",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=128m",
            "-e", "VSE_NETWORK_POLICY=none",
            "-v", f"{proposal_dir}:/proposal:ro",
            "-v", f"{public_case}:/output:rw",
            image_digest,
        ]
        execution_output = public_case / "execution_output.json"
        execution_receipt = public_case / "execution_producer_receipt.json"
        run_trusted_process(
            stage="execution",
            capsule_digest=capsule_digest,
            proposal_digest=proposal_digest,
            command=[
                *common, "execute", "--case-id", case_id,
                "--proposal", "/proposal/proposal.json",
                "--output", "/output/execution_output.json",
            ],
            container_digest=image_digest,
            trust_key=trust_key,
            output_path=execution_receipt,
            artifact_path=execution_output,
            runtime_mode="docker",
        )
        if not execution_output.is_file():
            raise SystemExit(f"execution output missing after trusted run: {case_id}")

        run_trusted_process(
            stage="evaluation",
            capsule_digest=capsule_digest,
            proposal_digest="",
            command=[
                *common, "evaluate",
                "--execution", "/output/execution_output.json",
                "--output", "/output/evaluation_output.json",
            ],
            container_digest=image_digest,
            trust_key=trust_key,
            output_path=public_case / "evaluation_producer_receipt.json",
            artifact_path=public_case / "evaluation_output.json",
            runtime_mode="docker",
        )
        print(json.dumps({"case_id": case_id, "proposal_digest": proposal_digest, "status": "executed_and_evaluated"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
