#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any


FORBIDDEN_TEXT = (
    "beyond value functions",
    "nonconvex simple bilevel optimization",
    "linearly constrained bilevel optimization",
    "neurips",
    "openreview",
    "arxiv",
    "realpilot_",
    "sealed target",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_row(index: int, rng: random.Random) -> dict[str, Any]:
    left = rng.randint(11, 97)
    right = rng.randint(3, 29)
    modulus = rng.randint(5, 17)
    offset = rng.randint(1, modulus - 1)
    combined = left * right + offset
    remainder = combined % modulus
    quotient = combined // modulus
    task_id = f"synthetic_modular_trace_{index:03d}"
    segments = [
        {
            "segment_type": "system",
            "role": "system",
            "content": (
                "Operate only on deterministic synthetic arithmetic fixtures. "
                "Return explicit intermediate state and do not cite external sources."
            ),
        },
        {
            "segment_type": "user",
            "role": "user",
            "content": (
                f"For toy task {task_id}, compute ({left} * {right} + {offset}) "
                f"divided by {modulus}. Report quotient and remainder."
            ),
        },
        {
            "segment_type": "assistant_plan",
            "role": "assistant",
            "content": canonical_json({
                "segment_type": "assistant_plan",
                "plan": ["multiply", "add_offset", "integer_divide", "verify_identity"],
                "assumption": "all values are exact integers",
            }),
        },
        {
            "segment_type": "assistant_action",
            "role": "assistant",
            "content": canonical_json({
                "segment_type": "assistant_action",
                "tool": "synthetic_integer_calculator",
                "arguments": {"left": left, "right": right, "offset": offset, "modulus": modulus},
            }),
        },
        {
            "segment_type": "tool_observation",
            "role": "tool",
            "content": canonical_json({
                "product_plus_offset": combined,
                "quotient": quotient,
                "remainder": remainder,
            }),
        },
        {
            "segment_type": "structured_result_explanation",
            "role": "assistant",
            "content": canonical_json({
                "segment_type": "structured_result_explanation",
                "result": {"quotient": quotient, "remainder": remainder},
                "verification": f"{modulus} * {quotient} + {remainder} = {combined}",
            }),
        },
        {
            "segment_type": "belief_update",
            "role": "assistant",
            "content": canonical_json({
                "segment_type": "belief_update",
                "status": "verified",
                "confidence": 1.0,
                "remaining_uncertainty": "none for this exact toy calculation",
            }),
        },
    ]
    return {
        "schema_version": 1,
        "task_id": task_id,
        "source_kind": "deterministic_synthetic_toy",
        "formal_split_eligible": False,
        "segments": segments,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("study_kind") != "qlora_pipeline_smoke":
        raise SystemExit("synthetic generator only supports qlora_pipeline_smoke")
    if any(config.get(key) is not False for key in (
        "scientific_claims_allowed", "eligible_for_champion", "eligible_for_training_library"
    )):
        raise SystemExit("smoke eligibility flags must all be false")
    count = int(config["dataset"]["row_count"])
    rng = random.Random(20260819)
    rows = [make_row(index, rng) for index in range(count)]
    serialized = "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows)
    lowered = serialized.lower()
    leaked = [pattern for pattern in FORBIDDEN_TEXT if pattern in lowered]
    if leaked:
        raise SystemExit("forbidden real-paper text in synthetic data: " + ", ".join(leaked))
    args.output_root.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output_root / "synthetic_train.jsonl"
    if dataset_path.exists() and dataset_path.read_text(encoding="utf-8") != serialized:
        raise FileExistsError(f"refusing to replace different dataset: {dataset_path}")
    dataset_path.write_text(serialized, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "dataset_kind": "deterministic_synthetic_toy",
        "row_count": count,
        "generation_seed": 20260819,
        "dataset_file": dataset_path.name,
        "dataset_sha256": file_hash(dataset_path),
        "formal_split_eligible": False,
        "eligible_for_training_library": False,
        "forbidden_text_scan": {"passed": True, "patterns": list(FORBIDDEN_TEXT)},
        "manifest_digest": "",
    }
    manifest["manifest_digest"] = hashlib.sha256(
        canonical_json(manifest).encode("utf-8")
    ).hexdigest()
    manifest_path = args.output_root / "dataset_manifest.json"
    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != rendered:
        raise FileExistsError(f"refusing to replace different manifest: {manifest_path}")
    manifest_path.write_text(rendered, encoding="utf-8")
    print(json.dumps({"dataset": str(dataset_path), "rows": count, "sha256": manifest["dataset_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
