#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_manifest(model_root: Path, expected_digest: str) -> dict[str, Any]:
    manifest_path = model_root / "MODEL_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_digest") != expected_digest:
        raise ValueError("model manifest digest differs from smoke config")
    payload = {
        "files": manifest.get("files"),
        "model_id": manifest.get("model_id"),
        "revision": manifest.get("revision"),
        "schema_version": manifest.get("schema_version"),
    }
    if hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest() != expected_digest:
        raise ValueError("model manifest internal digest mismatch")
    for item in manifest["files"]:
        path = (model_root / item["path"]).resolve()
        path.relative_to(model_root.resolve())
        if path.stat().st_size != int(item["bytes"]) or file_hash(path) != item["sha256"]:
            raise ValueError(f"model file mismatch: {item['path']}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--code", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.image_digest):
        raise SystemExit("image digest must be sha256:<64 lowercase hex>")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("study_kind") != "qlora_pipeline_smoke":
        raise SystemExit("wrong study_kind")
    for key in ("scientific_claims_allowed", "eligible_for_champion", "eligible_for_training_library"):
        if config.get(key) is not False:
            raise SystemExit(f"{key} must be false")
    if config["model"]["model_id"] != "qwen2.5-7b-instruct":
        raise SystemExit("only Qwen2.5 7B is allowed")
    if config["training"]["adapter_seeds"] != [17]:
        raise SystemExit("smoke must use exactly adapter seed 17")
    if int(config["training"]["max_optimizer_updates"]) != 100:
        raise SystemExit("smoke must use exactly 100 optimizer updates")
    if config["dataset"] != {
        "kind": "deterministic_synthetic_toy",
        "row_count": 128,
        "formal_split_eligible": False,
        "paper_capsules_allowed": False,
        "sealed_targets_allowed": False,
    }:
        raise SystemExit("smoke dataset policy mismatch")
    model_manifest = verify_model_manifest(args.model_root, config["model"]["manifest_digest"])
    dataset_manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    declared = dataset_manifest.get("manifest_digest", "")
    blank = dict(dataset_manifest)
    blank["manifest_digest"] = ""
    if hashlib.sha256(canonical_json(blank).encode("utf-8")).hexdigest() != declared:
        raise ValueError("dataset manifest digest mismatch")
    dataset_path = args.dataset_manifest.parent / dataset_manifest["dataset_file"]
    if file_hash(dataset_path) != dataset_manifest["dataset_sha256"]:
        raise ValueError("synthetic dataset hash mismatch")
    if dataset_manifest.get("formal_split_eligible") is not False or dataset_manifest.get("eligible_for_training_library") is not False:
        raise ValueError("synthetic dataset eligibility must be false")
    receipt = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "study_kind": "qlora_pipeline_smoke",
        "scientific_claims_allowed": False,
        "eligible_for_champion": False,
        "eligible_for_training_library": False,
        "config_sha256": file_hash(args.config),
        "dataset_manifest_sha256": file_hash(args.dataset_manifest),
        "dataset_sha256": dataset_manifest["dataset_sha256"],
        "model_manifest_file_sha256": file_hash(args.model_root / "MODEL_MANIFEST.json"),
        "model_manifest_digest": model_manifest["manifest_digest"],
        "model_revision": model_manifest["revision"],
        "train_image_digest": args.image_digest,
        "code_sha256": {path.name: file_hash(path) for path in sorted(args.code)},
        "adapter_seed": 17,
        "optimizer_updates": 100,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = hashlib.sha256(canonical_json(receipt).encode("utf-8")).hexdigest()
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.stat().st_size > 0 and args.output.read_text(encoding="utf-8") != rendered:
        raise FileExistsError(f"refusing to replace different smoke freeze: {args.output}")
    args.output.write_text(rendered, encoding="utf-8")
    print(json.dumps({"ready": True, "receipt_digest": receipt["receipt_digest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
