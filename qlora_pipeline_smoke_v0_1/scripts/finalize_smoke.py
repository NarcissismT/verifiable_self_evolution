#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sealed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    declared = value.get("receipt_digest", "")
    blank = dict(value)
    blank["receipt_digest"] = ""
    if hashlib.sha256(canonical_json(blank).encode("utf-8")).hexdigest() != declared:
        raise ValueError(f"receipt digest mismatch: {path.name}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    training = sealed(args.output_root / "training_receipt.json")
    reload = sealed(args.output_root / "reload_receipt.json")
    freeze = sealed(args.output_root / "smoke_freeze_manifest.json")
    if reload["training_receipt_digest"] != training["receipt_digest"]:
        raise ValueError("reload receipt is not bound to training receipt")
    if training["freeze_receipt_digest"] != freeze["receipt_digest"]:
        raise ValueError("training receipt is not bound to smoke freeze")
    for receipt in (training, reload, freeze):
        for key in ("scientific_claims_allowed", "eligible_for_champion", "eligible_for_training_library"):
            if receipt.get(key) is not False:
                raise ValueError(f"eligibility violation in {key}")
    checks = {
        "four_bit_loaded": training.get("four_bit_loaded") is True,
        "exact_100_optimizer_updates": training.get("optimizer_updates") == 100,
        "finite_nonzero_gradients": float(training.get("gradient_norm_min", 0.0)) > 0.0,
        "adapter_hashes_present": bool(training.get("adapter_file_hashes")),
        "fresh_process_reload": reload.get("fresh_process_reload") is True,
        "adapter_hashes_verified": reload.get("adapter_file_hashes_verified") is True,
        "offline_inference": reload.get("offline_inference_passed") is True,
    }
    report = {
        "schema_version": 1,
        "study_kind": "qlora_pipeline_smoke",
        "status": "passed" if all(checks.values()) else "failed",
        "scientific_claims_allowed": False,
        "eligible_for_champion": False,
        "eligible_for_training_library": False,
        "checks": checks,
        "training_receipt_digest": training["receipt_digest"],
        "reload_receipt_digest": reload["receipt_digest"],
        "freeze_receipt_digest": freeze["receipt_digest"],
        "metrics": {
            "optimizer_updates": training["optimizer_updates"],
            "loss_first": training["loss_first"],
            "loss_last": training["loss_last"],
            "gradient_norm_min": training["gradient_norm_min"],
            "gradient_norm_max": training["gradient_norm_max"],
            "cuda_peak_memory_bytes": training["cuda_peak_memory_bytes"],
            "trainable_parameter_count": training["trainable_parameter_count"],
        },
        "limitations": [
            "Synthetic/toy data only.",
            "One 7B adapter seed only.",
            "No paper-level VDS, causal comparison, power estimate, promotion, or recursive evolution.",
        ],
        "report_digest": "",
    }
    report["report_digest"] = hashlib.sha256(canonical_json(report).encode("utf-8")).hexdigest()
    path = args.output_root / "SMOKE_REPORT.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
