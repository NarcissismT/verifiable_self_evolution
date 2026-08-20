#!/usr/bin/env python3
"""Create the frozen SDK source/card/wrapper/image binding manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdk-root", type=Path, required=True)
    parser.add_argument("--execution-image-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.execution_image_digest.startswith("sha256:"):
        raise SystemExit("execution image digest must be sha256:<hex>")
    sdk_root = args.sdk_root
    card = sdk_root / "sdk_card.json"
    binding = {
        "schema_version": 1,
        "sdk_version": "v0.1.5-sdk-v1",
        "sdk_source_digest": sha256(sdk_root / "execution_sdk.py"),
        "sdk_card_digest": sha256(card),
        "sdk_lock_digest": sha256(sdk_root / "sdk_lock.json"),
        "wrapper_digest": sha256(sdk_root / "trusted_wrapper.py"),
        "numpy_version": "2.1.2",
        "execution_image_digest": args.execution_image_digest,
        "network_policy": "none",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n")
    print(json.dumps(binding, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
