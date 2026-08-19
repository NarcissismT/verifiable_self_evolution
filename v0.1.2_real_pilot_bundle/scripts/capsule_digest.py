#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("capsule", type=Path)
    args = parser.parse_args()
    value: Any = json.loads(args.capsule.read_text(encoding="utf-8"))
    canonical = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    print(hashlib.sha256(canonical.encode("utf-8")).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
