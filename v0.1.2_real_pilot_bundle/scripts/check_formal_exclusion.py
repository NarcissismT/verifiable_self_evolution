#!/usr/bin/env python3
"""Hard-fail if a real-pilot target appears anywhere in a formal selection manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable


def strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def normalized(value: str) -> str:
    return "".join(char.casefold() for char in value if char.isalnum())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    args = parser.parse_args()
    selected = "\n".join(strings(json.loads(args.selection_manifest.read_text(encoding="utf-8"))))
    selected_normalized = normalized(selected)
    exclusions = json.loads(args.exclusions.read_text(encoding="utf-8"))
    matches: list[str] = []
    for target in exclusions["targets"]:
        candidates = [target["target_id"], target["title"], *target["identifiers"]]
        for candidate in candidates:
            token = normalized(candidate)
            if len(token) >= 6 and token in selected_normalized:
                matches.append(f"{target['target_id']}:{candidate}")
    print(json.dumps({"passed": not matches, "matches": sorted(set(matches))}, indent=2))
    return 0 if not matches else 2


if __name__ == "__main__":
    sys.exit(main())
