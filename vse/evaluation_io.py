from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Iterable

from .contracts import Split
from .promotion import EvaluationCell, PromotionDecision


def read_evaluation_cells(path: Path) -> tuple[EvaluationCell, ...]:
    rows: list[EvaluationCell] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                value["split"] = Split(value["split"])
                value["vds_components"] = {
                    str(key): float(component)
                    for key, component in value["vds_components"].items()
                }
                rows.append(EvaluationCell(**value))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid evaluation cell at line {line_number}") from error
    if not rows:
        raise ValueError(f"evaluation receipt is empty: {path}")
    return tuple(rows)


def write_decision_once(path: Path, decision: PromotionDecision) -> None:
    value = asdict(decision)
    value["split_decisions"] = [
        {**item, "split": item["split"].value}
        for item in value["split_decisions"]
    ]
    serialized = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != serialized:
        raise FileExistsError(f"refusing to replace evaluation decision: {path}")
    path.write_text(serialized)

