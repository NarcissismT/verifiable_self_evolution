from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Any

from .contracts import Split, TrajectoryRecord


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    descriptor = os.open(path, flags, 0o640)
    try:
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(json.dumps(value, ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except Exception:
        raise


class TrajectoryRouter:
    def __init__(self, root: Path):
        self.root = root

    def route(self, record: TrajectoryRecord) -> Path:
        split = record.task.split
        if record.verification.split is not split:
            raise ValueError("task and verification split disagree")
        if split is Split.TRAIN:
            name = "success.jsonl" if record.verification.accepted else "counterexample.jsonl"
            destination = self.root / "libraries" / name
        elif split is Split.DEV:
            destination = self.root / "diagnostics" / "dev.jsonl"
        elif split is Split.PROMOTION:
            destination = self.root / "promotion" / "promotion_receipts.jsonl"
        else:
            destination = self.root / "final" / f"{split.value}_receipts.jsonl"
        append_jsonl(destination, record.payload())
        return destination


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def export_sft(success_library: Path, output: Path) -> int:
    rows = read_jsonl(success_library)
    exported: list[dict[str, str]] = []
    for row in rows:
        if row["task"]["split"] != Split.TRAIN.value:
            raise ValueError("non-training record found in success library")
        if not row["verification"]["accepted"]:
            raise ValueError("rejected record found in success library")
        proposal = row["proposal"]
        answer = {
            "hypothesis": proposal["hypothesis"],
            "solution": proposal["solution"],
            "experiment_code": proposal["experiment_code"],
            "seeds": proposal["seeds"],
            "baselines": proposal["baselines"],
            "primary_metric": proposal["primary_metric"],
            "secondary_metrics": proposal["secondary_metrics"],
            "expected_effect": proposal["expected_effect"],
            "power_assumptions": proposal["power_assumptions"],
            "stopping_rule": proposal["stopping_rule"],
            "resource_schedule": proposal["resource_schedule"],
        }
        exported.append(
            {
                "prompt": row["task"]["statement"],
                "completion": json.dumps(answer, ensure_ascii=True, sort_keys=True),
                "task_id": row["task"]["task_id"],
                "verification_digest": row["verification"]["report_digest"],
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size:
        raise FileExistsError(f"refusing to overwrite nonempty dataset: {output}")
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in exported)
    )
    return len(exported)
