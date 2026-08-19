from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Any

from .contracts import Split, TrajectoryRecord
from .hashing import content_hash


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


def verify_trajectory_payload(row: dict[str, Any], *, require_accepted: bool) -> None:
    task = dict(row["task"])
    proposal = dict(row["proposal"])
    verification = dict(row["verification"])
    executions = list(row["executions"])
    if task.get("split") != Split.TRAIN.value:
        raise ValueError("training export contains a non-training trajectory")
    task_digest = content_hash(task)
    proposal_digest = content_hash(proposal)
    if proposal.get("task_id") != task.get("task_id"):
        raise ValueError("proposal and task identity mismatch")
    execution_digests: list[str] = []
    for execution in executions:
        if execution.get("task_id") != task.get("task_id"):
            raise ValueError("execution and task identity mismatch")
        if execution.get("proposal_digest") != proposal_digest:
            raise ValueError("execution proposal digest mismatch")
        if not execution.get("trusted_evaluator_digest"):
            raise ValueError("execution is missing a trusted evaluator digest")
        execution_digests.append(content_hash(execution))
    if tuple(verification.get("execution_digests", ())) != tuple(execution_digests):
        raise ValueError("verification execution hash chain mismatch")
    report_digest = verification.get("report_digest", "")
    verification["report_digest"] = ""
    if content_hash(verification) != report_digest:
        raise ValueError("verification report digest mismatch")
    if verification.get("task_id") != task.get("task_id"):
        raise ValueError("verification and task identity mismatch")
    if require_accepted and (
        not verification.get("accepted") or verification.get("hard_failures")
    ):
        raise ValueError("unverified trajectory found in accepted training data")
    if not task_digest:
        raise AssertionError("unreachable empty task digest")


def _training_row(row: dict[str, Any], bucket: str) -> dict[str, str]:
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
    return {
        "prompt": row["task"]["statement"],
        "completion": json.dumps(answer, ensure_ascii=True, sort_keys=True),
        "task_id": row["task"]["task_id"],
        "verification_digest": row["verification"]["report_digest"],
        "bucket": bucket,
    }


def _write_rows_once(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = "".join(
        json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows
    )
    if path.exists() and path.read_text() != serialized:
        raise FileExistsError(f"refusing to replace training dataset: {path}")
    path.write_text(serialized)


def export_sft(success_library: Path, output: Path) -> int:
    rows = read_jsonl(success_library)
    exported: list[dict[str, str]] = []
    for row in rows:
        verify_trajectory_payload(row, require_accepted=True)
        exported.append(_training_row(row, "verified_success"))
    _write_rows_once(output, exported)
    return len(exported)


def export_training_datasets(
    success_library: Path,
    counterexample_library: Path,
    output_dir: Path,
) -> dict[str, int]:
    teacher_rows: list[dict[str, str]] = []
    verified_rows: list[dict[str, str]] = []
    corrected_rows: list[dict[str, str]] = []
    for row in read_jsonl(success_library):
        verify_trajectory_payload(row, require_accepted=True)
        bucket = (
            "teacher_anchor"
            if row["proposal"].get("source") == "closed_teacher"
            else "verified_success"
        )
        destination = teacher_rows if bucket == "teacher_anchor" else verified_rows
        destination.append(_training_row(row, bucket))
    for row in read_jsonl(counterexample_library):
        verify_trajectory_payload(row, require_accepted=False)
        correction = row.get("verified_correction")
        if correction is None:
            continue
        corrected_record = correction.get("trajectory")
        if not isinstance(corrected_record, dict):
            raise ValueError("corrected counterexample is missing its trajectory")
        verify_trajectory_payload(corrected_record, require_accepted=True)
        if corrected_record["task"]["task_id"] != row["task"]["task_id"]:
            raise ValueError("correction targets a different task")
        corrected_rows.append(_training_row(corrected_record, "corrected_counterexample"))
    outputs = {
        "teacher_anchor": teacher_rows,
        "verified_success": verified_rows,
        "corrected_counterexample": corrected_rows,
    }
    for name, rows in outputs.items():
        _write_rows_once(output_dir / f"{name}.jsonl", rows)
    manifest = {
        "schema_version": 1,
        "counts": {name: len(rows) for name, rows in outputs.items()},
        "mixture": {
            "teacher_anchor": 0.50,
            "verified_success": 0.30,
            "corrected_counterexample": 0.20,
        },
        "files": {name: f"{name}.jsonl" for name in outputs},
    }
    manifest["manifest_digest"] = content_hash(manifest)
    manifest_path = output_dir / "dataset_manifest.json"
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if manifest_path.exists() and manifest_path.read_text() != serialized:
        raise FileExistsError(f"refusing to replace dataset manifest: {manifest_path}")
    manifest_path.write_text(serialized)
    return {name: len(rows) for name, rows in outputs.items()}
