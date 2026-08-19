from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

from .contracts import Split, Task
from .hashing import content_hash


@dataclass(frozen=True)
class ManifestEntry:
    task_id: str
    split: Split
    task_digest: str

    def payload(self) -> dict[str, str]:
        return {
            "task_id": self.task_id,
            "split": self.split.value,
            "task_digest": self.task_digest,
        }


@dataclass(frozen=True)
class FrozenManifest:
    experiment_id: str
    entries: tuple[ManifestEntry, ...]
    config_digest: str
    manifest_digest: str = ""

    def payload(self, include_digest: bool = True) -> dict:
        value = {
            "experiment_id": self.experiment_id,
            "config_digest": self.config_digest,
            "entries": [entry.payload() for entry in self.entries],
        }
        if include_digest:
            value["manifest_digest"] = self.manifest_digest
        return value

    def sealed(self) -> "FrozenManifest":
        return FrozenManifest(
            experiment_id=self.experiment_id,
            entries=self.entries,
            config_digest=self.config_digest,
            manifest_digest=content_hash(self.payload(include_digest=False)),
        )

    def verify_task(self, task: Task) -> None:
        matches = [entry for entry in self.entries if entry.task_id == task.task_id]
        if len(matches) != 1:
            raise ValueError(f"task is not uniquely registered: {task.task_id}")
        entry = matches[0]
        if entry.split is not task.split:
            raise ValueError(f"split mismatch for task {task.task_id}")
        if entry.task_digest != task.digest:
            raise ValueError(f"content drift for frozen task {task.task_id}")


def build_manifest(
    experiment_id: str, tasks: Iterable[Task], config: dict
) -> FrozenManifest:
    entries = tuple(
        sorted(
            (
                ManifestEntry(
                    task_id=task.task_id,
                    split=task.split,
                    task_digest=task.digest,
                )
                for task in tasks
            ),
            key=lambda entry: entry.task_id,
        )
    )
    if len({entry.task_id for entry in entries}) != len(entries):
        raise ValueError("duplicate task id in manifest")
    required = {Split.TRAIN, Split.DEV, Split.PROMOTION, Split.HELDOUT, Split.OOD}
    present = {entry.split for entry in entries}
    missing = required - present
    if missing:
        raise ValueError(f"manifest is missing splits: {sorted(item.value for item in missing)}")
    constraints = config.get("split_constraints", {})
    for split_name, expected in constraints.items():
        split = Split(split_name)
        actual = sum(entry.split is split for entry in entries)
        if isinstance(expected, int) and actual != expected:
            raise ValueError(
                f"frozen split count mismatch for {split_name}: "
                f"expected={expected}, actual={actual}"
            )
        if isinstance(expected, dict):
            if "exact" in expected and actual != int(expected["exact"]):
                raise ValueError(
                    f"frozen split count mismatch for {split_name}: "
                    f"expected={expected['exact']}, actual={actual}"
                )
            if "minimum" in expected and actual < int(expected["minimum"]):
                raise ValueError(
                    f"frozen split minimum not met for {split_name}: "
                    f"minimum={expected['minimum']}, actual={actual}"
                )
            if "planned_exact" in expected and actual != int(expected["planned_exact"]):
                raise ValueError(
                    f"planned split count mismatch for {split_name}: "
                    f"expected={expected['planned_exact']}, actual={actual}"
                )
    return FrozenManifest(
        experiment_id=experiment_id,
        entries=entries,
        config_digest=content_hash(config),
    ).sealed()


def write_manifest_once(path: Path, manifest: FrozenManifest) -> None:
    if path.exists():
        current = json.loads(path.read_text())
        if current != manifest.payload():
            raise FileExistsError(f"refusing to replace frozen manifest: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.payload(), indent=2, sort_keys=True) + "\n")


def load_manifest(path: Path) -> FrozenManifest:
    value = json.loads(path.read_text())
    manifest = FrozenManifest(
        experiment_id=value["experiment_id"],
        config_digest=value["config_digest"],
        entries=tuple(
            ManifestEntry(
                task_id=item["task_id"],
                split=Split(item["split"]),
                task_digest=item["task_digest"],
            )
            for item in value["entries"]
        ),
        manifest_digest=value["manifest_digest"],
    )
    if manifest.sealed().manifest_digest != manifest.manifest_digest:
        raise ValueError(f"manifest digest mismatch: {path}")
    return manifest
