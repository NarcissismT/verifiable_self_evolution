from __future__ import annotations

from dataclasses import asdict, dataclass
import fcntl
import json
import os
from pathlib import Path
from typing import Any

from .hashing import content_hash


@dataclass(frozen=True)
class LedgerEntry:
    sequence: int
    event_type: str
    payload: dict[str, Any]
    bindings: dict[str, str]
    previous_hash: str
    entry_hash: str = ""

    def sealed(self) -> "LedgerEntry":
        value = asdict(self)
        value["entry_hash"] = ""
        return LedgerEntry(**{**asdict(self), "entry_hash": content_hash(value)})


class RunLedger:
    """Hash-chained append-only run ledger.

    The ledger is deliberately a small file protocol so it can be copied into
    an artifact bundle and verified without a database service. The hash chain
    detects local edits; an anchored head is required for external tamper
    resistance.
    """

    def __init__(self, path: Path):
        self.path = path

    def _read(self) -> list[LedgerEntry]:
        if not self.path.exists():
            return []
        entries: list[LedgerEntry] = []
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    entry = LedgerEntry(
                        sequence=int(value["sequence"]),
                        event_type=str(value["event_type"]),
                        payload=dict(value["payload"]),
                        bindings={str(k): str(v) for k, v in value["bindings"].items()},
                        previous_hash=str(value["previous_hash"]),
                        entry_hash=str(value["entry_hash"]),
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(f"invalid ledger entry at line {line_number}") from error
                if entry.sealed().entry_hash != entry.entry_hash:
                    raise ValueError(f"ledger entry hash mismatch at line {line_number}")
                entries.append(entry)
        previous = ""
        for expected_sequence, entry in enumerate(entries, 1):
            if entry.sequence != expected_sequence:
                raise ValueError("ledger sequence is not contiguous")
            if entry.previous_hash != previous:
                raise ValueError("ledger previous hash mismatch")
            previous = entry.entry_hash
        return entries

    def validate(self) -> tuple[LedgerEntry, ...]:
        return tuple(self._read())

    def append(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        bindings: dict[str, str],
    ) -> LedgerEntry:
        if not event_type:
            raise ValueError("ledger event_type is required")
        if any(not key or not value for key, value in bindings.items()):
            raise ValueError("ledger bindings must have nonempty keys and values")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        descriptor = os.open(self.path, flags, 0o640)
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            entries = self._read()
            previous_hash = entries[-1].entry_hash if entries else ""
            entry = LedgerEntry(
                sequence=len(entries) + 1,
                event_type=event_type,
                payload=payload,
                bindings=dict(sorted(bindings.items())),
                previous_hash=previous_hash,
            ).sealed()
            handle.write(json.dumps(asdict(entry), ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return entry

    def anchor_head(self, *, event_type: str, freeze_bindings_digest: str) -> Path:
        entries = self.validate()
        if not entries:
            raise ValueError("cannot anchor an empty ledger")
        head = entries[-1]
        anchor = {
            "sequence": head.sequence,
            "event_type": event_type,
            "head_hash": head.entry_hash,
            "freeze_bindings_digest": freeze_bindings_digest,
        }
        anchor["anchor_digest"] = content_hash(anchor)
        path = self.path.parent / "anchors" / f"{head.sequence:06d}-{event_type}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(anchor, indent=2, sort_keys=True) + "\n"
        if path.exists() and path.read_text() != serialized:
            raise FileExistsError(f"refusing to replace ledger head anchor: {path}")
        path.write_text(serialized)
        # Keep a small latest-head pointer for validators; immutable snapshots
        # remain in `ledger/anchors/` for every binding event.
        latest = self.path.parent / "head_anchor.json"
        latest.write_text(serialized)
        return path
