from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .hashing import content_hash, file_hash
from .ledger import RunLedger
from .paper_capsule import audit_capsule, capsule_from_mapping, sealed_target_from_mapping
from .semantic_review import load_semantic_review
from .paper_selection import (
    PaperCandidate,
    candidate_exclusion_reasons,
    eligible_candidates,
    select_frozen_papers,
)


def load_candidates_jsonl(path: Path) -> tuple[PaperCandidate, ...]:
    candidates: list[PaperCandidate] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                candidates.append(
                    PaperCandidate(
                        paper_id=value["paper_id"],
                        stratum=value["stratum"],
                        venue=value["venue"],
                        proceedings_year=int(value["proceedings_year"]),
                        official_main_or_proceedings=bool(value["official_main_or_proceedings"]),
                        public_paper=bool(value["public_paper"]),
                        public_code=bool(value["public_code"]),
                        public_data=bool(value["public_data"]),
                        requires_commercial_api=bool(value["requires_commercial_api"]),
                        requires_private_data=bool(value["requires_private_data"]),
                        requires_real_robot=bool(value["requires_real_robot"]),
                        estimated_gpu_hours=float(value["estimated_gpu_hours"]),
                        estimated_cpu_hours=float(value["estimated_cpu_hours"]),
                        public_timestamps_utc=dict(value["public_timestamps_utc"]),
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid candidate at line {line_number}") from error
    return tuple(candidates)


def initialize_formal_run(
    config: dict[str, Any], candidates_path: Path, root: Path
) -> dict[str, Any]:
    candidates = load_candidates_jsonl(candidates_path)
    window = config["target_publication_window"]
    resources = config["resource_budget"]
    eligible = eligible_candidates(
        candidates,
        publication_start_utc=window["start_utc"],
        publication_end_utc=window["end_utc"],
        max_gpu_hours=float(resources["gpu_hours_per_agent_task"]),
        max_cpu_hours=float(resources["cpu_hours_per_agent_task"]),
    )
    candidate_pool = {
        "schema_version": 1,
        "candidates": [
            {
                **asdict(candidate),
                "eligible": not candidate_exclusion_reasons(
                    candidate,
                    publication_start_utc=window["start_utc"],
                    publication_end_utc=window["end_utc"],
                    max_gpu_hours=float(resources["gpu_hours_per_agent_task"]),
                    max_cpu_hours=float(resources["cpu_hours_per_agent_task"]),
                ),
                "exclusion_reasons": list(
                    candidate_exclusion_reasons(
                        candidate,
                        publication_start_utc=window["start_utc"],
                        publication_end_utc=window["end_utc"],
                        max_gpu_hours=float(resources["gpu_hours_per_agent_task"]),
                        max_cpu_hours=float(resources["cpu_hours_per_agent_task"]),
                    )
                ),
            }
            for candidate in candidates
        ],
    }
    candidate_pool["candidate_pool_digest"] = content_hash(candidate_pool)
    selection = select_frozen_papers(
        eligible,
        config["selection_quotas"],
        sampling_seed=int(config["seeds"]["manifest_sampling"]),
        minimum_candidate_pool=int(config["capsule"]["candidate_pool_minimum"]),
        cutoff_days=int(config["cutoff"]["days_before_target"]),
        reserve_minimum_by_stratum=config["capsule"].get(
            "reserve_minimum_by_stratum", {}
        ),
    )
    sealed_config = dict(config)
    config_digest = content_hash(config)
    sealed_config["config_digest"] = config_digest
    config_path = root / "config.json"
    selection_path = root / "manifests" / "paper_selection.json"
    candidate_pool_path = root / "manifests" / "candidate_pool.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    config_serialized = json.dumps(sealed_config, indent=2, sort_keys=True) + "\n"
    selection_serialized = json.dumps(asdict(selection), indent=2, sort_keys=True) + "\n"
    candidate_pool_serialized = json.dumps(candidate_pool, indent=2, sort_keys=True) + "\n"
    for path, serialized in (
        (config_path, config_serialized),
        (selection_path, selection_serialized),
        (candidate_pool_path, candidate_pool_serialized),
    ):
        if path.exists() and path.read_text() != serialized:
            raise FileExistsError(f"refusing to replace frozen artifact: {path}")
        path.write_text(serialized)
    bindings = {
        "config_digest": config_digest,
        "paper_selection_digest": selection.assignment_digest,
    }
    ledger = RunLedger(root / "ledger" / "events.jsonl")
    if not ledger.validate():
        ledger.append(
            "formal_run_initialized",
            {
                "experiment_id": config["experiment_id"],
                "eligible_candidate_count": len(eligible),
                "assigned_count": sum(len(ids) for ids in selection.assignments.values()),
                "reserve_count": len(selection.reserved_ids),
            },
            bindings=bindings,
        )
    return {
        "config_digest": config_digest,
        "paper_selection_digest": selection.assignment_digest,
        "eligible_candidate_count": len(eligible),
        "assigned_count": sum(len(ids) for ids in selection.assignments.values()),
        "reserve_count": len(selection.reserved_ids),
    }


def _write_once(path: Path, serialized: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != serialized:
        raise FileExistsError(f"refusing to replace frozen artifact: {path}")
    path.write_text(serialized)


def seal_formal_capsules(
    root: Path,
    capsule_index: Path,
    *,
    public_root: Path,
    sealed_root: Path,
) -> dict[str, Any]:
    selection = json.loads((root / "manifests" / "paper_selection.json").read_text())
    expected: dict[str, str] = {}
    for split_name, paper_ids in selection["assignments"].items():
        for paper_id in paper_ids:
            expected[paper_id] = split_name
    index_rows = [
        json.loads(line)
        for line in capsule_index.read_text().splitlines()
        if line.strip()
    ]
    if len({row.get("paper_id") for row in index_rows}) != len(index_rows):
        raise ValueError("capsule index contains duplicate paper_id values")
    actual_ids = {row.get("paper_id") for row in index_rows}
    if actual_ids != set(expected):
        missing = sorted(set(expected) - actual_ids)
        extra = sorted(actual_ids - set(expected))
        raise ValueError(f"capsule index grid mismatch: missing={missing}, extra={extra}")
    manifest_entries: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for row in sorted(index_rows, key=lambda item: item["paper_id"]):
        paper_id = row["paper_id"]
        capsule_path = (public_root / row["capsule_json"]).resolve()
        target_path = (sealed_root / row["target_json"]).resolve()
        capsule_path.relative_to(public_root.resolve())
        target_path.relative_to(sealed_root.resolve())
        capsule = capsule_from_mapping(json.loads(capsule_path.read_text()))
        target = sealed_target_from_mapping(json.loads(target_path.read_text()))
        if capsule.capsule_id != paper_id:
            raise ValueError(f"capsule id does not match paper id: {paper_id}")
        if capsule.split.value != expected[paper_id]:
            raise ValueError(f"capsule split mismatch: {paper_id}")
        if capsule.cutoff_utc != selection["cutoffs_utc"][paper_id]:
            raise ValueError(f"capsule cutoff mismatch: {paper_id}")
        semantic_review_path = (public_root / row["semantic_review_receipt"]).resolve()
        semantic_review_path.relative_to(public_root.resolve())
        if not semantic_review_path.is_file():
            raise FileNotFoundError(semantic_review_path)
        load_semantic_review(
            semantic_review_path,
            capsule=capsule,
            target=target,
        )
        audit = audit_capsule(capsule, target, public_root, sealed_root)
        audit_value = asdict(audit)
        audit_value["audit_digest"] = content_hash(audit_value)
        audit_rows.append(audit_value)
        if not audit.passed:
            raise ValueError(f"capsule audit failed: {paper_id}: {audit.failures}")
        manifest_entries.append(
            {
                "paper_id": paper_id,
                "split": capsule.split.value,
                "cutoff_utc": capsule.cutoff_utc,
                "capsule_digest": capsule.digest,
                "target_commitment": capsule.target_commitment,
                "capsule_audit_digest": audit_value["audit_digest"],
                "semantic_leak_review_digest": capsule.semantic_leak_review_digest,
            }
        )
    manifest = {
        "schema_version": 1,
        "experiment_id": json.loads((root / "config.json").read_text())["experiment_id"],
        "paper_selection_digest": selection["assignment_digest"],
        "entries": manifest_entries,
    }
    manifest["task_manifest_digest"] = content_hash(manifest)
    _write_once(
        root / "manifests" / "capsule_task_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    _write_once(
        root / "manifests" / "capsule_audits.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in audit_rows),
    )
    return {
        "task_count": len(manifest_entries),
        "task_manifest_digest": manifest["task_manifest_digest"],
    }


def _verified_receipt(path: Path, digest_field: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    declared = str(value.get(digest_field, ""))
    payload = dict(value)
    payload[digest_field] = ""
    if not declared or content_hash(payload) != declared:
        raise ValueError(f"receipt digest mismatch: {path}")
    return value


def bind_formal_freeze(
    root: Path,
    *,
    evaluator_receipt: Path,
    contamination_receipt: Path,
    base_checkpoint_receipt: Path,
) -> dict[str, str]:
    config = json.loads((root / "config.json").read_text())
    selection = json.loads((root / "manifests" / "paper_selection.json").read_text())
    candidate_pool = json.loads((root / "manifests" / "candidate_pool.json").read_text())
    task_manifest = json.loads(
        (root / "manifests" / "capsule_task_manifest.json").read_text()
    )
    task_digest = task_manifest.pop("task_manifest_digest", "")
    if content_hash(task_manifest) != task_digest:
        raise ValueError("capsule task manifest digest mismatch")
    evaluator = _verified_receipt(evaluator_receipt, "receipt_digest")
    contamination = _verified_receipt(contamination_receipt, "audit_digest")
    base = _verified_receipt(base_checkpoint_receipt, "receipt_digest")
    if not evaluator.get("evaluator_digest"):
        raise ValueError("evaluator receipt has no evaluator digest")
    contamination_digest = str(contamination.get("audit_digest", ""))
    if not contamination_digest:
        raise ValueError("contamination receipt has no audit digest")
    if not base.get("checkpoint_digest") or base.get("checkpoint_id") != "base":
        raise ValueError("base checkpoint receipt is incomplete")
    base_group_digest = str(base.get("training_recipe_digest", ""))
    if not base_group_digest:
        base_group_digest = content_hash(
            {
                "group_kind": "frozen_base",
                "checkpoint_digest": str(base["checkpoint_digest"]),
            }
        )
    bindings = {
        "config_digest": str(config["config_digest"]),
        "paper_selection_digest": str(selection["assignment_digest"]),
        "candidate_pool_digest": str(candidate_pool["candidate_pool_digest"]),
        "task_manifest_digest": str(task_digest),
        "evaluator_digest": str(evaluator["evaluator_digest"]),
        "contamination_audit_digest": contamination_digest,
        "base_checkpoint_digest": str(base["checkpoint_digest"]),
        "base_group_digest": base_group_digest,
    }
    power_path = root / "manifests" / "power_receipt.json"
    if power_path.is_file():
        power = _verified_receipt(power_path, "receipt_digest")
        bindings["power_receipt_digest"] = str(power["receipt_digest"])
    rubric_path = root / "manifests" / "human_review_rubric.json"
    if rubric_path.is_file():
        bindings["rubric_digest"] = file_hash(rubric_path)
    bindings["freeze_bindings_digest"] = content_hash(bindings)
    for path, value in (
        (root / "manifests" / "trusted_evaluator_receipt.json", evaluator),
        (root / "manifests" / "contamination_receipt.json", contamination),
        (root / "manifests" / "base_checkpoint_receipt.json", base),
    ):
        _write_once(path, json.dumps(value, indent=2, sort_keys=True) + "\n")
    _write_once(
        root / "manifests" / "freeze_bindings.json",
        json.dumps(bindings, indent=2, sort_keys=True) + "\n",
    )
    ledger = RunLedger(root / "ledger" / "events.jsonl")
    entries = ledger.validate()
    if entries:
        ledger.anchor_head(
            event_type="freeze_bindings",
            freeze_bindings_digest=bindings["freeze_bindings_digest"],
        )
        # `anchor_head` writes the latest pointer and an immutable snapshot.
    return bindings
