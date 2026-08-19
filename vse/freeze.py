from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .hashing import content_hash, file_hash


@dataclass(frozen=True)
class FreezeCheckReport:
    experiment_id: str
    ready: bool
    failures: tuple[str, ...]
    checked_artifacts: dict[str, str]
    report_digest: str = ""

    def sealed(self) -> "FreezeCheckReport":
        value = asdict(self)
        value["report_digest"] = ""
        return FreezeCheckReport(
            **{**asdict(self), "report_digest": content_hash(value)}
        )


def _required_path(
    root: Path | None,
    relative: str,
    name: str,
    failures: list[str],
    checked: dict[str, str],
) -> None:
    if root is None:
        failures.append(f"unchecked_required_artifact:{name}")
        return
    path = root / relative
    if not path.is_file():
        failures.append(f"missing_required_artifact:{name}")
        return
    checked[name] = file_hash(path)


def check_freeze(config: dict[str, Any], root: Path | None = None) -> FreezeCheckReport:
    failures: list[str] = []
    checked: dict[str, str] = {}
    readiness = config.get("freeze_readiness", {})
    if readiness.get("status") != "formally_frozen_ready_to_launch":
        failures.append(f"freeze_status:{readiness.get('status', 'missing')}")
    for item in readiness.get("pending", []):
        failures.append(f"pending:{item}")

    design = config.get("research_design", {})
    required_design = {
        "primary_endpoint",
        "primary_comparison",
        "secondary_comparisons",
        "final_comparator",
        "outer_statistical_unit",
        "recursive_evolution_in_primary_study",
    }
    for key in sorted(required_design - set(design)):
        failures.append(f"missing_research_design:{key}")
    if design.get("primary_endpoint") != "verified_discovery_score":
        failures.append("primary_endpoint_is_not_vds")
    if design.get("final_comparator") != "frozen_base":
        failures.append("final_comparator_is_not_frozen_base")

    constraints = config.get("split_constraints", {})
    quotas = config.get("selection_quotas", {})
    required_splits = {"train", "dev", "promotion", "heldout", "ood"}
    if set(constraints) != required_splits:
        failures.append("split_constraints_are_not_complete")
    if set(quotas) != required_splits:
        failures.append("selection_quotas_are_not_complete")
    total_required = 0
    for split_name in sorted(required_splits):
        constraint = constraints.get(split_name, {})
        required = int(
            constraint.get(
                "exact",
                constraint.get("minimum", constraint.get("planned_exact", 0)),
            )
        )
        total_required += required
        if "planned_exact" in constraint:
            failures.append(f"split_size_pending_power_confirmation:{split_name}")
        if sum(int(value) for value in quotas.get(split_name, {}).values()) != required:
            failures.append(f"selection_quota_count_mismatch:{split_name}")
    pool_minimum = int(config.get("capsule", {}).get("candidate_pool_minimum", 0))
    if pool_minimum < total_required:
        failures.append("candidate_pool_minimum_below_split_total")

    if config.get("power", {}).get("status") != "confirmed_from_independent_pilot":
        failures.append("final_sample_sizes_pending_power_confirmation")
    for model in config.get("models", []):
        if not model.get("checkpoint_file_hashes"):
            failures.append(f"missing_model_checkpoint_hashes:{model.get('model_id', 'unknown')}")
        if not model.get("tokenizer_file_hashes"):
            failures.append(f"missing_tokenizer_hashes:{model.get('model_id', 'unknown')}")
    containers = config.get("containers", {})
    for key in (
        "train_image_digest",
        "evaluator_image_digest",
        "trusted_evaluator_repository_commit",
    ):
        if not containers.get(key):
            failures.append(f"missing_container_binding:{key}")
    for key, value in config.get("decoding", {}).items():
        if key in {"freeze_before_generation", "agent_action_and_token_budget_is_frozen"}:
            continue
        if value is None:
            failures.append(f"missing_decoding_value:{key}")

    for relative, name in (
        ("manifests/paper_selection.json", "paper_selection"),
        ("manifests/candidate_pool.json", "candidate_pool"),
        ("manifests/capsule_audits.jsonl", "capsule_audits"),
        ("manifests/contamination_receipt.json", "contamination_receipt"),
        ("manifests/trusted_evaluator_receipt.json", "trusted_evaluator_receipt"),
        ("manifests/power_receipt.json", "power_receipt"),
        ("manifests/human_review_rubric.json", "human_review_rubric"),
        ("manifests/freeze_bindings.json", "freeze_bindings"),
    ):
        _required_path(root, relative, name, failures, checked)

    return FreezeCheckReport(
        experiment_id=str(config.get("experiment_id", "missing")),
        ready=not failures,
        failures=tuple(sorted(set(failures))),
        checked_artifacts=checked,
    ).sealed()


def write_report(path: Path, report: FreezeCheckReport) -> None:
    serialized = json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != serialized:
        raise FileExistsError(f"refusing to replace freeze check report: {path}")
    path.write_text(serialized)
