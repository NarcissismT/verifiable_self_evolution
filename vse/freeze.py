from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .hashing import content_hash, file_hash
from .ledger import RunLedger


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


def _verified_json(path: Path, digest_field: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    declared = str(value.get(digest_field, ""))
    blank_payload = dict(value)
    blank_payload[digest_field] = ""
    omitted_payload = dict(value)
    omitted_payload.pop(digest_field, None)
    if not declared or declared not in {
        content_hash(blank_payload),
        content_hash(omitted_payload),
    }:
        raise ValueError(f"{path.name}:{digest_field}_mismatch")
    return value


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
    if int(config.get("evolution", {}).get("maximum_generations", 0)) != 1:
        failures.append("primary_study_must_be_non_recursive")
    if int(config.get("evolution", {}).get("maximum_promotion_attempts", 0)) != 1:
        failures.append("primary_study_must_have_one_promotion_attempt")

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
    reserve_minima = {
        str(key): int(value)
        for key, value in config.get("capsule", {})
        .get("reserve_minimum_by_stratum", {})
        .items()
    }
    formal_strata = {
        stratum for split in quotas.values() for stratum in split
    }
    if set(reserve_minima) != formal_strata or any(
        value <= 0 for value in reserve_minima.values()
    ):
        failures.append("reserve_minimum_by_stratum_is_incomplete")
    if pool_minimum < total_required + sum(reserve_minima.values()):
        failures.append("candidate_pool_minimum_below_splits_plus_reserve")

    if config.get("power", {}).get("status") != "confirmed_from_independent_pilot":
        failures.append("final_sample_sizes_pending_power_confirmation")
    shift_definitions = config.get("ood_shift_definitions", {})
    for stratum in ("ood_constrained_bilevel_differentiable_optimization", "ood_safe_control_learning_to_optimize"):
        definition = shift_definitions.get(stratum, {})
        for key in ("shift_id", "shifted_variables", "id_boundary", "ood_boundary", "rationale"):
            if not definition.get(key):
                failures.append(f"missing_ood_shift_definition:{stratum}:{key}")
        if definition.get("dev_exposure") is not False:
            failures.append(f"ood_shift_is_exposed_to_dev:{stratum}")
        if definition.get("independent_review_required") is not True:
            failures.append(f"ood_independent_review_not_required:{stratum}")
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

    required_paths = (
        ("manifests/paper_selection.json", "paper_selection"),
        ("manifests/candidate_pool.json", "candidate_pool"),
        ("manifests/capsule_task_manifest.json", "capsule_task_manifest"),
        ("manifests/capsule_audits.jsonl", "capsule_audits"),
        ("manifests/contamination_receipt.json", "contamination_receipt"),
        ("manifests/trusted_evaluator_receipt.json", "trusted_evaluator_receipt"),
        ("manifests/power_receipt.json", "power_receipt"),
        ("manifests/human_review_rubric.json", "human_review_rubric"),
        ("manifests/base_checkpoint_receipt.json", "base_checkpoint_receipt"),
        ("manifests/freeze_bindings.json", "freeze_bindings"),
    )
    for relative, name in required_paths:
        _required_path(root, relative, name, failures, checked)

    if root is not None:
        try:
            config_path = root / "config.json"
            sealed_config = json.loads(config_path.read_text())
            declared_config = sealed_config.pop("config_digest", "")
            if content_hash(sealed_config) != declared_config:
                failures.append("config_internal_digest_mismatch")
            provided_config = dict(config)
            provided_config.pop("config_digest", None)
            if content_hash(provided_config) != declared_config:
                failures.append("provided_config_does_not_match_run_config")
            selection = _verified_json(root / "manifests" / "paper_selection.json", "assignment_digest")
            task_manifest = _verified_json(root / "manifests" / "capsule_task_manifest.json", "task_manifest_digest")
            candidate_pool = _verified_json(root / "manifests" / "candidate_pool.json", "candidate_pool_digest")
            assigned_ids: set[str] = set()
            for split_name, split_quotas in quotas.items():
                ids = [str(value) for value in selection.get("assignments", {}).get(split_name, [])]
                if len(ids) != sum(int(value) for value in split_quotas.values()):
                    failures.append(f"selection_assignment_count_mismatch:{split_name}")
                if assigned_ids & set(ids):
                    failures.append("selection_assignments_are_not_disjoint")
                assigned_ids.update(ids)
            reserved_ids = {str(value) for value in selection.get("reserved_ids", [])}
            if assigned_ids & reserved_ids:
                failures.append("selection_reserve_overlaps_assignment")
            strata_by_id = {
                str(key): str(value)
                for key, value in selection.get("strata_by_id", {}).items()
            }
            for stratum, minimum in reserve_minima.items():
                actual = sum(
                    strata_by_id.get(paper_id) == stratum for paper_id in reserved_ids
                )
                if actual < minimum:
                    failures.append(f"reserve_count_below_minimum:{stratum}")
            candidate_ids = {
                str(row.get("paper_id", ""))
                for row in candidate_pool.get("candidates", [])
            }
            if not assigned_ids | reserved_ids <= candidate_ids:
                failures.append("selection_contains_non_candidate_paper")
            task_ids = {
                str(row.get("paper_id", ""))
                for row in task_manifest.get("entries", [])
            }
            if task_ids != assigned_ids:
                failures.append("task_manifest_selection_grid_mismatch")
            audit_ids: set[str] = set()
            for line in (root / "manifests" / "capsule_audits.jsonl").read_text().splitlines():
                if not line.strip():
                    continue
                audit = json.loads(line)
                declared_audit = str(audit.get("audit_digest", ""))
                payload = dict(audit)
                payload.pop("audit_digest", None)
                if content_hash(payload) != declared_audit or not audit.get("passed"):
                    failures.append("capsule_audit_receipt_invalid")
                audit_ids.add(str(audit.get("capsule_id", "")))
            if audit_ids != assigned_ids:
                failures.append("capsule_audit_selection_grid_mismatch")
            for path, field in (
                (root / "manifests" / "contamination_receipt.json", "audit_digest"),
                (root / "manifests" / "trusted_evaluator_receipt.json", "receipt_digest"),
                (root / "manifests" / "base_checkpoint_receipt.json", "receipt_digest"),
                (root / "manifests" / "power_receipt.json", "receipt_digest"),
            ):
                _verified_json(path, field)
            evaluator = json.loads((root / "manifests" / "trusted_evaluator_receipt.json").read_text())
            contamination = json.loads((root / "manifests" / "contamination_receipt.json").read_text())
            base = json.loads((root / "manifests" / "base_checkpoint_receipt.json").read_text())
            power = json.loads((root / "manifests" / "power_receipt.json").read_text())
            if int(power.get("pilot_eval_n", 0)) < 12:
                failures.append("power_receipt_pilot_eval_n_below_12")
            if int(power.get("final_id_tasks", 0)) != int(quotas["heldout"].get("constrained_safe_rl", 0) + quotas["heldout"].get("general_sum_equilibrium_marl", 0) + quotas["heldout"].get("bilevel_stackelberg_alignment_optimization", 0)):
                failures.append("power_receipt_id_sample_mismatch")
            if int(power.get("final_ood_tasks", 0)) != sum(int(value) for value in quotas["ood"].values()):
                failures.append("power_receipt_ood_sample_mismatch")
            if float(power.get("target_power", 0.0)) < float(config.get("power", {}).get("target_power", 1.0)):
                failures.append("power_receipt_target_power_mismatch")
            if power.get("status") != "confirmed_from_independent_pilot":
                failures.append("power_receipt_status_unconfirmed")
            if set(power.get("covered_strata", [])) != formal_strata:
                failures.append("power_receipt_stratum_coverage_mismatch")
            if power.get("variance_strategy") != "max_across_strata":
                failures.append("power_receipt_variance_strategy_not_conservative")
            rubric = json.loads((root / "manifests" / "human_review_rubric.json").read_text())
            if rubric.get("vds_components") != sorted({"empirical", "hypothesis", "experiment", "novelty", "calibration"}):
                failures.append("rubric_vds_schema_unbound")
            if rubric.get("evaluator_digest") != evaluator.get("evaluator_digest"):
                failures.append("rubric_evaluator_binding_mismatch")
            bindings = _verified_json(root / "manifests" / "freeze_bindings.json", "freeze_bindings_digest")
            required_bindings = {
                "config_digest": declared_config,
                "paper_selection_digest": selection["assignment_digest"],
                "candidate_pool_digest": candidate_pool["candidate_pool_digest"],
                "task_manifest_digest": task_manifest["task_manifest_digest"],
                "power_receipt_digest": power["receipt_digest"],
                "rubric_digest": file_hash(root / "manifests" / "human_review_rubric.json"),
                "evaluator_digest": evaluator.get("evaluator_digest"),
                "contamination_audit_digest": contamination.get("audit_digest"),
                "base_checkpoint_digest": base.get("checkpoint_digest"),
                "base_group_digest": base.get("training_recipe_digest")
                or content_hash(
                    {
                        "group_kind": "frozen_base",
                        "checkpoint_digest": base.get("checkpoint_digest"),
                    }
                ),
            }
            for key, expected in required_bindings.items():
                if bindings.get(key) != expected:
                    failures.append(f"freeze_binding_mismatch:{key}")
            ledger = RunLedger(root / "ledger" / "events.jsonl")
            entries = ledger.validate()
            anchor_path = root / "ledger" / "head_anchor.json"
            if not entries or not anchor_path.is_file():
                failures.append("ledger_head_anchor_missing")
            else:
                anchor = json.loads(anchor_path.read_text())
                if anchor.get("head_hash") != entries[-1].entry_hash or anchor.get("freeze_bindings_digest") != bindings["freeze_bindings_digest"]:
                    failures.append("ledger_head_anchor_mismatch")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            failures.append(f"freeze_bundle_content_error:{type(error).__name__}")

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
