#!/usr/bin/env python3
"""Run the v0.1.5 SDK-conformance confirmation gate."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import time
from typing import Any

from vse.contracts import CandidateSource, Split, Task
from vse.hashing import canonical_json, content_hash, file_hash
from vse.proposal_io import parse_model_proposal

from sdk_api_linter import lint_code


SEEDS = (1103, 2207, 3301, 4409)
BASELINES = ("sdk_zero_v1",)
METRICS = frozenset(("lower_residual", "kkt_ni_gap", "regret", "oracle_calls", "seed_reproducibility"))
SCHEDULE = (1000.0, 2000.0, 4000.0)
GATE = "v0.1.5_sdk_conformance_confirmation"
EVALUATOR_VERSION = "v0.1.5-sdk-conformance-evaluator-v1"


def task_for(case: dict[str, Any]) -> Task:
    return Task(task_id=str(case["case_id"]), family=str(case["family"]), split=Split.DEV,
                statement=str(case["public_problem"]), instance={"confirmation_case": True},
                verifier_version=EVALUATOR_VERSION,
                tags=("confirmation", "permanently-excluded", "sdk-conformance"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def proposal_binding(proposal: Any) -> dict[str, str]:
    hypothesis = proposal.payload()["hypothesis"]
    code = proposal.experiment_code
    hypothesis_digest = sha256_bytes(canonical_json(hypothesis).encode())
    code_digest = sha256_bytes(code.encode())
    return {"hypothesis_digest": hypothesis_digest, "code_digest": code_digest,
            "binding_digest": content_hash({"hypothesis_digest": hypothesis_digest, "code_digest": code_digest})}


def audit_code(code: str) -> None:
    lowered = code.lower()
    if any(token in lowered for token in ("socket", "requests", "urllib", "subprocess", "open(", "sealed",
                                         "neurips", "openreview", "arxiv", "/etc/", "target_digest")):
        raise ValueError("candidate contains a forbidden network/filesystem/target token")
    issues = lint_code(code)
    if issues:
        first = issues[0]
        raise ValueError(f"{first.code}@{first.line}: {first.message}")


def frozen_view(value: dict[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in value if key not in {"candidate_id", "experiment_code", "parent_candidate_ids"}}


def proposal_from_file(path: Path, case: dict[str, Any], model_digest: str):
    if not path.is_file():
        raise ValueError("final proposal is missing")
    value = json.loads(path.read_text())
    task = task_for(case)
    if value.get("task_id") != task.task_id or value.get("model_digest") != model_digest:
        raise ValueError("proposal identity mismatch")
    subset = {key: value[key] for key in (
        "hypothesis", "solution", "experiment_code", "seeds", "baselines", "primary_metric",
        "secondary_metrics", "expected_effect", "power_assumptions", "stopping_rule", "resource_schedule")}
    parent_ids = tuple(str(item) for item in value.get("parent_candidate_ids", []))
    proposal = parse_model_proposal(json.dumps(subset), task=task, source=CandidateSource.STUDENT,
                                    model_id=str(value["model_id"]), model_digest=model_digest,
                                    round_index=int(value["round_index"]), frozen_seeds=SEEDS,
                                    mandatory_baselines=BASELINES, allowed_baselines=frozenset(BASELINES),
                                    allowed_metrics=METRICS, frozen_resource_schedule=SCHEDULE,
                                    parent_candidate_ids=parent_ids)
    if canonical_json(proposal.payload()) != canonical_json(value):
        raise ValueError("proposal payload mismatch")
    binding_path = path.parent / "proposal_binding.json"
    binding = json.loads(binding_path.read_text()) if binding_path.is_file() else {}
    if binding != proposal_binding(proposal):
        raise ValueError("hypothesis/code binding mismatch")
    chain_path = path.parent / "proposal_chain.json"
    chain = json.loads(chain_path.read_text()) if chain_path.is_file() else {}
    if chain.get("final_digest") != proposal.digest:
        raise ValueError("proposal chain final digest mismatch")
    if len(chain.get("repairs", [])) > 2:
        raise ValueError("maximum code repairs exceeded")
    if chain.get("frozen_fields_digest") != sha256_bytes(canonical_json(frozen_view(value)).encode()):
        raise ValueError("repair changed frozen proposal fields")
    previous = chain.get("initial_digest", "")
    formed_children = []
    for item in chain.get("repairs", []):
        if item.get("child_digest"):
            if item.get("parent_digest") != previous:
                raise ValueError("repair parent hash chain mismatch")
            previous = item["child_digest"]
            formed_children.append(item)
    if previous != proposal.digest:
        raise ValueError("repair hash chain does not end at final proposal")
    if formed_children and parent_ids != (str(formed_children[-1]["parent_digest"]),):
        raise ValueError("final proposal immediate parent mismatch")
    if not formed_children and parent_ids:
        raise ValueError("unrepaired proposal must not have a parent")
    return proposal


@dataclass(frozen=True)
class Run:
    ok: bool
    result: dict[str, Any]
    error_category: str
    error_message: str
    runtime_seconds: float


def execute(code: str, problem: dict[str, Any], seed: int, budget: int, timeout: float = 10.0) -> Run:
    root_path = Path(__file__).resolve().parents[1]
    wrapper = root_path / "sdk" / "trusted_wrapper.py"
    sdk_root = wrapper.parent
    with tempfile.TemporaryDirectory(prefix="vse-sdk-gate-") as temporary:
        root = Path(temporary)
        candidate = root / "candidate.py"
        candidate.write_text(code)
        started = time.monotonic()
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(wrapper), str(candidate), str(sdk_root)],
                input=json.dumps({"problem": problem, "seed": seed, "budget": budget}),
                text=True, capture_output=True, timeout=timeout, cwd=root,
                env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": str(seed), "VSE_NETWORK_POLICY": "none"},
                check=False)
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
            return Run(bool(payload.get("ok")) and completed.returncode == 0, dict(payload.get("result", {})),
                       str(payload.get("error_category", "")), str(payload.get("error_message", "")),
                       time.monotonic() - started)
        except subprocess.TimeoutExpired as error:
            return Run(False, {}, "TimeoutExpired", str(error)[:2000], timeout)
        except (json.JSONDecodeError, IndexError, TypeError, ValueError) as error:
            return Run(False, {}, type(error).__name__, str(error)[:2000], time.monotonic() - started)


BASELINE_CODE = """def solve(problem, seed, budget):
    del seed
    return {'point': [0.0] * int(problem['dimension']), 'oracle_calls': min(1, int(budget))}
"""


def finite_or_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: finite_or_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_or_none(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def evaluate_case(case: dict[str, Any], proposal: Any) -> dict[str, Any]:
    import sdk_metrics
    audit_code(proposal.experiment_code)
    executions: list[dict[str, Any]] = []
    for seed in SEEDS:
        curve: list[dict[str, Any]] = []
        baseline_curve: list[dict[str, Any]] = []
        final: Run | None = None
        final_metrics: dict[str, float] = {"lower_residual": None, "kkt_ni_gap": None, "regret": None, "primal_feasibility": None}
        baseline_final: dict[str, float] = {}
        for budget in (int(item) for item in SCHEDULE):
            candidate = execute(proposal.experiment_code, case["problem"], seed, budget)
            baseline = execute(BASELINE_CODE, case["problem"], seed, budget)
            if not baseline.ok:
                raise RuntimeError("frozen baseline failed under trusted wrapper")
            baseline_metrics = sdk_metrics.metrics(case["problem"], baseline.result["point"])
            candidate_metrics = sdk_metrics.metrics(case["problem"], candidate.result["point"]) if candidate.ok else dict(final_metrics)
            curve.append({"budget": float(budget), "quality": -candidate_metrics["regret"] if candidate.ok else -1.0e12,
                          "runtime_seconds": candidate.runtime_seconds})
            baseline_curve.append({"budget": float(budget), "quality": -baseline_metrics["regret"],
                                   "runtime_seconds": baseline.runtime_seconds})
            final, final_metrics, baseline_final = candidate, candidate_metrics, baseline_metrics
        assert final is not None
        repeat = execute(proposal.experiment_code, case["problem"], seed, int(SCHEDULE[-1]))
        executable = final.ok and repeat.ok
        executions.append({"seed": seed, "executable": executable, "seed_reproducible": executable and final.result == repeat.result,
                           "result": final.result, "error_category": final.error_category, "error_message": final.error_message,
                           "trusted_metrics": finite_or_none(final_metrics), "baseline_metrics": baseline_final,
                           "resource_quality_curve": curve, "baseline_resource_quality_curve": baseline_curve})
    executable_records = [item for item in executions if item["executable"] and item["seed_reproducible"]]
    conditional_vds = (sum(math.exp(-float(item["trusted_metrics"]["regret"])) for item in executable_records) /
                       len(executable_records)) if executable_records else 0.0
    all_executable = len(executable_records) == len(SEEDS)
    return {"case_id": case["case_id"], "parser_valid": True, "executable": all_executable,
            "executable_seed_count": len(executable_records), "conditional_vds": finite_or_none(conditional_vds),
            "unconditional_vds": finite_or_none(conditional_vds if all_executable else 0.0),
            "hard_pass": all(float(item["trusted_metrics"]["regret"]) <= 1e-8 for item in executable_records) if all_executable else False,
            "executions": executions}


def negative_controls(proposals_root: Path, control_root: Path, cases: list[dict[str, Any]], model_digest: str) -> dict[str, Any]:
    source_case = next((case for case in cases if (proposals_root / case["case_id"] / "proposal.json").is_file()), None)
    if source_case is None:
        return {"passed": False, "controls": {}, "error": "no parser-valid proposal available"}
    source_path = proposals_root / source_case["case_id"] / "proposal.json"
    source = json.loads(source_path.read_text())
    controls: dict[str, dict[str, Any]] = {}
    controls["swapped_case"] = {**source, "task_id": next(case["case_id"] for case in cases if case["case_id"] != source_case["case_id"])}
    controls["broken_solver"] = {**source, "experiment_code": "def solve(problem, seed, budget):\n    return {'point': [float('nan')], 'oracle_calls': budget}\n"}
    controls["fabricated_metrics"] = {**source, "experiment_code": "def solve(problem, seed, budget):\n    return {'point': [0.0], 'oracle_calls': 1, 'trusted_metrics': {'regret': 0.0}}\n"}
    controls["network_import"] = {**source, "experiment_code": "import socket\ndef solve(problem, seed, budget):\n    return {'point': [0.0], 'oracle_calls': 1}\n"}
    controls["altered_seed"] = {**source, "seeds": [1103, 2207, 3301, 9999]}
    controls["target_leakage"] = {**source, "experiment_code": "def solve(problem, seed, budget):\n    raise RuntimeError('sealed target')\n"}
    controls["random_hypothesis"] = {**source, "hypothesis": {**source["hypothesis"], "claim": "Random unrelated intervention improves the objective."}}
    statuses: dict[str, dict[str, Any]] = {}
    for name, value in controls.items():
        path = control_root / "negative_controls" / name / "proposal.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        try:
            proposal = proposal_from_file(path, source_case, model_digest)
            audit_code(proposal.experiment_code)
            run = execute(proposal.experiment_code, source_case["problem"], SEEDS[0], int(SCHEDULE[-1]))
            statuses[name] = {"rejected": not run.ok, "reason": run.error_category or "control unexpectedly executable"}
        except Exception as error:
            statuses[name] = {"rejected": True, "reason": f"{type(error).__name__}: {error}"}
    return {"passed": all(item["rejected"] for item in statuses.values()), "controls": statuses}


def verify_sdk_binding(binding_path: Path, sdk_root: Path, image_digest: str) -> dict[str, Any]:
    binding = json.loads(binding_path.read_text())
    expected = {"sdk_source_digest": file_hash(sdk_root / "execution_sdk.py"),
                "sdk_card_digest": file_hash(sdk_root / "sdk_card.json"),
                "sdk_lock_digest": file_hash(sdk_root / "sdk_lock.json"),
                "wrapper_digest": file_hash(sdk_root / "trusted_wrapper.py")}
    for key, digest in expected.items():
        if binding.get(key) != digest:
            raise ValueError(f"SDK binding mismatch: {key}")
    if binding.get("execution_image_digest") != image_digest or binding.get("network_policy") != "none":
        raise ValueError("execution image/network binding mismatch")
    return binding


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--proposals-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-digest", required=True)
    parser.add_argument("--execution-container-digest", required=True)
    parser.add_argument("--sdk-binding", type=Path, required=True)
    parser.add_argument("--generation-report", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.case_manifest.read_text())
    cases = list(manifest["cases"])
    args.output_root.mkdir(parents=True, exist_ok=True)
    sdk_root = Path(__file__).resolve().parents[1] / "sdk"
    binding_error = ""
    try:
        binding = verify_sdk_binding(args.sdk_binding, sdk_root, args.execution_container_digest)
    except Exception as error:
        binding, binding_error = {}, f"{type(error).__name__}: {error}"
    results: list[dict[str, Any]] = []
    for case in cases:
        path = args.proposals_root / case["case_id"] / "proposal.json"
        try:
            proposal = proposal_from_file(path, case, args.model_digest)
            try:
                result = evaluate_case(case, proposal)
            except Exception as error:
                result = {"case_id": case["case_id"], "parser_valid": True, "executable": False,
                          "conditional_vds": 0.0, "unconditional_vds": 0.0,
                          "execution_error": f"{type(error).__name__}: {error}"}
        except Exception as error:
            result = {"case_id": case["case_id"], "parser_valid": False, "executable": False,
                      "conditional_vds": 0.0, "unconditional_vds": 0.0,
                      "error": f"{type(error).__name__}: {error}"}
        results.append(result)
    negative = negative_controls(args.proposals_root, args.output_root, cases, args.model_digest)
    generation = json.loads(args.generation_report.read_text()) if args.generation_report and args.generation_report.is_file() else {}
    report = {"schema_version": 1, "gate": GATE, "model_digest": args.model_digest,
              "execution_container_digest": args.execution_container_digest,
              "sdk_binding_digest": file_hash(args.sdk_binding) if not binding_error else "",
              "sdk_binding_error": binding_error,
              "parser_valid_rate": sum(int(item.get("parser_valid", False)) for item in results) / len(results),
              "raw_schema_complete_rate": generation.get("raw_schema_complete_rate"),
              "frozen_protocol_field_hydration": generation.get("frozen_protocol_field_hydration", []),
              "replay_of_frozen_raw_outputs": generation.get("replay_of_frozen_raw_outputs", False),
              "model_calls_during_replay": generation.get("model_calls_during_replay"),
              "initial_execution_rate": generation.get("initial_execution_rate"),
              "execution_rate_after_repair": generation.get("execution_rate_after_repair"),
              "valid_solver_result_cases": sum(int(item.get("executable", False)) for item in results),
              "valid_solver_result_minimum": 4, "all_four_seeds_required_per_case": True,
              "scientific_hard_pass_cases": sum(int(item.get("hard_pass", False)) for item in results),
              "maximum_code_repairs": 2, "hypothesis_frozen_during_repair": True,
              "negative_controls": negative, "cases": results,
              "candidate_supplied_metrics_trusted": False, "scientific_hard_pass_launch_gate": False,
              "scientific_claims_allowed": False, "eligible_for_champion": False,
              "eligible_for_training_library": False}
    report["sdk_contract_gate_passed"] = (not binding_error and report["parser_valid_rate"] == 1.0 and
                                          report["valid_solver_result_cases"] >= 4 and negative["passed"])
    report["report_digest"] = content_hash({key: value for key, value in report.items() if key != "report_digest"})
    (args.output_root / "sdk_conformance_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    receipt = {"schema_version": 1, "gate": GATE, "model_digest": args.model_digest,
               "execution_container_digest": args.execution_container_digest,
               "case_manifest_digest": file_hash(args.case_manifest), "sdk_binding_digest": report["sdk_binding_digest"],
               "report_digest": report["report_digest"], "parser_valid_rate": report["parser_valid_rate"],
               "raw_schema_complete_rate": report["raw_schema_complete_rate"],
               "frozen_protocol_field_hydration": report["frozen_protocol_field_hydration"],
               "replay_of_frozen_raw_outputs": report["replay_of_frozen_raw_outputs"],
               "model_calls_during_replay": report["model_calls_during_replay"],
               "initial_execution_rate": report["initial_execution_rate"],
               "execution_rate_after_repair": report["execution_rate_after_repair"],
               "valid_solver_result_cases": report["valid_solver_result_cases"],
               "scientific_hard_pass_cases": report["scientific_hard_pass_cases"],
               "sdk_contract_gate_passed": report["sdk_contract_gate_passed"],
               "negative_controls_passed": negative["passed"], "candidate_supplied_metrics_trusted": False,
               "scientific_hard_pass_launch_gate": False, "scientific_claims_allowed": False,
               "eligible_for_champion": False, "eligible_for_training_library": False}
    receipt["receipt_digest"] = content_hash(receipt)
    environment = {"schema_version": 1, "generation_image_digest": args.execution_container_digest,
                   "execution_container_digest": args.execution_container_digest,
                   "sdk_binding_digest": report["sdk_binding_digest"], "network_policy": "none",
                   "numpy_version": "2.1.2", "scientific_claims_allowed": False,
                   "eligible_for_champion": False, "eligible_for_training_library": False}
    (args.output_root / "sdk_conformance_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    (args.output_root / "sdk_conformance_environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["sdk_contract_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
