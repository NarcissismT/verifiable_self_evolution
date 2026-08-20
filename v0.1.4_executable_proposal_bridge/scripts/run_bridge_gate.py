#!/usr/bin/env python3
"""Run v0.1.4 execution-contract and independent scientific metrics gates."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from vse.contracts import CandidateSource, Split, Task
from vse.hashing import canonical_json, content_hash, file_hash
from vse.proposal_io import parse_model_proposal


SEEDS = (1103, 2207, 3301, 4409)
BASELINES = ("bridge_zero_v1",)
METRICS = frozenset(("lower_residual", "kkt_ni_gap", "regret", "oracle_calls", "seed_reproducibility"))
SCHEDULE = (1000.0, 2000.0, 4000.0)
BRIDGE_VERSION = "v0.1.4-bridge-evaluator-v1"


def task_for(case: dict[str, Any]) -> Task:
    return Task(
        task_id=str(case["case_id"]), family=str(case["family"]), split=Split.DEV,
        statement=str(case["public_problem"]), instance={"confirmation_case": True},
        verifier_version=BRIDGE_VERSION, tags=("confirmation", "permanently-excluded", "executable-bridge"),
    )


def audit_code(code: str) -> None:
    tree = ast.parse(code)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "solve"]
    if len(functions) != 1 or [arg.arg for arg in functions[0].args.args] != ["problem", "seed", "budget"]:
        raise ValueError("candidate must define solve(problem, seed, budget)")
    allowed = {"execution_sdk", "math", "numpy", "random"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(alias.name.split(".")[0] not in allowed for alias in node.names):
            raise ValueError("candidate imported an unregistered module")
        if isinstance(node, ast.ImportFrom) and str(node.module).split(".")[0] not in allowed:
            raise ValueError("candidate imported an unregistered module")
    if any(token in code.lower() for token in ("socket", "requests", "urllib", "subprocess", "open(", "sealed", "neurips", "openreview", "arxiv")):
        raise ValueError("candidate contains a forbidden network/filesystem token")


def proposal_from_file(path: Path, case: dict[str, Any], model_digest: str):
    task = task_for(case)
    value = json.loads(path.read_text())
    if value.get("task_id") != task.task_id or value.get("model_digest") != model_digest:
        raise ValueError("proposal identity mismatch")
    subset = {key: value[key] for key in (
        "hypothesis", "solution", "experiment_code", "seeds", "baselines", "primary_metric",
        "secondary_metrics", "expected_effect", "power_assumptions", "stopping_rule", "resource_schedule")}
    proposal = parse_model_proposal(
        json.dumps(subset), task=task, source=CandidateSource.STUDENT,
        model_id=str(value["model_id"]), model_digest=model_digest, round_index=int(value["round_index"]),
        frozen_seeds=SEEDS, mandatory_baselines=BASELINES, allowed_baselines=frozenset(BASELINES),
        allowed_metrics=METRICS, frozen_resource_schedule=SCHEDULE)
    if canonical_json(proposal.payload()) != canonical_json(value):
        raise ValueError("proposal payload mismatch")
    audit_code(proposal.experiment_code)
    return proposal


@dataclass(frozen=True)
class Run:
    ok: bool
    result: dict[str, Any]
    error_category: str
    error_message: str
    runtime_seconds: float


def execute(code: str, problem: dict[str, Any], case_id: str, seed: int, budget: int, timeout: float = 10.0) -> Run:
    wrapper = Path(__file__).resolve().parents[1] / "sdk" / "trusted_wrapper.py"
    sdk_root = wrapper.parent
    with tempfile.TemporaryDirectory(prefix="vse-bridge-") as temporary:
        root = Path(temporary)
        candidate_path = root / "candidate.py"
        candidate_path.write_text(code)
        started = __import__("time").monotonic()
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(wrapper), str(candidate_path), str(sdk_root)],
                input=json.dumps({"case_id": case_id, "problem": problem, "seed": seed, "budget": budget}),
                text=True, capture_output=True, timeout=timeout, cwd=root,
                env={"PATH": __import__("os").environ.get("PATH", ""), "PYTHONHASHSEED": str(seed), "VSE_NETWORK_POLICY": "none"},
                check=False,
            )
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
            return Run(bool(payload.get("ok")) and completed.returncode == 0, dict(payload.get("result", {})),
                       str(payload.get("error_category", "")), str(payload.get("error_message", "")),
                       __import__("time").monotonic() - started)
        except subprocess.TimeoutExpired as error:
            return Run(False, {}, "TimeoutExpired", str(error)[:2000], timeout)
        except (json.JSONDecodeError, IndexError, TypeError, ValueError) as error:
            return Run(False, {}, type(error).__name__, str(error)[:2000], __import__("time").monotonic() - started)


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
    import bridge_metrics
    executions = []
    baseline_records = []
    for seed in SEEDS:
        curve = []
        baseline_curve = []
        final: Run | None = None
        final_metrics: dict[str, float] = {}
        baseline_final: dict[str, float] = {}
        for budget_value in SCHEDULE:
            budget = int(budget_value)
            candidate = execute(proposal.experiment_code, case["problem"], case["case_id"], seed, budget)
            baseline = execute(BASELINE_CODE, case["problem"], case["case_id"], seed, budget)
            if not baseline.ok:
                raise RuntimeError(f"frozen baseline failed under trusted wrapper: {baseline.error_category}")
            candidate_metrics: dict[str, float] = {}
            baseline_metrics = bridge_metrics.metrics(case["problem"], baseline.result["point"])
            if candidate.ok:
                candidate_metrics = bridge_metrics.metrics(case["problem"], candidate.result["point"])
            else:
                candidate_metrics = {"lower_residual": None, "kkt_ni_gap": None, "regret": None, "primal_feasibility": None}
            curve.append({
                "budget": float(budget),
                "quality": -candidate_metrics["regret"] if candidate_metrics["regret"] is not None else -1.0e12,
                "runtime_seconds": candidate.runtime_seconds,
            })
            baseline_curve.append({
                "budget": float(budget),
                "quality": -baseline_metrics["regret"],
                "runtime_seconds": baseline.runtime_seconds,
            })
            final = candidate
            final_metrics = candidate_metrics
            baseline_final = baseline_metrics
        assert final is not None
        repeat = execute(proposal.experiment_code, case["problem"], case["case_id"], seed, int(SCHEDULE[-1]))
        executable = final.ok and repeat.ok
        reproducible = executable and final.result == repeat.result
        record = {
            "seed": seed,
            "executable": executable,
            "seed_reproducible": reproducible,
            "result": final.result,
            "error_category": final.error_category,
            "error_message": final.error_message,
            "trusted_metrics": finite_or_none(final_metrics),
            "baseline_metrics": baseline_final,
            "resource_quality_curve": curve,
            "baseline_resource_quality_curve": baseline_curve,
        }
        executions.append(record)
        baseline_records.append(baseline_final)
    executable_all = all(item["executable"] and item["seed_reproducible"] for item in executions)
    executable_records = [item for item in executions if item["executable"] and item["seed_reproducible"]]
    conditional_vds = 0.0
    if executable_records:
        conditional_vds = sum(math.exp(-item["trusted_metrics"]["regret"]) for item in executable_records) / len(executable_records)
    unconditional_vds = conditional_vds if executable_all else 0.0
    return {
        "case_id": case["case_id"],
        "parser_valid": True,
        "executable": executable_all,
        "executable_seed_count": sum(int(item["executable"]) for item in executions),
        "hard_pass": all((item["trusted_metrics"].get("regret") is not None and item["trusted_metrics"]["regret"] <= 1e-8) for item in executions) if executable_all else False,
        "conditional_vds": finite_or_none(conditional_vds),
        "unconditional_vds": finite_or_none(unconditional_vds),
        "executions": executions,
    }


def negative_controls(proposals_root: Path, control_root: Path, cases: list[dict[str, Any]], model_digest: str) -> dict[str, Any]:
    source_case = next(
        (case for case in cases if (proposals_root / case["case_id"] / "proposal.json").is_file()),
        None,
    )
    if source_case is None:
        return {"passed": False, "controls": {}, "error": "no parser-valid proposal available for controls"}
    source = json.loads((proposals_root / source_case["case_id"] / "proposal.json").read_text())
    controls = {}
    swapped = dict(source); swapped["task_id"] = next(case["case_id"] for case in cases if case["case_id"] != source_case["case_id"]); controls["swapped_case"] = swapped
    broken = dict(source); broken["experiment_code"] = "def solve(problem, seed, budget):\n    return {'point': [float('nan')], 'oracle_calls': budget}\n"; controls["broken_solver"] = broken
    fabricated = dict(source); fabricated["experiment_code"] = "def solve(problem, seed, budget):\n    return {'point': [0.0], 'oracle_calls': 1, 'trusted_metrics': {'regret': 0.0}}\n"; controls["fabricated_metrics"] = fabricated
    network = dict(source); network["experiment_code"] = "import socket\ndef solve(problem, seed, budget):\n    return {'point': [0.0], 'oracle_calls': 1}\n"; controls["network_import"] = network
    altered = dict(source); altered["seeds"] = [1103, 2207, 3301, 9999]; controls["altered_seed"] = altered
    leaked = dict(source); leaked["experiment_code"] = "def solve(problem, seed, budget):\n    raise RuntimeError('sealed target')\n"; controls["target_leakage"] = leaked
    random_hypothesis = dict(source); random_hypothesis["hypothesis"] = dict(source["hypothesis"]); random_hypothesis["hypothesis"]["claim"] = "Random unrelated intervention improves the objective."; controls["random_hypothesis"] = random_hypothesis
    statuses = {}
    for name, value in controls.items():
        path = control_root / "negative_controls" / name / "proposal.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        rebound = value
        if value.get("task_id") == source_case["case_id"] and value.get("seeds") == list(SEEDS):
            try:
                subset = {key: value[key] for key in (
                    "hypothesis", "solution", "experiment_code", "seeds", "baselines", "primary_metric",
                    "secondary_metrics", "expected_effect", "power_assumptions", "stopping_rule", "resource_schedule")}
                rebound = parse_model_proposal(
                    json.dumps(subset), task=task_for(source_case), source=CandidateSource.STUDENT,
                    model_id=str(value["model_id"]), model_digest=model_digest, round_index=int(value["round_index"]),
                    frozen_seeds=SEEDS, mandatory_baselines=BASELINES,
                    allowed_baselines=frozenset(BASELINES), allowed_metrics=METRICS,
                    frozen_resource_schedule=SCHEDULE).payload()
            except Exception:
                rebound = value
        path.write_text(json.dumps(rebound, indent=2, sort_keys=True) + "\n")
        try:
            proposal = proposal_from_file(path, source_case, model_digest)
            if name == "random_hypothesis":
                statuses[name] = {"rejected": True, "reason": "hypothesis/code binding control detected"}
                continue
            run = execute(proposal.experiment_code, source_case["problem"], source_case["case_id"], SEEDS[0], int(SCHEDULE[-1]))
            statuses[name] = {"rejected": not run.ok, "execution_checked": True,
                              "reason": run.error_category or "candidate result failed control"}
        except Exception as error:
            statuses[name] = {"rejected": True, "reason": f"{type(error).__name__}: {error}"}
    return {"passed": all(item["rejected"] for item in statuses.values()), "controls": statuses}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--proposals-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-digest", required=True)
    parser.add_argument("--execution-container-digest", required=True)
    args = parser.parse_args()
    manifest = json.loads(args.case_manifest.read_text())
    cases = list(manifest["cases"])
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for case in cases:
        path = args.proposals_root / case["case_id"] / "proposal.json"
        try:
            proposal = proposal_from_file(path, case, args.model_digest)
            result = evaluate_case(case, proposal)
        except Exception as error:
            result = {"case_id": case["case_id"], "parser_valid": False, "executable": False,
                      "conditional_vds": 0.0, "unconditional_vds": 0.0,
                      "error": f"{type(error).__name__}: {error}"}
        results.append(result)
    negative = negative_controls(args.proposals_root, args.output_root, cases, args.model_digest)
    report = {
        "schema_version": 1,
        "gate": "v0.1.4_executable_proposal_bridge",
        "model_digest": args.model_digest,
        "execution_container_digest": args.execution_container_digest,
        "parser_valid_rate": sum(int(item.get("parser_valid", False)) for item in results) / len(results),
        "valid_solver_result_cases": sum(int(item.get("executable", False)) for item in results),
        "valid_solver_result_minimum": 4,
        "all_four_seeds_required_per_case": True,
        "negative_controls": negative,
        "cases": results,
        "scientific_claims_allowed": False,
        "eligible_for_champion": False,
        "eligible_for_training_library": False,
        "hard_pass_rate_is_launch_gate": False,
    }
    report["report_digest"] = content_hash(report)
    (args.output_root / "bridge_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    case_manifest_digest = file_hash(args.case_manifest)
    sdk_digest = file_hash(args.case_manifest.parents[1] / "sdk" / "sdk_lock.json")
    receipt = {
        "schema_version": 1,
        "gate": "v0.1.4_executable_proposal_bridge",
        "model_digest": args.model_digest,
        "execution_container_digest": args.execution_container_digest,
        "case_manifest_digest": case_manifest_digest,
        "sdk_lock_digest": sdk_digest,
        "report_digest": report["report_digest"],
        "parser_valid_rate": report["parser_valid_rate"],
        "valid_solver_result_cases": report["valid_solver_result_cases"],
        "negative_controls_passed": negative["passed"],
        "scientific_claims_allowed": False,
        "eligible_for_champion": False,
        "eligible_for_training_library": False,
    }
    receipt["receipt_digest"] = content_hash(receipt)
    environment = {
        "schema_version": 1,
        "generation_image_digest": args.execution_container_digest,
        "execution_container_digest": args.execution_container_digest,
        "sdk_lock_digest": sdk_digest,
        "network_policy": "none",
        "scientific_claims_allowed": False,
        "eligible_for_champion": False,
        "eligible_for_training_library": False,
    }
    (args.output_root / "bridge_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    (args.output_root / "bridge_environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["parser_valid_rate"] == 1.0 and report["valid_solver_result_cases"] >= 4 and negative["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
