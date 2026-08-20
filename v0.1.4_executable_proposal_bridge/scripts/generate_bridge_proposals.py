#!/usr/bin/env python3
"""Generate strict multi-line solve() proposals from a local Qwen checkpoint."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vse.contracts import CandidateSource, Split, Task
from vse.proposal_io import parse_model_proposal


SEEDS = (1103, 2207, 3301, 4409)
BASELINES = ("bridge_zero_v1",)
METRICS = frozenset(("lower_residual", "kkt_ni_gap", "regret", "oracle_calls", "seed_reproducibility"))
SCHEDULE = (1000.0, 2000.0, 4000.0)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def task_for(case: dict[str, Any]) -> Task:
    return Task(
        task_id=str(case["case_id"]), family=str(case["family"]), split=Split.DEV,
        statement=str(case["public_problem"]), instance={"confirmation_case": True},
        verifier_version="v0.1.4-bridge-evaluator-v1",
        tags=("confirmation", "permanently-excluded", "executable-bridge"),
    )


def extract_json(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for start, marker in enumerate(text):
        if marker != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "experiment_code" in value and "hypothesis" in value:
            return value
    raise ValueError("no complete proposal JSON object")


def audit_code(code: str) -> None:
    tree = ast.parse(code)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "solve"]
    if len(functions) != 1 or [arg.arg for arg in functions[0].args.args] != ["problem", "seed", "budget"]:
        raise ValueError("code must define exactly solve(problem, seed, budget)")
    allowed = {"execution_sdk", "math", "numpy", "random"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] not in allowed for alias in node.names):
                raise ValueError("candidate imported an unregistered module")
        if isinstance(node, ast.ImportFrom) and str(node.module).split(".")[0] not in allowed:
            raise ValueError("candidate imported an unregistered module")
    lowered = code.lower()
    if any(token in lowered for token in ("socket", "requests", "urllib", "subprocess", "open(", "sealed", "neurips", "openreview", "arxiv")):
        raise ValueError("candidate contains network, filesystem, or target token")


def public_runtime_check(code: str, case: dict[str, Any]) -> None:
    """Expose only public syntax/runtime/schema failures, never evaluator metrics."""
    bridge_root = Path(__file__).resolve().parents[1]
    wrapper = bridge_root / "sdk" / "trusted_wrapper.py"
    sdk_root = bridge_root / "sdk"
    with tempfile.TemporaryDirectory(prefix="vse-bridge-public-check-") as temporary:
        candidate = Path(temporary) / "candidate.py"
        candidate.write_text(code)
        completed = subprocess.run(
            [sys.executable, "-I", str(wrapper), str(candidate), str(sdk_root)],
            input=json.dumps({"case_id": case["case_id"], "problem": case["problem"], "seed": SEEDS[0], "budget": int(SCHEDULE[0])}),
            text=True, capture_output=True, timeout=10.0, cwd=temporary,
            env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": str(SEEDS[0]), "VSE_NETWORK_POLICY": "none"},
            check=False,
        )
        try:
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as error:
            raise ValueError(f"public_runtime_invalid_wrapper_output:{error}") from error
        if completed.returncode != 0 or not payload.get("ok"):
            category = str(payload.get("error_category", "RuntimeError"))
            message = str(payload.get("error_message", "candidate failed public runtime check"))
            raise ValueError(f"public_runtime_error:{category}:{message[:500]}")


def prompt_for(case: dict[str, Any]) -> str:
    schema = {
        "hypothesis": {"claim": "string", "mechanism": "string", "assumptions": ["string"],
                       "alternative_explanations": ["string"], "null_hypothesis": "string",
                       "predicted_failure_mode": "string", "discriminating_observation": "string"},
        "solution": {"api": "solve_v1", "algorithm_family": "string"},
        "experiment_code": "normal multi-line Python source encoded as a JSON string",
        "seeds": list(SEEDS), "baselines": list(BASELINES), "primary_metric": "regret",
        "secondary_metrics": ["lower_residual", "kkt_ni_gap", "oracle_calls", "seed_reproducibility"],
        "expected_effect": {"direction": "lower", "minimum_delta": 0.0},
        "power_assumptions": {"alpha": 0.05, "target_power": 0.8, "unit": "seed"},
        "stopping_rule": "return within the supplied budget", "resource_schedule": list(SCHEDULE),
    }
    return f"""Return exactly one JSON object and no markdown.
Use every field and type in this exact schema: {canonical(schema)}
The experiment_code may be normal multi-line Python represented with JSON newline escapes;
one physical line using semicolons is also valid and often safer for JSON.
To avoid JSON escaping mistakes, do not use triple-quoted strings and use single quotes for
Python string literals inside experiment_code. A valid compact example is:
"def solve(problem, seed, budget):\\n    return {{'point': [0.0] * problem['dimension'], 'oracle_calls': 0}}"
It must define exactly solve(problem, seed, budget) and return exactly
{{"point": list[float], "oracle_calls": int}}. A trusted wrapper handles JSON, seed setup, validation, and errors.
Allowed imports are math, random, numpy as np, and execution_sdk. SciPy, filesystem, network, subprocess,
candidate metrics, and hidden evaluator access are forbidden. The SDK is target-neutral and exposes
Budget, finite_vector, initialize_seed, and project_box. Do not write stdin/stdout or wrapper code.
Use public fields kind, dimension, target, bounds, and optional offset. Produce deterministic output.
CASE_ID: {case['case_id']}
PUBLIC_PROBLEM: {case['public_problem']}
PUBLIC_INSTANCE_SCHEMA: {canonical(case['problem'])}
JSON:"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-digest", required=True)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.model_digest):
        raise SystemExit("invalid model digest")
    manifest = json.loads(args.case_manifest.read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, local_files_only=True, torch_dtype=torch.float16,
        device_map="auto", low_cpu_mem_usage=True)
    model.eval()
    device = next(model.parameters()).device
    generation_results: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        task = task_for(case)
        prompt = prompt_for(case)
        last_public_error = ""
        previous_output = ""
        proposal = None
        parser_valid_proposal = None
        parser_valid_raw = ""
        for attempt in range(3):
            repair = "" if not last_public_error else (
                "\nThe previous output failed a public syntax/runtime/schema check: "
                + last_public_error + ". Repair it without using hidden metrics or evaluator details. "
                "Return all required fields, including solution and stopping_rule, and JSON-escape every "
                "quote/newline inside experiment_code. The prior output was:\n" + previous_output[:12000])
            messages = [{"role": "system", "content": "Emit one strict proposal JSON object."},
                        {"role": "user", "content": prompt + repair}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "{"
            encoded = {key: value.to(device) for key, value in tokenizer(text, return_tensors="pt").items()}
            with torch.no_grad():
                generated = model.generate(**encoded, max_new_tokens=args.max_new_tokens, do_sample=False,
                                           num_beams=1, pad_token_id=tokenizer.eos_token_id)
            continuation = tokenizer.decode(generated[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True)
            raw = continuation if continuation.lstrip().startswith("{") else "{" + continuation
            previous_output = raw
            case_root = args.output_root / str(case["case_id"])
            case_root.mkdir(parents=True, exist_ok=True)
            (case_root / f"raw_attempt_{attempt + 1}.txt").write_text(raw)
            try:
                value = extract_json(raw)
                audit_code(str(value["experiment_code"]))
                parsed = parse_model_proposal(
                    json.dumps(value), task=task, source=CandidateSource.STUDENT,
                    model_id="qwen2.5-7b-instruct", model_digest=args.model_digest, round_index=0,
                    frozen_seeds=SEEDS, mandatory_baselines=BASELINES,
                    allowed_baselines=frozenset(BASELINES), allowed_metrics=METRICS,
                    frozen_resource_schedule=SCHEDULE)
                parser_valid_proposal = parsed
                parser_valid_raw = raw
                public_runtime_check(parsed.experiment_code, case)
                proposal = parsed
                (case_root / "raw_model_output.txt").write_text(raw)
                (case_root / "proposal.json").write_text(json.dumps(proposal.payload(), indent=2, sort_keys=True) + "\n")
                (case_root / "candidate.py").write_text(proposal.experiment_code)
                break
            except (SyntaxError, ValueError, TypeError, KeyError) as error:
                last_public_error = f"{type(error).__name__}: {error}"[:1000]
        if proposal is None:
            if parser_valid_proposal is not None:
                case_root = args.output_root / str(case["case_id"])
                (case_root / "raw_model_output.txt").write_text(parser_valid_raw)
                (case_root / "proposal.json").write_text(json.dumps(parser_valid_proposal.payload(), indent=2, sort_keys=True) + "\n")
                (case_root / "candidate.py").write_text(parser_valid_proposal.experiment_code)
                result = {
                    "case_id": case["case_id"], "parser_valid": True,
                    "public_runtime_valid": False, "proposal_digest": parser_valid_proposal.digest,
                    "public_error": last_public_error,
                }
            else:
                result = {"case_id": case["case_id"], "parser_valid": False,
                          "public_runtime_valid": False, "error": last_public_error}
            generation_results.append(result)
            print(json.dumps(result, sort_keys=True))
            continue
        result = {"case_id": case["case_id"], "parser_valid": True,
                  "public_runtime_valid": True, "proposal_digest": proposal.digest}
        generation_results.append(result)
        print(json.dumps(result, sort_keys=True))
    report = {
        "schema_version": 1,
        "gate": "v0.1.4_executable_proposal_bridge",
        "initial_attempts_per_case": 1,
        "maximum_repairs_per_case": 2,
        "cases": generation_results,
        "parser_valid_rate": sum(int(item["parser_valid"]) for item in generation_results) / len(generation_results),
        "public_runtime_valid_rate": sum(int(item["public_runtime_valid"]) for item in generation_results) / len(generation_results),
    }
    (args.output_root / "generation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["parser_valid_rate"] == 1.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
