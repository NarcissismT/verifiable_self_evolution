#!/usr/bin/env python3
"""Generate proposals and perform public, code-only SDK repairs.

The initial model call creates the complete ExperimentProposal.  Once that
proposal parses, all repair calls can replace only ``experiment_code``;
hypothesis, design, seeds, baselines, metrics, and schedule are frozen and a
parent digest is recorded for every child.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vse.contracts import CandidateSource, Split, Task
from vse.proposal_io import parse_model_proposal

from sdk_api_linter import lint_code


SEEDS = (1103, 2207, 3301, 4409)
BASELINES = ("sdk_zero_v1",)
METRICS = frozenset(("lower_residual", "kkt_ni_gap", "regret", "oracle_calls", "seed_reproducibility"))
SCHEDULE = (1000.0, 2000.0, 4000.0)
EVALUATOR_VERSION = "v0.1.5-sdk-conformance-evaluator-v1"
MAX_REPAIRS = 2
PROPOSAL_KEYS = {
    "hypothesis", "solution", "experiment_code", "seeds", "baselines", "primary_metric",
    "secondary_metrics", "expected_effect", "power_assumptions", "stopping_rule", "resource_schedule",
}
FROZEN_PROTOCOL_FIELDS: dict[str, Any] = {
    "seeds": list(SEEDS),
    "baselines": list(BASELINES),
    "primary_metric": "regret",
    "secondary_metrics": ["lower_residual", "kkt_ni_gap", "oracle_calls", "seed_reproducibility"],
    "expected_effect": {"direction": "lower", "minimum_delta": 0.0},
    "power_assumptions": {"alpha": 0.05, "target_power": 0.8, "unit": "seed"},
    "stopping_rule": "return within the supplied budget",
    "resource_schedule": list(SCHEDULE),
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def task_for(case: dict[str, Any]) -> Task:
    return Task(task_id=str(case["case_id"]), family=str(case["family"]), split=Split.DEV,
                statement=str(case["public_problem"]), instance={"confirmation_case": True},
                verifier_version=EVALUATOR_VERSION,
                tags=("confirmation", "permanently-excluded", "sdk-conformance"))


def extract_object(text: str, required: tuple[str, ...]) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for start, marker in enumerate(text):
        if marker != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and all(key in value for key in required):
            return value
    raise ValueError("no complete JSON object with required fields")


def normalize_code_encoding(code: str) -> tuple[str, bool]:
    """Normalize one common JSON double-escaping error before Python parsing.

    This is a representation-only compiler pass. It never inserts identifiers,
    field values, algorithms, or evaluator information.
    """
    normalized = code
    if "\n" not in normalized and "\\n" in normalized:
        normalized = normalized.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "    ")
    return normalized, normalized != code


def hydrate_frozen_protocol_fields(value: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Fill only system-owned constants; never synthesize semantic model fields."""
    hydrated = dict(value)
    inserted: list[str] = []
    for key, expected in FROZEN_PROTOCOL_FIELDS.items():
        if key not in hydrated:
            hydrated[key] = json.loads(json.dumps(expected))
            inserted.append(key)
    return hydrated, inserted


def audit_code(code: str) -> list[dict[str, Any]]:
    try:
        ast.parse(code)
    except SyntaxError as error:
        return [{"code": "syntax_error", "message": error.msg, "line": int(error.lineno or 0)}]
    lowered = code.lower()
    if any(token in lowered for token in ("socket", "requests", "urllib", "subprocess", "open(",
                                         "sealed", "neurips", "openreview", "arxiv", "/etc/", "target_digest")):
        return [{"code": "forbidden_token", "message": "candidate contains a forbidden network/filesystem/target token", "line": 0}]
    return [item.payload() for item in lint_code(code)]


def public_runtime_check(code: str, case: dict[str, Any]) -> None:
    """Run only the public wrapper; no hidden quality or evaluator data is exposed."""
    issues = audit_code(code)
    if issues:
        first = issues[0]
        raise ValueError(f"public_linter_error:{first['code']}:{first['message']}")
    bridge_root = Path(__file__).resolve().parents[1]
    wrapper = bridge_root / "sdk" / "trusted_wrapper.py"
    sdk_root = bridge_root / "sdk"
    with tempfile.TemporaryDirectory(prefix="vse-sdk-public-check-") as temporary:
        root = Path(temporary)
        candidate = root / "candidate.py"
        candidate.write_text(code)
        completed = subprocess.run(
            [sys.executable, "-I", str(wrapper), str(candidate), str(sdk_root)],
            input=json.dumps({"problem": case["problem"], "seed": SEEDS[0], "budget": int(SCHEDULE[0])}),
            text=True, capture_output=True, timeout=10.0, cwd=root,
            env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": str(SEEDS[0]), "VSE_NETWORK_POLICY": "none"},
            check=False,
        )
        try:
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as error:
            raise ValueError(f"public_runtime_invalid_wrapper_output:{error}") from error
        if completed.returncode != 0 or not payload.get("ok"):
            raise ValueError("public_runtime_error:%s:%s" % (
                payload.get("error_category", "RuntimeError"), str(payload.get("error_message", "candidate failed"))[:500]))


def interface_example_code() -> str:
    return ("from execution_sdk import project_box\n\n"
            "def solve(problem, seed, budget):\n"
            "    del seed, budget\n"
            "    point = project_box([0.0] * int(problem['dimension']), problem['bounds'])\n"
            "    return {'point': point, 'oracle_calls': 0}")


def prompt_for(case: dict[str, Any], card: dict[str, str]) -> str:
    example = interface_example_code()
    schema = {
        "hypothesis": {"claim": "string", "mechanism": "string", "assumptions": ["string"],
                       "alternative_explanations": ["string"], "null_hypothesis": "string",
                       "predicted_failure_mode": "string", "discriminating_observation": "string"},
        "solution": {"api": "solve_v1", "algorithm_family": "string"},
        "experiment_code": example,
        "seeds": list(SEEDS), "baselines": list(BASELINES), "primary_metric": "regret",
        "secondary_metrics": ["lower_residual", "kkt_ni_gap", "oracle_calls", "seed_reproducibility"],
        "expected_effect": {"direction": "lower", "minimum_delta": 0.0},
        "power_assumptions": {"alpha": 0.05, "target_power": 0.8, "unit": "seed"},
        "stopping_rule": "return within the supplied budget", "resource_schedule": list(SCHEDULE),
    }
    schema_text = json.dumps(schema, ensure_ascii=True, sort_keys=False, separators=(",", ":"))
    return f"""Return exactly one JSON object and no markdown.
Use every field and type in this exact schema. Its experiment_code value is executable target-neutral
interface code, not placeholder prose: {schema_text}
The experiment_code is normal multi-line Python encoded as a JSON string. It must define exactly
solve(problem, seed, budget) and return exactly {{"point": list[float], "oracle_calls": int}}.
The trusted wrapper owns stdin/stdout JSON, initializes both random and numpy seeds, and performs
finite/shape/schema checks. The only public SDK imports a student should need are:
{json.dumps(card, sort_keys=True)}
Pure interface example (it is deliberately a zero-point example and is not a solver fallback):
{json.dumps(example)}
Allowed imports are math, random, numpy as np, and execution_sdk. Do not call initialize_seed or
finite_vector; those checks are wrapper-owned. Do not access files, network, subprocesses, hidden
targets, candidate metrics, or evaluator internals. Use only public problem fields and return a
finite point. The frozen seeds, baselines, metrics, and resource schedule must be copied exactly.
Before ending the JSON object, verify that both solution and stopping_rule are present. Never emit
literal placeholder phrases such as 'normal multi-line source' or 'replacement source'.
CASE_ID: {case['case_id']}
PUBLIC_MATHEMATICAL_PROBLEM: {case['public_problem']}
PUBLIC_INSTANCE_SCHEMA: {canonical(case['problem'])}
JSON:"""


def repair_prompt(case: dict[str, Any], card: dict[str, str], parent: Any, code: str, error: str) -> str:
    repair_schema = {"experiment_code": interface_example_code(),
                     "repair_explanation": "Explain only the public syntax/API/schema correction.",
                     "parent_digest": parent.digest}
    return f"""Return exactly one JSON object and no markdown with exactly these keys:
{json.dumps(repair_schema, ensure_ascii=True, sort_keys=False, indent=2)}
The experiment_code shown above is executable target-neutral interface code, not placeholder prose.
The experiment_code value must be Python source beginning with an import or def, never another JSON
object encoded inside the string.
This is a CODE-ONLY repair. Keep the parent hypothesis, solution, seeds, baselines, metrics,
power assumptions, stopping rule, and resource schedule unchanged. Do not rewrite them and do not
use hidden quality values. The frozen SDK card is {canonical(card)}. Students should import only
Budget/project_box from execution_sdk; the wrapper initializes seeds and validates point/schema.
The original code was:
{code}
The public compiler/runtime error was: {error}
Repair only syntax, name, SDK-API, or public result-schema mistakes. The function must remain
solve(problem, seed, budget) and return {{'point': list[float], 'oracle_calls': int}}.
CASE_ID: {case['case_id']}
JSON:"""


def generate_text(model: Any, tokenizer: Any, prompt: str, device: Any, max_new_tokens: int) -> str:
    messages = [{"role": "system", "content": "Emit one strict JSON object."}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = {key: value.to(device) for key, value in tokenizer(text, return_tensors="pt").items()}
    with torch.no_grad():
        generated = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                                   num_beams=1, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(generated[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True)


def proposal_with_code(value: dict[str, Any], task: Task, model_digest: str, code: str, parent_ids: tuple[str, ...]):
    payload = {key: value[key] for key in (
        "hypothesis", "solution", "experiment_code", "seeds", "baselines", "primary_metric",
        "secondary_metrics", "expected_effect", "power_assumptions", "stopping_rule", "resource_schedule")}
    payload["experiment_code"] = code
    return parse_model_proposal(
        json.dumps(payload), task=task, source=CandidateSource.STUDENT,
        model_id="qwen2.5-7b-instruct", model_digest=model_digest, round_index=0,
        frozen_seeds=SEEDS, mandatory_baselines=BASELINES,
        allowed_baselines=frozenset(BASELINES), allowed_metrics=METRICS,
        frozen_resource_schedule=SCHEDULE, parent_candidate_ids=parent_ids)


def frozen_view(proposal: Any) -> dict[str, Any]:
    payload = proposal.payload()
    return {key: payload[key] for key in payload if key not in {"candidate_id", "experiment_code", "parent_candidate_ids"}}


def proposal_binding(proposal: Any) -> dict[str, str]:
    hypothesis_digest = hashlib.sha256(canonical(proposal.payload()["hypothesis"]).encode()).hexdigest()
    code_digest = hashlib.sha256(proposal.experiment_code.encode()).hexdigest()
    return {"hypothesis_digest": hypothesis_digest, "code_digest": code_digest,
            "binding_digest": hashlib.sha256(canonical({"hypothesis_digest": hypothesis_digest,
                                                         "code_digest": code_digest}).encode()).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-digest", required=True)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--sdk-card", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()
    if len(args.model_digest) != 64 or any(char not in "0123456789abcdef" for char in args.model_digest):
        raise SystemExit("model digest must be a lowercase sha256 hex digest")
    manifest = json.loads(args.case_manifest.read_text())
    card = json.loads(args.sdk_card.read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True, torch_dtype=torch.float16,
                                                  device_map="auto", low_cpu_mem_usage=True)
    model.eval()
    device = next(model.parameters()).device
    records: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        root = args.output_root / str(case["case_id"])
        root.mkdir(parents=True, exist_ok=True)
        task = task_for(case)
        initial_raw = generate_text(model, tokenizer, prompt_for(case, card), device, args.max_new_tokens)
        (root / "raw_initial_output.txt").write_text(initial_raw)
        parent = None
        proposal = None
        parser_error = ""
        try:
            raw_value = extract_object(initial_raw, ("hypothesis", "experiment_code"))
            raw_schema_complete = set(raw_value) == PROPOSAL_KEYS
            value, hydrated_fields = hydrate_frozen_protocol_fields(raw_value)
            raw_code = str(value["experiment_code"])
            code, initial_code_normalized = normalize_code_encoding(raw_code)
            proposal = proposal_with_code(value, task, args.model_digest, code, ())
            parent = proposal
            (root / "initial_proposal.json").write_text(json.dumps(proposal.payload(), indent=2, sort_keys=True) + "\n")
        except (ValueError, TypeError, KeyError, SyntaxError) as error:
            parser_error = f"{type(error).__name__}: {error}"[:1000]
            initial_code_normalized = False
            raw_schema_complete = False
            hydrated_fields = []

        initial_execution_valid = False
        initial_error = parser_error
        if proposal is not None:
            try:
                public_runtime_check(proposal.experiment_code, case)
                initial_execution_valid = True
                initial_error = ""
            except (ValueError, TypeError, SyntaxError, subprocess.SubprocessError) as error:
                initial_error = str(error)[:1000]

        repairs: list[dict[str, Any]] = []
        current = proposal
        current_error = initial_error
        for repair_index in range(MAX_REPAIRS):
            if current is None or not current_error:
                break
            repair_raw = generate_text(model, tokenizer, repair_prompt(case, card, current, current.experiment_code, current_error), device, args.max_new_tokens)
            (root / f"raw_repair_{repair_index + 1}.txt").write_text(repair_raw)
            repair_record: dict[str, Any] = {"attempt": repair_index + 1, "parent_digest": current.digest,
                                             "raw_digest": hashlib.sha256(repair_raw.encode()).hexdigest()}
            try:
                repair = extract_object(repair_raw, ("experiment_code", "repair_explanation", "parent_digest"))
                if set(repair) != {"experiment_code", "repair_explanation", "parent_digest"}:
                    raise ValueError("code-only repair schema mismatch")
                if str(repair["parent_digest"]) != current.digest:
                    raise ValueError("repair parent digest mismatch")
                replacement, code_normalized = normalize_code_encoding(str(repair["experiment_code"]))
                child = proposal_with_code(json.loads(json.dumps(current.payload(), sort_keys=True)), task,
                                           args.model_digest, replacement, (current.digest,))
                if frozen_view(child) != frozen_view(current):
                    raise ValueError("repair changed a frozen proposal field")
                repair_record["child_digest"] = child.digest
                repair_record["code_digest"] = hashlib.sha256(replacement.encode()).hexdigest()
                repair_record["code_encoding_normalized"] = code_normalized
                current = child
                try:
                    public_runtime_check(replacement, case)
                    current_error = ""
                    repair_record["public_runtime_valid"] = True
                except (ValueError, TypeError, KeyError, SyntaxError, subprocess.SubprocessError) as error:
                    current_error = str(error)[:1000]
                    repair_record["public_runtime_valid"] = False
                    repair_record["error"] = current_error
            except (ValueError, TypeError, KeyError, SyntaxError, subprocess.SubprocessError) as error:
                current_error = str(error)[:1000]
                repair_record["public_runtime_valid"] = False
                repair_record["error"] = current_error
            repairs.append(repair_record)

        final = current
        if final is not None:
            (root / "proposal.json").write_text(json.dumps(final.payload(), indent=2, sort_keys=True) + "\n")
            (root / "candidate.py").write_text(final.experiment_code)
        chain = {"schema_version": 1, "initial_digest": parent.digest if parent else "",
                 "final_digest": final.digest if final else "", "repairs": repairs,
                 "initial_code_encoding_normalized": initial_code_normalized,
                 "frozen_fields_digest": hashlib.sha256(canonical(frozen_view(parent)).encode()).hexdigest() if parent else "",
                 "frozen_fields": ["hypothesis", "solution", "seeds", "baselines", "primary_metric",
                                   "secondary_metrics", "expected_effect", "power_assumptions",
                                   "stopping_rule", "resource_schedule"]}
        (root / "proposal_chain.json").write_text(json.dumps(chain, indent=2, sort_keys=True) + "\n")
        if final is not None:
            (root / "proposal_binding.json").write_text(json.dumps(proposal_binding(final), indent=2, sort_keys=True) + "\n")
        final_execution_valid = final is not None and not current_error
        records.append({"case_id": case["case_id"], "parser_valid": proposal is not None,
                        "raw_schema_complete": raw_schema_complete,
                        "hydrated_frozen_protocol_fields": hydrated_fields,
                        "initial_execution_valid": initial_execution_valid,
                        "public_runtime_valid": final_execution_valid,
                        "initial_error": initial_error, "final_error": current_error,
                        "repairs_used": len(repairs), "proposal_digest": final.digest if final else ""})
        print(json.dumps(records[-1], sort_keys=True))
    report = {"schema_version": 1, "gate": "v0.1.5_sdk_conformance_confirmation",
              "maximum_code_repairs": MAX_REPAIRS, "cases": records,
              "parser_valid_rate": sum(int(item["parser_valid"]) for item in records) / len(records),
              "raw_schema_complete_rate": sum(int(item["raw_schema_complete"]) for item in records) / len(records),
              "frozen_protocol_field_hydration": sorted(FROZEN_PROTOCOL_FIELDS),
              "initial_execution_rate": sum(int(item["initial_execution_valid"]) for item in records) / len(records),
              "execution_rate_after_repair": sum(int(item["public_runtime_valid"]) for item in records) / len(records),
              "hypothesis_frozen_during_repair": True,
              "scientific_hard_pass_launch_gate": False}
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "generation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["parser_valid_rate"] == 1.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
