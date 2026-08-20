#!/usr/bin/env python3
"""Generate complete ExperimentProposal JSON objects from a local Qwen model."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vse.contracts import CandidateSource, Split, Task
from vse.proposal_io import parse_model_proposal


CASES = (
    "realpilot_flatness_penalty",
    "realpilot_nonconvex_simple",
    "realpilot_linear_coupling",
)
SEEDS = (1031, 2063, 4099, 8191)
BASELINES = (
    "precutoff_penalty_synthesis_1",
    "precutoff_penalty_synthesis_2",
    "precutoff_penalty_synthesis_3",
)
METRICS = (
    "lower_residual",
    "upper_regret",
    "primal_feasibility",
    "oracle_calls",
    "seed_reproducibility",
)
RESOURCE_SCHEDULE = (1000.0, 2000.0, 4000.0)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_json(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    candidates = [text]
    candidates.extend(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S))
    for start, marker in enumerate(text):
        if marker != "{":
            continue
        try:
            _, consumed = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        candidates.append(text[start : start + consumed])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "experiment_code" in value and "hypothesis" in value:
            return value
    raise ValueError("Qwen output did not contain one complete ExperimentProposal JSON object")


def task_for(case_id: str, capsule: dict[str, Any]) -> Task:
    return Task(
        task_id=case_id,
        family=str(capsule["field"]),
        split=Split.DEV,
        statement=str(capsule["public_problem"]),
        instance={"case_id": case_id, "capsule_digest": hashlib.sha256(
            canonical_json(capsule).encode("utf-8")
        ).hexdigest()},
        verifier_version="causal-real-pilot-evaluator-v1",
        tags=("real-pilot", "engineering-gate", case_id),
    )


def prompt_for(case_id: str, capsule: dict[str, Any], task: Task) -> str:
    context = capsule["research_context"]
    schema = {
        "hypothesis": {
            "claim": "string",
            "mechanism": "string",
            "assumptions": ["string"],
            "alternative_explanations": ["string"],
            "null_hypothesis": "string",
            "predicted_failure_mode": "string",
            "discriminating_observation": "string",
        },
        "solution": {"api": "stdin_json_context_v1", "algorithm_family": "string"},
        "experiment_code": "python source string",
        "seeds": list(SEEDS),
        "baselines": list(BASELINES),
        "primary_metric": "upper_regret",
        "secondary_metrics": ["lower_residual", "primal_feasibility", "oracle_calls", "seed_reproducibility"],
        "expected_effect": {"direction": "lower", "minimum_delta": 0.0},
        "power_assumptions": {"alpha": 0.05, "target_power": 0.8, "unit": "seed"},
        "stopping_rule": "string",
        "resource_schedule": list(RESOURCE_SCHEDULE),
    }
    return f"""You are a student researcher in a causal proposal gate. Use only the public pre-cutoff capsule below.
Never name, infer, or reproduce the hidden target paper, authors, venue, later results, or target code.
Return exactly one JSON object matching the schema below and no markdown, code fence, or second object.
Before emitting, check that every required top-level key is present, especially expected_effect.
The experiment_code is the actual student implementation and will be executed verbatim by a network-disabled runner.
It must be a complete Python program (not a function or pseudocode), use Python standard library only,
read one JSON object from stdin with json.load(sys.stdin), and print one JSON object with json.dumps(...).
The experiment_code JSON value must contain no newline character at all: write one physical line of Python with semicolons.
Never place a literal newline inside the JSON string and never use a multi-line code block.
Use only imports json, math, sys, and random; never numpy, scipy, torch, or external packages.
At runtime assign payload=json.load(sys.stdin), then ctx=payload['context'], case_id=ctx['case_id'],
and problem=ctx['problem']; do not treat the whole payload as the problem. Output keys must be
point (a finite list of exactly 2 floats for the first two cases or exactly 4 for linear_coupling),
oracle_calls (integer), and unit_tests (object with passed and total).
These output keys belong inside the JSON printed by experiment_code, never at proposal top level.
Use case_id branches and only public problem fields. For flat_penalty, enforce y*y near a*x and
choose the sign closest to upper_target. For nonconvex_simple, use sqrt(a) roots and choose signs
near upper_target. For linear_coupling, derive y=lower_matrix*x+lower_offset and use a deterministic
finite grid over the public box while checking constraint_matrix and constraint_bound. These are
algorithm requirements, not a hidden implementation; write the complete student program yourself.
The program must be deterministic for every frozen seed and must not return a random point or echo input.
Do not claim trusted metrics in the output: an independent evaluator computes all metrics from point and hidden problem.
Implement a real first-order/finite-search procedure appropriate to the public problem, not a fixed adapter copied from elsewhere.

CASE_ID: {case_id}
TASK_DIGEST: {task.digest}
SCHEMA:
{canonical_json(schema)}
PUBLIC_PROBLEM:
{capsule['public_problem']}
PUBLIC_CONTEXT:
{canonical_json({'known_results': context.get('known_results', []), 'known_failures_and_conflicts': context.get('known_failures_and_conflicts', []), 'candidate_metrics': context.get('candidate_metrics', [])})}
FINAL JSON (start immediately):
"""


def validate_code(code: str) -> None:
    tree = ast.parse(code)
    if not any(isinstance(node, ast.Import) and any(alias.name == "json" for alias in node.names) for node in tree.body):
        raise ValueError("experiment code must import json")
    allowed_imports = {"json", "math", "sys", "random"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(alias.name.split(".")[0] not in allowed_imports for alias in node.names):
            raise ValueError("experiment code may only import json, math, sys, random")
        if isinstance(node, ast.ImportFrom) and str(node.module).split(".")[0] not in allowed_imports:
            raise ValueError("experiment code may only import json, math, sys, random")
    forbidden = ("openreview", "arxiv", "requests", "urllib", "socket", "subprocess", "target.pdf", "scipy", "numpy")
    lowered = code.lower()
    if any(item in lowered for item in forbidden):
        raise ValueError("experiment code contains forbidden network/target token")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-digest", required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--cpu-threads", type=int, default=32)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--attempts", type=int, default=5)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.model_digest):
        raise SystemExit("model digest must be lowercase SHA-256")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(min(4, args.cpu_threads))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=getattr(torch, args.dtype),
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    model_device = next(model.parameters()).device
    args.output_root.mkdir(parents=True, exist_ok=True)
    for case_id in CASES:
        case_root = args.output_root / case_id
        case_root.mkdir(parents=True, exist_ok=True)
        capsule = json.loads((args.public_root / case_id / "capsule.json").read_text(encoding="utf-8"))
        task = task_for(case_id, capsule)
        prompt = prompt_for(case_id, capsule, task)
        proposal = None
        last_error = ""
        previous_output = ""
        for attempt in range(1, args.attempts + 1):
            attempt_prompt = prompt
            if previous_output:
                # A short repair prompt prevents the model from continuing a code-only
                # completion after the first malformed emission. The code remains
                # model-generated and is still revalidated by the strict parser.
                attempt_prompt = f"""The previous emission was code-only or invalid. Produce a single valid JSON object now.
Do not emit Python outside the JSON object, markdown, a code fence, or a second object.
The object must have exactly these top-level keys:
{canonical_json(sorted({"hypothesis", "solution", "experiment_code", "seeds", "baselines", "primary_metric", "secondary_metrics", "expected_effect", "power_assumptions", "stopping_rule", "resource_schedule"}))}
The hypothesis object must have exactly:
{canonical_json(sorted({"claim", "mechanism", "assumptions", "alternative_explanations", "null_hypothesis", "predicted_failure_mode", "discriminating_observation"}))}
Use frozen seeds {list(SEEDS)}, baselines {list(BASELINES)}, metrics {list(METRICS)}, and resource schedule {list(RESOURCE_SCHEDULE)}.
The field types are strict: solution is an object such as {{"api":"stdin_json_context_v1","algorithm_family":"first_order_bilevel"}};
assumptions, alternative_explanations, and secondary_metrics are arrays of strings;
expected_effect is an object such as {{"direction":"lower","minimum_delta":0.0}};
power_assumptions is an object such as {{"alpha":0.05,"target_power":0.8,"unit":"seed"}};
stopping_rule is a string; seeds and resource_schedule are numeric arrays.
experiment_code must be a one-line stdlib-only Python program that reads json.load(sys.stdin),
then reads payload['context']['case_id'] and payload['context']['problem'] before printing point,
oracle_calls, and unit_tests as JSON. It must branch on case_id and compute a point from problem fields.
It must import only json, math, sys, random. The output must never echo the input payload.
Do not include trusted metrics in its output. Correct any invalid imports or syntax from the prior code.
CASE_ID: {case_id}
PUBLIC_PROBLEM: {capsule['public_problem']}
Previous invalid emission for context (you may replace it):
{previous_output[:12000]}
Return the JSON object immediately."""
            if last_error:
                attempt_prompt += (
                    "\nA previous emission was rejected by the strict parser. "
                    f"Reason: {last_error}. Emit a corrected single JSON object now. "
                    "Do not repeat the rejected object and do not use a code fence.\n"
                )
            if hasattr(tokenizer, "apply_chat_template"):
                chat_text = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": "You emit one strict JSON object for a research code gate."},
                        {"role": "user", "content": attempt_prompt},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                chat_text = attempt_prompt
            # Prefill the assistant with the JSON object opener. This constrains
            # the first token without post-processing or supplying proposal data.
            chat_text += "{"
            encoded = tokenizer(chat_text, return_tensors="pt")
            encoded = {key: value.to(model_device) for key, value in encoded.items()}
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.eos_token_id,
                )
            continuation = tokenizer.decode(generated[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True)
            raw = continuation if continuation.lstrip().startswith("{") else "{" + continuation
            (case_root / f"raw_model_output_attempt_{attempt}.txt").write_text(raw, encoding="utf-8")
            try:
                value = extract_json(raw)
                code = value["experiment_code"]
                if not isinstance(code, str) or len(code.encode("utf-8")) > 256_000:
                    raise ValueError("invalid experiment_code size")
                validate_code(code)
                proposal = parse_model_proposal(
                    json.dumps(value),
                    task=task,
                    source=CandidateSource.STUDENT,
                    model_id="qwen2.5-7b-instruct",
                    model_digest=args.model_digest,
                    round_index=0,
                    frozen_seeds=SEEDS,
                    mandatory_baselines=(BASELINES[0],),
                    allowed_baselines=frozenset(BASELINES),
                    allowed_metrics=frozenset(METRICS),
                    frozen_resource_schedule=RESOURCE_SCHEDULE,
                )
                break
            except Exception as error:
                previous_output = raw.strip()
                last_error = f"{type(error).__name__}: {error}"
        if proposal is None:
            raise ValueError(f"all Qwen proposal attempts failed for {case_id}: {last_error}")
        (case_root / "raw_model_output.txt").write_text(raw, encoding="utf-8")
        (case_root / "experiment_code.py").write_text(proposal.experiment_code, encoding="utf-8")
        (case_root / "proposal.json").write_text(json.dumps(proposal.payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"case_id": case_id, "proposal_digest": proposal.digest, "code_sha256": file_hash(case_root / "experiment_code.py"), "raw_output_chars": len(raw)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
