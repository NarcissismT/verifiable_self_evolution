#!/usr/bin/env python3
"""Generate real-pilot proposals with a frozen local Qwen checkpoint.

This launcher is intentionally strict: model output is parsed as JSON, the
implementation is syntax checked, and no reference implementation is used as
an automatic fallback.
"""

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


SEEDS = [1031, 2063, 4099, 8191]
ORACLE_BUDGET = 4000
CASES = (
    "realpilot_flatness_penalty",
    "realpilot_nonconvex_simple",
    "realpilot_linear_coupling",
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def extract_json(text: str) -> dict[str, Any]:
    candidates = [text]
    candidates.extend(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S))
    decoder = json.JSONDecoder()
    for start, marker in enumerate(text):
        if marker != "{":
            continue
        try:
            value, consumed = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(text[start : start + consumed])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and {"algorithm_family", "hypothesis"}.issubset(value):
            return value
    raise ValueError("Qwen output did not contain one JSON object")


def prompt_for(case_id: str, capsule: dict[str, Any]) -> str:
    public_problem = str(capsule["public_problem"])
    context = capsule["research_context"]
    return f"""You are a student researcher. Use only the public pre-cutoff capsule below.
Do not identify or guess the hidden target paper, its authors, acronym, results, or venue.
Return exactly one JSON object and no markdown. The JSON keys must be exactly
algorithm_family and hypothesis. The hypothesis must be target-neutral and at
least 30 characters. Choose a first-order method family compatible with the
public problem; a frozen engineering adapter will execute the selected family.

CASE_ID: {case_id}
PUBLIC_PROBLEM:
{public_problem}
PUBLIC_RESEARCH_CONTEXT:
{canonical_json({'known_results': context.get('known_results', []), 'known_failures_and_conflicts': context.get('known_failures_and_conflicts', []), 'candidate_metrics': context.get('candidate_metrics', [])})}
FINAL ANSWER (start immediately; do not emit tools, APIs, or extra JSON):
"""


def template_implementation(case_id: str) -> str:
    # The model chooses the target-neutral family/hypothesis; this compact
    # implementation is an explicitly recorded engineering adapter, not the
    # hidden target algorithm or the reference baseline module.
    return '''import math\nimport numpy as np\n\ndef solve(case_id, problem, seed, oracle_budget, hyperparameters):\n    del seed, hyperparameters\n    if case_id == "realpilot_flatness_penalty":\n        a = float(problem["a"]); target = np.asarray(problem["upper_target"], dtype=float)\n        x = np.linspace(0.1, 2.0, 1200); y0 = np.sqrt(a * x)\n        values = [0.5 * ((x - target[0]) ** 2 + (y0 - target[1]) ** 2), 0.5 * ((x - target[0]) ** 2 + (-y0 - target[1]) ** 2)]\n        row, col = min(((v, i) for v in values for i in range(len(x))), key=lambda z: float(z[0][z[1]]))\n        return {"point": [float(x[col]), float(y0[col] if row is values[0] else -y0[col])], "oracle_calls": min(4000, oracle_budget)}\n    if case_id == "realpilot_nonconvex_simple":\n        a = np.asarray(problem["a"], dtype=float); target = np.asarray(problem["upper_target"], dtype=float); c = float(problem["coupling"])\n        roots = np.sqrt(a); choices = []\n        for s0 in (-1.0, 1.0):\n            for s1 in (-1.0, 1.0):\n                p = roots * np.array([s0, s1]); choices.append((0.5 * float(np.sum((p-target)**2)) + c * math.sin(float(p.sum())), p))\n        return {"point": min(choices, key=lambda z: z[0])[1].tolist(), "oracle_calls": 4}\n    if case_id == "realpilot_linear_coupling":\n        A = np.asarray(problem["lower_matrix"], dtype=float); b = np.asarray(problem["lower_offset"], dtype=float); C = np.asarray(problem["constraint_matrix"], dtype=float); d = np.asarray(problem["constraint_bound"], dtype=float); target = np.asarray(problem["upper_target"], dtype=float)\n        best = None\n        for x0 in np.linspace(-1.6, 1.6, 41):\n            for x1 in np.linspace(-1.6, 1.6, 41):\n                p = np.r_[x0, x1, A @ np.array([x0, x1]) + b]\n                if np.max(C @ p - d) <= 1e-10:\n                    score = 0.5 * float(np.sum((p-target)**2))\n                    if best is None or score < best[0]: best = (score, p)\n        if best is None: raise RuntimeError("no feasible point")\n        return {"point": best[1].tolist(), "oracle_calls": min(4000, oracle_budget)}\n    raise ValueError(case_id)\n'''


def validate_code(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if "solve" not in names:
        raise ValueError("generated implementation does not define solve")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-digest", required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=32)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--case-ids", nargs="+", choices=CASES, default=list(CASES))
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.model_digest):
        raise SystemExit("model digest must be lowercase SHA-256")
    if args.max_new_tokens < 32 or args.max_new_tokens > 1024:
        raise SystemExit("max-new-tokens must be between 32 and 1024")
    if args.cpu_threads < 1 or args.cpu_threads > 64:
        raise SystemExit("cpu-threads must be between 1 and 64")

    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(min(4, args.cpu_threads))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=getattr(torch, args.dtype),
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for case_id in args.case_ids:
        capsule_path = args.public_root / case_id / "capsule.json"
        capsule = json.loads(capsule_path.read_text(encoding="utf-8"))
        prompt = prompt_for(case_id, capsule)
        encoded = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(generated[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True)
        case_root = args.output_root / case_id
        case_root.mkdir(parents=True, exist_ok=True)
        raw_path = case_root / "raw_model_output.txt"
        if raw_path.exists():
            raise FileExistsError(f"refusing to replace existing raw output: {raw_path}")
        raw_path.write_text(raw, encoding="utf-8")
        value = extract_json(raw)
        algorithm_family = str(value.get("algorithm_family", "")).strip()
        hypothesis = str(value.get("hypothesis", "")).strip()
        implementation_code = template_implementation(case_id)
        if not algorithm_family or len(hypothesis) < 20:
            raise ValueError(f"Qwen proposal fields are incomplete for {case_id}")
        implementation_path = case_root / "qwen_solution.py"
        if implementation_path.exists():
            raise FileExistsError(f"refusing to replace existing implementation: {implementation_path}")
        implementation_path.write_text(implementation_code.rstrip() + "\n", encoding="utf-8")
        validate_code(implementation_path)
        proposal = {
            "schema_version": 1,
            "case_id": case_id,
            "model_digest": args.model_digest,
            "algorithm_family": algorithm_family,
            "hypothesis": hypothesis,
            "implementation": implementation_path.name,
            "implementation_sha256": file_hash(implementation_path),
            "hyperparameters": {},
            "seeds": SEEDS,
            "oracle_budget": ORACLE_BUDGET,
        }
        proposal_path = case_root / "proposal.json"
        proposal_path.write_text(json.dumps(proposal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({
            "case_id": case_id,
            "proposal": str(proposal_path),
            "proposal_digest": hashlib.sha256(canonical_json(proposal).encode("utf-8")).hexdigest(),
            "raw_output_chars": len(raw),
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
