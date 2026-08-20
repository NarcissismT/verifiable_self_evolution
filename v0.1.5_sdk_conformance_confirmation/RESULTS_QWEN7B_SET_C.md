# Qwen2.5-7B v0.1.5 SDK-conformance result

The final set-C result replays the already frozen raw Qwen outputs through a
target-neutral protocol compiler. The replay made zero model calls. It inserted
only a missing system-frozen `resource_schedule=[1000,2000,4000]` and normalized
literal JSON `\\n` sequences into Python newlines. It did not create or modify
hypotheses, solutions, identifiers, algorithms, problem values, or metrics.

```yaml
model_digest: 6791085ae67e1e7dd6cdff568b903a9698fefd042397c8a0d84c33869afde37e
execution_image_digest: sha256:64a5804ef49f7cfaaaca42d53e9d2515695e772ce34452a90ba0e52a75defa8f
raw_schema_complete_rate: 0.0
canonical_parser_valid_rate: 1.0
initial_execution_rate: 0.8
execution_rate_after_repair: 0.8
valid_solver_result_cases: 4_of_5
all_four_seeds_per_passing_case: true
negative_controls_rejected: 7_of_7
scientific_hard_pass_cases: 0_of_5
sdk_contract_gate: PASS
scientific_solver_capability: NOT_ESTABLISHED
qlora: NO_GO
```

The gate pass means the base model can now reach the trusted wrapper reliably
enough to separate execution conformance from algorithm quality. Four passing
programs used the public zero-point interface pattern and matched the zero
baseline; none met the scientific hard threshold. The fifth program contained
an unterminated string after transparent newline normalization and remained
fail-closed. No code-repair call was used during replay.

All set-C cases are permanently excluded from formal train/dev/promotion,
held-out, and OOD splits. Raw outputs, canonical proposals, bindings, chains,
receipt, and evaluator report are archived in
`artifacts/sdk_conformance_qwen7b_set_c/`.

Per the preregistered sequence, this result unlocks only a closed-teacher
engineering acquisition dry-run on another permanently excluded fixture. It
does not unlock QLoRA, 14B, champion promotion, held-out/OOD, or recursion.
