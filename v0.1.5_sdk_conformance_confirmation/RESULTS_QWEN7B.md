# Qwen2.5-7B v0.1.5 protocol preflight

The frozen Qwen2.5-7B-Instruct checkpoint was run once on an initial five-case
protocol-development set. No QLoRA or teacher acquisition was run. This run is
not a valid conformance decision: a post-run audit found that two failed repair
children shared the initial proposal as parent instead of forming a continuous
child chain. Those five cases were retired immediately.

```yaml
model_digest: 6791085ae67e1e7dd6cdff568b903a9698fefd042397c8a0d84c33869afde37e
execution_image_digest: sha256:64a5804ef49f7cfaaaca42d53e9d2515695e772ce34452a90ba0e52a75defa8f
parser_valid_rate: 1.0
initial_execution_rate: 0.0
execution_rate_after_repair: 0.0
valid_solver_result_cases: 0_of_5
negative_controls_rejected: 7_of_7
gate: INVALID_PROTOCOL_PREFLIGHT_FAIL_CLOSED
qlora: NO_GO
```

All five initial outputs parsed as complete ExperimentProposal objects. The
candidate code nevertheless failed the public execution contract. Three final
codes were syntactically invalid placeholder text; two lacked a statically
verifiable `solve` return schema. Each case consumed exactly the preregistered
two repair attempts. No third repair or prompt retuning was performed on these
cases.

The output therefore diagnoses protocol representation failure, not scientific
solver quality. The prompt exposed placeholder strings that the model copied,
and no case reached trusted execution. Because the parent-chain implementation
was nonconforming, these rates are diagnostic only. The archived proposals,
hypothesis/code bindings, branched repair records, generation report,
SDK/image binding, report, exact preflight manifest, and receipt are in
`artifacts/sdk_conformance_qwen7b/`.

The corrected implementation uses executable target-neutral schema examples,
a continuous child hash chain, and a fresh permanently excluded confirmation
set. The preflight cases are never reused.
