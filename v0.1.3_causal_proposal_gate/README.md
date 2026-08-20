# v0.1.3 causal proposal gate

This gate is the next experiment after the isolated 7B QLoRA engineering
smoke. It does not train a model and does not run any causal QLoRA arm. Its
sole purpose is to test the causal binding:

```text
Qwen raw output -> strict ExperimentProposal -> generated code -> execution
-> independently computed metrics -> VerificationReport -> TrajectoryRecord
```

The student implementation is always the exact `experiment_code` emitted by
Qwen. The old fixed adapter is never used as a fallback. It is only available
as a separately labeled negative-control artifact.

The gate runs three permanently excluded engineering cases from the v0.1.2
bundle, using the frozen four seeds. It requires all three cases to pass for
the positive run and requires every negative control to fail closed:

- swapped case code/hypothesis;
- correct hypothesis with broken code;
- random hypothesis with the fixed adapter;
- fabricated trusted metrics;
- altered seed/baseline/resource schedule;
- target leakage;
- network marker, timeout, or evaluator digest tampering.

No output from this directory is eligible for a training library, champion,
promotion, held-out/OOD evaluation, or scientific claim. A passing gate only
unblocks design work for the later single-generation 7B causal pilot.

The real-Qwen command is:

```bash
bash v0.1.3_causal_proposal_gate/scripts/run_qwen_gate.sh \
  /tmp/vse-qwen7b-model \
  v0.1.2_real_pilot_bundle \
  v0.1.3_causal_proposal_gate/runs/qwen7b
```
