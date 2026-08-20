# Qwen 7B gate result

The real run used the local Qwen 7B manifest
`6791085ae67e1e7dd6cdff568b903a9698fefd042397c8a0d84c33869afde37e` with
network policy `none`. The generation image digest was
`sha256:64a5804ef49f7cfaaaca42d53e9d2515695e772ce34452a90ba0e52a75defa8f`;
the execution image digest was
`sha256:f669006c1ce0d3761b4017e4c600c3e0424e4670a0b9262ec53cb33d3406a666`.

All three permanently excluded cases produced a strict, parser-valid
`ExperimentProposal` from Qwen. Each case then ran four frozen seeds through
the generated `experiment_code`, independent trusted metrics, the hard
verifier, and receipt/trajectory sealing.

| case | positive result | VDS |
| --- | --- | ---: |
| `realpilot_flatness_penalty` | rejected by trusted verifier | 0.0 |
| `realpilot_nonconvex_simple` | rejected by trusted verifier | 0.0 |
| `realpilot_linear_coupling` | rejected by trusted verifier | 0.0 |

The submissions were rejected because the emitted programs did not produce a
valid solver result for the evaluator. No fixed adapter or template fallback
was used. The seven registered negative controls all passed fail-closed; the
code/hypothesis mutation controls were rebound with new candidate digests and
actually executed before rejection. The negative-control receipts therefore do
not depend only on stale proposal identity fields.

Gate status:

```yaml
v0.1.3_causal_proposal_gate: BLOCKED_BY_EXECUTABLE_PROPOSAL
single_generation_7b_causal_qlora: BLOCKED
scientific_claims_allowed: false
eligible_for_champion: false
eligible_for_training_library: false
```

The complete small receipts, raw model outputs, proposals, execution records,
verification reports, trajectories, and bound receipts are under
`runs/qwen7b_seed17_gate/`. The HMAC key is intentionally not archived in the
repository. This run does not authorize QLoRA, promotion, held-out/OOD access,
14B replication, or recursive evolution.
