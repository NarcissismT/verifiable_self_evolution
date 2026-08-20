# v0.1.4 Qwen 7B confirmation result

The new five-case confirmation set was permanently marked ineligible for
formal splits. Qwen 7B generated parser-valid proposals for all five cases:

```yaml
parser_valid_rate: 1.0
public_runtime_valid_rate_after_two_repairs: 0.2
negative_controls: 7/7 rejected
```

The trusted wrapper then executed every proposal at budgets 1000, 2000, and
4000 for all four frozen seeds, repeated the final budget for reproducibility,
and computed metrics independently. Candidate-reported metrics were not read.

| case | all four seeds executable/reproducible | conditional VDS | unconditional VDS |
| --- | ---: | ---: | ---: |
| `confirm_quadratic_1d` | no (0/4) | 0.000 | 0.000 |
| `confirm_quadratic_2d` | no (0/4) | 0.000 | 0.000 |
| `confirm_box_projection_2d` | no (0/4) | 0.000 | 0.000 |
| `confirm_affine_shift_2d` | yes (4/4) | 0.771 | 0.771 |
| `confirm_seeded_target_2d` | no (0/4) | 0.000 | 0.000 |

The executable failures were public runtime-contract failures, including
`np` not being defined and incorrect use of the frozen SDK function signature.
They were not silently converted into scientific failures, and no hidden
optimum or evaluator detail was exposed to the model.

The receipt binds model digest, generation/execution image digest, case
manifest digest, SDK lock digest, and report digest. The result is below the
preregistered minimum of 4/5 cases, so the bridge remains blocked.

```yaml
v0.1.4_executable_proposal_bridge: BLOCKED_1_OF_5_EXECUTABLE
single_generation_7b_causal_qlora: BLOCKED
scientific_claims_allowed: false
eligible_for_champion: false
eligible_for_training_library: false
```

The result is a confirmation-gate failure, not evidence that QLoRA or GPU
training is broken. It does not authorize training on these trajectories.
