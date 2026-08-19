You are given one frozen pre-cutoff evidence capsule. Propose a target-neutral
first-order algorithm for the public problem. Do not guess a later paper,
paper title, author, algorithm acronym, venue result or hidden evaluator.

Your output consists of:

1. `proposal.json`, conforming to `schemas/proposal.schema.json`;
2. one Python implementation referenced by `proposal.json`.

The implementation must export:

```python
def solve(case_id, problem, seed, oracle_budget, hyperparameters):
    return {"point": [float, ...], "oracle_calls": int}
```

Constraints:

- `implementation_sha256` must be the SHA-256 of the referenced implementation
  file; the evaluator independently rechecks this binding.

- use exactly seeds `[1031, 2063, 4099, 8191]` and oracle budget `4000`;
- use only Python and NumPy;
- do not access the network, filesystem paths outside the proposal directory,
  environment secrets, clocks or randomness other than the supplied seed;
- return two coordinates for the flatness and nonconvex-simple cases, four for
  the linearly coupled case;
- record the SHA-256 model/checkpoint manifest digest in `model_digest`;
- explain the pre-cutoff reasoning in `hypothesis` without naming a post-cutoff
  target.

The full public capsule JSON and frozen source text follow this instruction in
the actual generation runtime.
