# v0.1.4 executable proposal bridge

This is a new confirmation gate after the historical v0.1.3 failure. The
three v0.1.3 real-pilot cases remain permanently recorded as a failed
development result; they are not tuned until they pass.

v0.1.4 separates two questions:

1. **Execution contract:** can the model emit a proposal whose program imports
   the frozen target-neutral SDK, defines `solve(problem, seed, budget)`, runs
   under the network-disabled wrapper, and returns a finite point plus a legal
   oracle count?
2. **Scientific capability:** conditional VDS, unconditional VDS, lower
   residual, KKT/NI-style gap, regret, and resource-to-quality improvement are
   reported independently. Scientific thresholds are not an execution launch
   gate in this bridge.

The SDK does not contain a case solver or a reference algorithm. It only
provides deterministic seed initialization, a finite-vector check, box
projection, and a budget counter. Candidate code may be normal multi-line
Python. The wrapper owns stdin/stdout JSON, seed setup, budget enforcement,
schema checks, timeout, and error sealing.

The five confirmation cases are new, independent, and permanently excluded
from every formal split. The preregistered bridge acceptance rule is:

```yaml
parser_valid_rate: 1.0
valid_solver_result: 4_of_5_cases
all_four_seeds_required_per_case: true
negative_controls_rejected: 7_of_7
container_and_receipt_binding: exact
candidate_supplied_metrics_trusted: false
hard_pass_rate: reported_only
```

No bridge output is eligible for QLoRA training data, champion promotion,
held-out/OOD evaluation, or scientific claims.

The model generation command is:

```bash
bash v0.1.4_executable_proposal_bridge/scripts/run_qwen_bridge.sh \
  /tmp/vse-qwen7b-model \
  v0.1.4_executable_proposal_bridge/runs/qwen7b
```

The deterministic SDK/wrapper contract can be tested without a model:

```bash
python -m unittest discover -s v0.1.4_executable_proposal_bridge/tests -q
```
