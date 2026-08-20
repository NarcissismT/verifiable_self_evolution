# v0.1.5 SDK-conformance confirmation

This package follows the v0.1.5 review decision. The historical v0.1.4
five-case result remains a frozen failure and is not retuned. This package
uses five new synthetic cases that are permanently excluded from every formal
split.

The gate separates three measurements:

* `parser_valid_rate`: complete structured proposals;
* `initial_execution_rate`: the first code before any repair;
* `execution_rate_after_repair`: executable code after at most two public,
  code-only repairs.

Repairs receive only the frozen SDK card, the original code, a public linter or
runtime/schema error, and the parent proposal digest. The hypothesis, solution,
seeds, baselines, metrics, power assumptions, stopping rule, and resource
schedule are checked unchanged. A proposal binding and parent hash chain are
required by the trusted gate. Candidate-reported metrics are never consumed.

The public SDK card is `sdk/sdk_card.json`; `sdk/sdk_lock.json` freezes the
allowlist and NumPy version. `scripts/bind_sdk.py` binds SDK source, card,
wrapper, lock, and execution image digests.

Prerecorded acceptance rule:

```yaml
confirmation_cases: 5_new_cases
parser_valid: 5_of_5
public_runtime_valid: 4_of_5_cases
all_four_seeds_per_case: true
maximum_code_repairs: 2
hypothesis_frozen_during_repair: true
negative_controls: 7_of_7_rejected
candidate_metrics_trusted: false
scientific_hard_pass: report_only
qlora: blocked_until_teacher_and_pilot_freeze
```

Local contract tests (no model) are runnable with:

```bash
cd verifiable_self_evolution
PYTHONPATH=. python -m unittest discover -s v0.1.5_sdk_conformance_confirmation/tests -q
```

The model command is:

```bash
bash v0.1.5_sdk_conformance_confirmation/scripts/run_qwen_sdk_confirmation.sh \
  /path/to/Qwen2.5-7B-Instruct \
  /path/to/output
```

The expected result of this package is a conformance receipt, not a scientific
claim. Even a passing SDK gate only unlocks the separately frozen closed-teacher
dry-run; it does not launch QLoRA, 14B, promotion, held-out/OOD, or recursion.

The final set-C receipt records both raw and canonical parsing. The trusted
protocol compiler may hydrate only predeclared system-frozen administrative
fields; it cannot synthesize `hypothesis`, `solution`, or `experiment_code`.
This keeps JSON copying errors separate from research behavior while preserving
the raw output and every transformation digest.
