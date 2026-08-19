# Verifiable Self-Evolution

This directory is an isolated experiment for verifier-gated improvement of a
7B/14B open model with QLoRA. It does not import code, configurations, tasks,
or results from any sibling workspace.

The formal paper-rediscovery freeze is documented in
[`FREEZE_V0_1.md`](FREEZE_V0_1.md) and
[`configs/paper_rediscovery_v0_1.json`](configs/paper_rediscovery_v0_1.json).
The quadratic configuration remains a control-plane smoke test only.
The hardening status and remaining exit conditions are tracked in
[`PROGRESS_V0_1_1.md`](PROGRESS_V0_1_1.md).

The current milestone is `v0.1.2-pre-pilot-fix`. It deliberately stops
before real QLoRA: the formal run is blocked until freeze artifacts and an
independent 3-5 capsule vertical slice are complete. The causal one-generation
pilot and later recursive study are separate configurations:
`configs/paper_rediscovery_causal_pilot_v0_2.json` and
`configs/paper_rediscovery_recursive_v1.json`.

The closed teacher has two permitted roles only:

1. produce bootstrap candidates for generation zero on training tasks;
2. repair a machine-identified hard state from the training pool.

Teacher output is never accepted as truth. It passes through the same executor
and hard verifier as every other candidate. The teacher cannot see held-out or
OOD tasks and cannot judge promotion.

## Frozen control flow

```text
train/dev task
  -> champion + legacy models + closed teacher (restricted) + variants
  -> structured hypothesis and discriminating experiment
  -> sandboxed code execution over frozen seeds and baselines
  -> analytic/KKT/unit/power/resource verification
  -> success library or counterexample library
  -> QLoRA training candidate

candidate checkpoint + current champion
  -> frozen promotion evaluator
  -> preregistered promotion gate
  -> promote or reject

After all generations are complete, the champion and final candidate are
evaluated once on separate final ID held-out and final OOD sets.
```

The training loop cannot consume held-out/OOD trajectories. The promotion gate
cannot consume training results, teacher judgments, or a mutable task list.

## What is implemented

- strict JSON-compatible contracts for tasks, hypotheses, executions, and
  verification reports;
- content-addressed frozen split manifests;
- a closed-teacher policy guard;
- atomic success and counterexample libraries;
- a task-plugin interface and hard-verifier interface;
- promotion based only on paired promotion-set receipts;
- one-time final ID held-out/OOD evaluation with a separate non-inferiority gate;
- an actual QLoRA SFT entrypoint, kept optional so the control-plane tests run
  without CUDA or training dependencies;
- a standalone box-constrained quadratic toy plugin that exercises analytic
  solutions, KKT residuals, unit checks, repeated seeds, and resource metrics.

The toy plugin validates orchestration only. A formal experiment must add its
own task plugin and freeze its task generator, verifier version, splits,
baselines, seeds, and promotion thresholds before generating training data.

## Quick check

From this directory:

```bash
python -m unittest discover -s tests -v
python -m vse.cli init --config configs/experiment_v1.json --root runs/demo
python -m vse.cli run-toy --root runs/demo
```

The second command seals the toy train/dev/promotion/heldout/OOD manifests. `run-toy` writes
training successes and failures plus dev diagnostics. It deliberately does not
run formal evaluation, train a model, or claim a scientific result.

Formal control-plane commands are available when their required manifests exist:

```bash
python -m vse.cli init-formal \
  --config configs/paper_rediscovery_v0_1.json \
  --candidates manifests/candidates.jsonl \
  --root runs/formal
python -m vse.cli freeze-check \
  --config configs/paper_rediscovery_v0_1.json \
  --root runs/formal
python -m vse.cli run-vertical-slice \
  --manifest pilot/vertical_slice_manifest.json \
  --public-root pilot/public \
  --sealed-root pilot/sealed \
  --trust-key /sealed/keys/vse-receipt.key \
  --output runs/formal/vertical_slice_report.json
```

`freeze-check` exits nonzero for pending implementation or artifact bindings.
`run-vertical-slice` requires 3-5 independent engineering cases marked as
excluded from every formal split. It validates trusted producer and semantic
review receipts; these cases are not training data and cannot estimate formal
paper-level power.

Real stage receipts must be produced through `produce-trusted-receipt` with a
Docker command that contains `--network none`. The private trust key stays
outside the public capsule; the manifest freezes only its SHA-256 digest.

## QLoRA handoff

Export and revalidate all frozen replay buckets:

```bash
python -m vse.cli export-training-data \
  --success-library runs/demo/libraries/success.jsonl \
  --counterexample-library runs/demo/libraries/counterexample.jsonl \
  --output-dir runs/demo/datasets
```

Formal QLoRA is intentionally blocked until `freeze-check` passes. Once it is
green in a future sealed run, the only accepted training invocation is:

```bash
python train_qlora.py \
  --config runs/formal/config.json \
  --run-root runs/formal \
  --model-profile qwen2.5-7b-instruct \
  --model-path /sealed/models/qwen2.5-7b \
  --dataset-manifest runs/formal/datasets/dataset_manifest.json \
  --adapter-seed 17 \
  --container-digest sha256:<frozen-train-image> \
  --output runs/formal/checkpoints/adapter_seed_17
```

The training script reads all model, tokenizer, QLoRA, replay, and seed values
from the sealed configuration, checks file hashes and exact token packing, and
records adapter/environment provenance. It refuses CLI hyperparameter drift.

## Real-task acceptance boundary

Do not begin a formal round until all of the following are filled in and
sealed:

- task family and OOD shift definition;
- hard metric directions, scales, and failure thresholds;
- analytic or independently implemented reference solver;
- baseline list and common resource unit;
- generation, execution, and evaluation seed registries;
- minimum detectable effect and power target;
- 7B/14B base checkpoint, tokenizer, prompt format, and QLoRA hyperparameters;
- sandbox/container image and network policy;
- held-out/OOD promotion rule and maximum number of promotion attempts.

Passing unit tests establishes only that the experiment controls are wired
correctly. It is not evidence that a real paper capsule, QLoRA training, or a
model upgrade succeeded.
