# Verifiable Self-Evolution

This directory is an isolated experiment for verifier-gated improvement of a
7B/14B open model with QLoRA. It does not import code, configurations, tasks,
or results from any sibling workspace.

The formal paper-rediscovery freeze is documented in
[`FREEZE_V0_1.md`](FREEZE_V0_1.md) and
[`configs/paper_rediscovery_v0_1.json`](configs/paper_rediscovery_v0_1.json).
The quadratic configuration remains a control-plane smoke test only.

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

## QLoRA handoff

Export verifier-approved training records:

```bash
python -m vse.cli export-sft \
  --library runs/demo/libraries/success.jsonl \
  --output runs/demo/datasets/sft.jsonl
```

Install the optional training dependencies in the intended GPU environment,
then run:

```bash
python train_qlora.py \
  --model <local-or-hub-7b-or-14b-model> \
  --dataset runs/demo/datasets/sft.jsonl \
  --output runs/demo/checkpoints/candidate_r1
```

The formal v0.1 base model identifiers, revisions, QLoRA settings, and target
GPU are frozen in the paper-rediscovery configuration. The training script
records resolved arguments and refuses to overwrite a nonempty output.

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
correctly. It is not evidence that QLoRA training or a model upgrade succeeded.
