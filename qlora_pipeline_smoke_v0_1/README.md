# 7B QLoRA pipeline smoke v0.1

This directory is an isolated engineering smoke test. It is not a paper
rediscovery study, a causal model comparison, a power pilot, or a source of
champion/training-library records.

Frozen eligibility flags:

```yaml
study_kind: qlora_pipeline_smoke
scientific_claims_allowed: false
eligible_for_champion: false
eligible_for_training_library: false
```

The smoke uses only deterministic synthetic/toy examples. It verifies:

- offline Qwen2.5-7B loading from the frozen file manifest;
- NF4 4-bit quantization and one visible CUDA device;
- segmented assistant-only loss masking;
- 100 optimizer updates with finite, nonzero trainable gradients;
- adapter save and SHA-256 manifest;
- fresh-process adapter reload and network-disabled inference.

It does not call formal `train_qlora.py` and does not weaken its full
`freeze-check`. It also does not use anything under the real-paper pilot's
`run/sealed`, `run/public`, `run/proposals`, or evaluator outputs.

Build and prepare:

```bash
bash qlora_pipeline_smoke_v0_1/scripts/build_image.sh
python qlora_pipeline_smoke_v0_1/scripts/generate_synthetic_data.py \
  --config qlora_pipeline_smoke_v0_1/config/smoke_config.json \
  --output-root qlora_pipeline_smoke_v0_1/data
```

Run on one free GPU:

```bash
bash qlora_pipeline_smoke_v0_1/scripts/run_smoke.sh \
  /tmp/vse-qwen7b-model \
  qlora_pipeline_smoke_v0_1/runs/seed17 \
  0
```

The launcher refuses a GPU with less than 10 GiB free memory and refuses to
overwrite a nonempty output directory.
