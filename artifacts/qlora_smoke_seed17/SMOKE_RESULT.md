# QLoRA Smoke Result

Status: `passed`

This is an engineering-only `qlora_pipeline_smoke` result. It is not a
scientific result and is permanently ineligible for champion promotion or the
formal training library.

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Revision: `d1e200fcf95ef0d4326873ddf63e5562d5f1fdbb`
- Model manifest digest: `6791085ae67e1e7dd6cdff568b903a9698fefd042397c8a0d84c33869afde37e`
- Runtime image digest: `sha256:f669006c1ce0d3761b4017e4c600c3e0424e4670a0b9262ec53cb33d3406a666`
- Data: 128 deterministic synthetic toy rows, dataset SHA-256 recorded in `dataset_manifest.json`
- Adapter seed: `17`
- Optimizer updates: `100`
- Quantization: NF4 4-bit, double quantization, BF16 compute
- Trainable parameters: `5,046,272`
- Peak GPU memory: `8,829,634,048` bytes
- Loss: `3.8211371899` at step 1, `0.0002387454` at step 100
- Gradient norm: minimum `0.0035526890`, maximum `5.3480783029`

Passed checks:

- exact segmented loss mask: assistant plan/action/result/belief on; system/user/tool observation off;
- finite, nonzero gradient at every optimizer update;
- adapter save with SHA-256 file manifest;
- fresh-process 4-bit base plus adapter reload;
- offline deterministic inference with `VSE_NETWORK_POLICY=none`.

Eligibility is fixed to:

```yaml
scientific_claims_allowed: false
eligible_for_champion: false
eligible_for_training_library: false
```

This smoke does not validate model-generated research proposals, paper-level
VDS, causal arm differences, power, promotion, held-out/OOD behavior, or
recursive evolution. The next permitted step is the separate v0.1.3 causal
proposal gate.
