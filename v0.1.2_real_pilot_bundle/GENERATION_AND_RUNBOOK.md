# Real generation and trusted-stage runbook

The current repository receipt API requires `proposal_digest` before launching
the trusted generation command. A model proposal is therefore produced with a
two-pass deterministic replay:

1. rehearsal: run the exact generation image with temperature 0, frozen seed,
   frozen decoding parameters, `--network none`, read-only model and read-only
   public capsule mounts; write `proposal.json` and its implementation;
2. calculate and record the proposal digest;
3. delete only the rehearsal output copy, replay the exact same command through
   `produce-trusted-receipt`, and require byte-identical proposal and code
   hashes. A non-identical replay is a hard failure.

Do not pass `sealed/`, `review_packet/`, `admin/`, evaluator source, evaluation
outputs or target metadata to the generation container.

For the no-training engineering rehearsal, the complete local command is:

```bash
BUNDLE=v0.1.2_real_pilot_bundle
RUN="$BUNDLE/run"
bash "$BUNDLE/scripts/run_minimal_qwen_pilot.sh" \
  /tmp/vse-qwen7b-model \
  "$RUN"
```

The script uses the checkpoint's `MODEL_MANIFEST.json`, runs Qwen with
`--network none`, and writes proposal, raw-model-output, execution,
evaluation, and `MINIMAL_ENGINEERING_PILOT_REPORT.json` artifacts. It is
deliberately separate from the trusted two-pass producer flow below: the
result is an engineering rehearsal and cannot satisfy formal `ready`
preflight or unlock QLoRA.

For each case, after the semantic review has updated `capsule.json`:

```bash
CASE=realpilot_flatness_penalty
RUN=pilot/real_v0_1_2/run
BUNDLE=pilot/real_v0_1_2
MODEL_DIGEST=$(tr -d '\n' < /sealed/model/model_digest.txt)
GEN_IMAGE=$(tr -d '\n' < /sealed/model/generation_image_digest.txt)

PYTHONPATH=. python "$BUNDLE/scripts/proposal_digest.py" \
  "$RUN/proposals/$CASE/proposal.json" \
  --expected-model-digest "$MODEL_DIGEST" \
  > "$RUN/proposals/$CASE/proposal_digest.txt"
PYTHONPATH=. python "$BUNDLE/scripts/capsule_digest.py" \
  "$RUN/public/$CASE/capsule.json" \
  > "$RUN/public/$CASE/capsule_digest.txt"
```

The real generation command is installation-specific because the frozen Qwen
checkpoint/runtime paths are still external artifacts. Its receipt must follow
this shape (replace `YOUR_OFFLINE_GENERATOR_COMMAND...` with the same replayed
command used in rehearsal):

```bash
PYTHONPATH=. python -m vse.cli produce-trusted-receipt \
  --stage generation \
  --capsule-digest "$(tr -d '\n' < "$RUN/public/$CASE/capsule_digest.txt")" \
  --proposal-digest "$(tr -d '\n' < "$RUN/proposals/$CASE/proposal_digest.txt")" \
  --container-digest "$GEN_IMAGE" \
  --trust-key "$RUN/trusted_producer.key" \
  --runtime-mode docker \
  --output "$RUN/public/$CASE/generation_producer_receipt.json" \
  --artifact-path "$RUN/proposals/$CASE/proposal.json" \
  -- docker run --rm --network none --read-only --cap-drop ALL \
     --security-opt no-new-privileges --gpus all \
     -v "$RUN/public/$CASE:/capsule:ro" \
     -v "/sealed/model:/model:ro" \
     -v "$RUN/proposals/$CASE:/proposal:rw" \
     "$GEN_IMAGE" YOUR_OFFLINE_GENERATOR_COMMAND...
```

The command itself must refuse to overwrite a different proposal and must
print the proposal/code hashes. The digest-pinned generation image must already
contain the prompt template and generator code; do not install packages during
the run.

After all three generation producer receipts exist:

```bash
PYTHONPATH=. python "$BUNDLE/scripts/run_trusted_stages.py" \
  --run-root "$RUN" \
  --image-digest-file "$RUN/evaluator_image_digest.txt" \
  --trust-key "$RUN/trusted_producer.key"

PYTHONPATH=. python "$BUNDLE/scripts/build_stage_receipts.py" \
  --bundle-root "$BUNDLE" \
  --run-root "$RUN" \
  --trust-key "$RUN/trusted_producer.key" \
  --model-digest-file /sealed/model/model_digest.txt \
  --generation-image-digest-file /sealed/model/generation_image_digest.txt

PYTHONPATH=. python "$BUNDLE/scripts/preflight.py" \
  --bundle-root "$BUNDLE" --run-root "$RUN" --phase ready \
  --trust-key "$RUN/trusted_producer.key"
```

Only if the last command exits 0 should `vse run-vertical-slice` be used to
write the immutable report.
