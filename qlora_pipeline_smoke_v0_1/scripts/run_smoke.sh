#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_ROOT="${1:?usage: $0 MODEL_ROOT OUTPUT_ROOT GPU_INDEX [IMAGE]}"
OUTPUT_ROOT="${2:?usage: $0 MODEL_ROOT OUTPUT_ROOT GPU_INDEX [IMAGE]}"
GPU_INDEX="${3:?usage: $0 MODEL_ROOT OUTPUT_ROOT GPU_INDEX [IMAGE]}"
IMAGE="${4:-vse-qlora-smoke:0.1}"

if [[ -e "$OUTPUT_ROOT" ]] && [[ -n "$(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing to overwrite nonempty output: $OUTPUT_ROOT" >&2
  exit 2
fi
if [[ ! -f "$MODEL_ROOT/MODEL_MANIFEST.json" ]]; then
  echo "missing MODEL_MANIFEST.json under $MODEL_ROOT" >&2
  exit 2
fi
IMAGE_DIGEST="$(docker image inspect --format '{{.Id}}' "$IMAGE")"
LOCKED_DIGEST="$(tr -d '\n' < "$ROOT/docker/image_digest.lock")"
if [[ "$IMAGE_DIGEST" != "$LOCKED_DIGEST" ]]; then
  echo "runtime image differs from locked image digest" >&2
  exit 2
fi
FREE_MIB="$(nvidia-smi --id="$GPU_INDEX" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "$FREE_MIB" =~ ^[0-9]+$ ]] || (( FREE_MIB < 10240 )); then
  echo "GPU $GPU_INDEX has only ${FREE_MIB:-unknown} MiB free; require at least 10240 MiB" >&2
  exit 3
fi

WORK_ROOT="$(mktemp -d /tmp/vse-qlora-smoke.XXXXXX)"
mkdir -p "$WORK_ROOT/input/data" "$WORK_ROOT/output"
cp "$ROOT/config/smoke_config.json" "$WORK_ROOT/input/config.json"
cp "$ROOT/data/dataset_manifest.json" "$WORK_ROOT/input/data/"
cp "$ROOT/data/synthetic_train.jsonl" "$WORK_ROOT/input/data/"
python "$ROOT/scripts/prepare_smoke_freeze.py" \
  --config "$ROOT/config/smoke_config.json" \
  --dataset-manifest "$ROOT/data/dataset_manifest.json" \
  --model-root "$MODEL_ROOT" \
  --image-digest "$IMAGE_DIGEST" \
  --code "$ROOT/scripts/train_and_verify.py" \
         "$ROOT/scripts/generate_synthetic_data.py" \
         "$ROOT/scripts/prepare_smoke_freeze.py" \
         "$ROOT/scripts/finalize_smoke.py" \
  --output "$WORK_ROOT/input/smoke_freeze_manifest.json"

COMMON=(
  --rm --network none
  -e VSE_NETWORK_POLICY=none
  -e "VSE_RUNTIME_IMAGE_DIGEST=$IMAGE_DIGEST"
  -v "$MODEL_ROOT:/model:ro"
  -v "$WORK_ROOT/input:/input:ro"
  -v "$WORK_ROOT/output:/output:rw"
)
docker run "${COMMON[@]}" "$IMAGE" validate-mask \
  --config /input/config.json \
  --freeze /input/smoke_freeze_manifest.json \
  --dataset-manifest /input/data/dataset_manifest.json \
  --model-root /model
docker run --gpus "device=$GPU_INDEX" "${COMMON[@]}" "$IMAGE" train \
  --config /input/config.json \
  --freeze /input/smoke_freeze_manifest.json \
  --dataset-manifest /input/data/dataset_manifest.json \
  --model-root /model \
  --output /output
docker run --gpus "device=$GPU_INDEX" "${COMMON[@]}" "$IMAGE" reload \
  --config /input/config.json \
  --freeze /input/smoke_freeze_manifest.json \
  --dataset-manifest /input/data/dataset_manifest.json \
  --model-root /model \
  --output /output

cp "$WORK_ROOT/input/config.json" "$WORK_ROOT/output/config.json"
cp "$WORK_ROOT/input/smoke_freeze_manifest.json" "$WORK_ROOT/output/smoke_freeze_manifest.json"
cp "$WORK_ROOT/input/data/dataset_manifest.json" "$WORK_ROOT/output/dataset_manifest.json"
python "$ROOT/scripts/finalize_smoke.py" --output-root "$WORK_ROOT/output"
mkdir -p "$OUTPUT_ROOT"
cp -a "$WORK_ROOT/output/." "$OUTPUT_ROOT/"
echo "QLoRA pipeline smoke passed: $OUTPUT_ROOT/SMOKE_REPORT.json"
