#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_ROOT="$(cd "${1:?usage: $0 MODEL_ROOT BUNDLE_ROOT OUTPUT_ROOT [IMAGE]}" && pwd)"
BUNDLE_ROOT="$(cd "${2:?usage: $0 MODEL_ROOT BUNDLE_ROOT OUTPUT_ROOT [IMAGE]}" && pwd)"
OUTPUT_ROOT="$(mkdir -p "${3:?usage: $0 MODEL_ROOT BUNDLE_ROOT OUTPUT_ROOT [IMAGE]}" && cd "$3" && pwd)"
IMAGE="${4:-registry.intsig.net/zhuochu_yang/diffsynth:v2-diffusers}"
MODEL_DIGEST="$(python - "$MODEL_ROOT/MODEL_MANIFEST.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
digest = value.get("manifest_digest", "")
if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
    raise SystemExit("invalid model manifest digest")
print(digest)
PY
)"
if [[ -e "$OUTPUT_ROOT" ]] && [[ -n "$(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing to overwrite nonempty gate output: $OUTPUT_ROOT" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT/proposals" "$OUTPUT_ROOT/run" "$OUTPUT_ROOT/keys"
KEY_PATH="$OUTPUT_ROOT/keys/gate_hmac.key"
if [[ ! -f "$KEY_PATH" ]]; then
  umask 077
  openssl rand -out "$KEY_PATH" 32
fi
MODEL_IMAGE_DIGEST="$(docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null || true)"
if [[ -z "$MODEL_IMAGE_DIGEST" ]]; then
  echo "model generation image is not available locally: $IMAGE" >&2
  exit 2
fi

docker run --rm --gpus device=0 --network none \
  -e VSE_NETWORK_POLICY=none -e PYTHONPATH=/repo \
  -v "$MODEL_ROOT:/model:ro" \
  -v "$BUNDLE_ROOT/run/public:/public:ro" \
  -v "$OUTPUT_ROOT/proposals:/output:rw" \
  -v "$(cd "$ROOT/.." && pwd):/repo:ro" \
  "$IMAGE" python /repo/v0.1.3_causal_proposal_gate/scripts/generate_causal_proposals.py \
  --model-path /model --model-digest "$MODEL_DIGEST" \
  --public-root /public --output-root /output \
  --max-new-tokens 1536 --dtype float16

docker run --rm --network none \
  -e VSE_NETWORK_POLICY=none -e PYTHONPATH=/repo \
  -v "$(cd "$ROOT/.." && pwd):/repo:ro" \
  -v "$BUNDLE_ROOT:/bundle:ro" \
  -v "$OUTPUT_ROOT:/output:rw" \
  --entrypoint python "$IMAGE" /repo/v0.1.3_causal_proposal_gate/scripts/run_gate.py positive \
  --bundle-root /bundle \
  --proposals-root /output/proposals \
  --output-root /output/run \
  --model-digest "$MODEL_DIGEST" \
  --hmac-key /output/keys/gate_hmac.key

docker run --rm --network none \
  -e VSE_NETWORK_POLICY=none -e PYTHONPATH=/repo \
  -v "$(cd "$ROOT/.." && pwd):/repo:ro" \
  -v "$BUNDLE_ROOT:/bundle:ro" \
  -v "$OUTPUT_ROOT:/output:rw" \
  --entrypoint python "$IMAGE" /repo/v0.1.3_causal_proposal_gate/scripts/run_gate.py negative-controls \
  --bundle-root /bundle \
  --proposals-root /output/proposals \
  --output-root /output/run \
  --model-digest "$MODEL_DIGEST" \
  --hmac-key /output/keys/gate_hmac.key

cat > "$OUTPUT_ROOT/gate_environment.json" <<EOF
{
  "schema_version": 1,
  "model_digest": "$MODEL_DIGEST",
  "generation_image_digest": "$MODEL_IMAGE_DIGEST",
  "execution_container_digest": "$MODEL_IMAGE_DIGEST",
  "network_policy": "none",
  "scientific_claims_allowed": false,
  "eligible_for_champion": false,
  "eligible_for_training_library": false
}
EOF
echo "causal proposal gate completed: $OUTPUT_ROOT/run"
