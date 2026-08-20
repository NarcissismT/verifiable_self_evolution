#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_ROOT="$(cd "${1:?usage: $0 MODEL_ROOT OUTPUT_ROOT [IMAGE]}" && pwd)"
OUTPUT_ROOT="$(mkdir -p "${2:?usage: $0 MODEL_ROOT OUTPUT_ROOT [IMAGE]}" && cd "$2" && pwd)"
IMAGE="${3:-registry.intsig.net/zhuochu_yang/diffsynth:v2-diffusers}"
MODEL_DIGEST="$(python - "$MODEL_ROOT/MODEL_MANIFEST.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
digest = value.get("manifest_digest", "")
if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
    raise SystemExit("invalid model manifest digest")
print(digest)
PY
)"
IMAGE_DIGEST="$(docker image inspect --format '{{.Id}}' "$IMAGE")"
mkdir -p "$OUTPUT_ROOT/proposals" "$OUTPUT_ROOT/run"

set +e
docker run --rm --gpus device=0 --network none -e VSE_NETWORK_POLICY=none -e PYTHONPATH=/repo \
  -v "$MODEL_ROOT:/model:ro" -v "$ROOT/cases:/cases:ro" -v "$ROOT/sdk:/sdk:ro" \
  -v "$OUTPUT_ROOT/proposals:/output:rw" -v "$(cd "$ROOT/.." && pwd):/repo:ro" "$IMAGE" \
  python /repo/v0.1.5_sdk_conformance_confirmation/scripts/generate_sdk_proposals.py \
  --model-path /model --model-digest "$MODEL_DIGEST" --case-manifest /cases/case_manifest.json \
  --sdk-card /sdk/sdk_card.json --output-root /output --max-new-tokens 3072
GENERATION_STATUS=$?
set -e

python "$ROOT/scripts/bind_sdk.py" --sdk-root "$ROOT/sdk" --execution-image-digest "$IMAGE_DIGEST" \
  --output "$OUTPUT_ROOT/run/sdk_binding.json"

set +e
docker run --rm --network none -e VSE_NETWORK_POLICY=none -e PYTHONPATH=/repo \
  -v "$(cd "$ROOT/.." && pwd):/repo:ro" -v "$ROOT:/bridge:ro" \
  -v "$OUTPUT_ROOT/proposals:/proposals:ro" -v "$OUTPUT_ROOT/run:/run:rw" --entrypoint python "$IMAGE" \
  /bridge/scripts/run_sdk_gate.py --case-manifest /bridge/cases/case_manifest.json \
  --proposals-root /proposals --output-root /run --model-digest "$MODEL_DIGEST" \
  --execution-container-digest "$IMAGE_DIGEST" --sdk-binding /run/sdk_binding.json \
  --generation-report /proposals/generation_report.json
GATE_STATUS=$?
set -e

python - "$OUTPUT_ROOT" "$GENERATION_STATUS" "$GATE_STATUS" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
generation_status, gate_status = map(int, sys.argv[2:4])
value = {
    "schema_version": 1,
    "gate": "v0.1.5_sdk_conformance_confirmation",
    "generation_process_exit": generation_status,
    "gate_process_exit": gate_status,
    "generation_completed": (root / "proposals" / "generation_report.json").is_file(),
    "gate_completed": (root / "run" / "sdk_conformance_report.json").is_file(),
    "qlora_launch_allowed": False,
}
(root / "run" / "overall_report.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
print(json.dumps(value, indent=2, sort_keys=True))
PY
echo "v0.1.5 SDK confirmation completed: $OUTPUT_ROOT/run (generation=$GENERATION_STATUS gate=$GATE_STATUS)"
if [[ "$GATE_STATUS" -ne 0 ]]; then exit "$GATE_STATUS"; fi
if [[ "$GENERATION_STATUS" -ne 0 ]]; then exit "$GENERATION_STATUS"; fi
