#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_PATH="${1:?usage: $0 MODEL_PATH [RUN_ROOT] [IMAGE]}"
RUN_ROOT="${2:-$BUNDLE_ROOT/run}"
IMAGE="${3:-registry.intsig.net/zhuochu_yang/diffsynth:v2-diffusers}"
CASES=(realpilot_flatness_penalty realpilot_nonconvex_simple realpilot_linear_coupling)

MODEL_MANIFEST="$MODEL_PATH/MODEL_MANIFEST.json"
if [[ ! -f "$MODEL_MANIFEST" ]]; then
  echo "missing model manifest: $MODEL_MANIFEST" >&2
  exit 2
fi
MODEL_DIGEST="$(python - "$MODEL_MANIFEST" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
digest = value.get("manifest_digest", "")
if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
    raise SystemExit("invalid model manifest digest")
print(digest)
PY
)"

for case_id in "${CASES[@]}"; do
  if [[ -e "$RUN_ROOT/proposals/$case_id" || -e "$RUN_ROOT/evaluator/$case_id" ]]; then
    echo "refusing to overwrite existing case: $case_id" >&2
    exit 2
  fi
done

WORK_ROOT="$(mktemp -d /tmp/vse-minimal-qwen-pilot.XXXXXX)"
mkdir -p "$WORK_ROOT/public" "$WORK_ROOT/proposals" "$WORK_ROOT/evaluator" "$WORK_ROOT/student"
cp -a "$RUN_ROOT/public/." "$WORK_ROOT/public/"
cp "$BUNDLE_ROOT/student/generate_qwen_proposals.py" "$WORK_ROOT/student/"
cp "$BUNDLE_ROOT/evaluator/trusted_evaluator.py" "$WORK_ROOT/trusted_evaluator.py"

docker run --rm --network none \
  -e VSE_NETWORK_POLICY=none -e OMP_NUM_THREADS=32 -e MKL_NUM_THREADS=32 \
  -v "$MODEL_PATH:/model:ro" \
  -v "$WORK_ROOT/public:/public:ro" \
  -v "$WORK_ROOT/proposals:/proposals:rw" \
  -v "$WORK_ROOT/student:/student:ro" \
  "$IMAGE" python /student/generate_qwen_proposals.py \
  --model-path /model --model-digest "$MODEL_DIGEST" \
  --public-root /public --output-root /proposals \
  --max-new-tokens 256 --cpu-threads 32 --dtype float32

docker run --rm --network none -e VSE_NETWORK_POLICY=none \
  -v "$WORK_ROOT:/work:rw" "$IMAGE" bash -lc '
    set -eu
    for case_id in realpilot_flatness_penalty realpilot_nonconvex_simple realpilot_linear_coupling; do
      python /work/trusted_evaluator.py execute \
        --case-id "$case_id" \
        --proposal "/work/proposals/$case_id/proposal.json" \
        --output "/work/evaluator/$case_id.execution.json"
      python /work/trusted_evaluator.py evaluate \
        --execution "/work/evaluator/$case_id.execution.json" \
        --output "/work/evaluator/$case_id.evaluation.json"
    done
  '

mkdir -p "$RUN_ROOT/proposals" "$RUN_ROOT/evaluator"
for case_id in "${CASES[@]}"; do
  cp -a "$WORK_ROOT/proposals/$case_id" "$RUN_ROOT/proposals/"
  mkdir -p "$RUN_ROOT/evaluator/$case_id"
  cp "$WORK_ROOT/evaluator/$case_id.execution.json" "$RUN_ROOT/evaluator/$case_id/execution.json"
  cp "$WORK_ROOT/evaluator/$case_id.evaluation.json" "$RUN_ROOT/evaluator/$case_id/evaluation.json"
done
cp "$MODEL_MANIFEST" "$RUN_ROOT/model_manifest.json"

python - "$RUN_ROOT" "$MODEL_DIGEST" <<'PY'
import hashlib, json, sys
from pathlib import Path

run_root = Path(sys.argv[1])
model_digest = sys.argv[2]
cases = []
for case_dir in sorted((run_root / "proposals").iterdir()):
    proposal = json.loads((case_dir / "proposal.json").read_text(encoding="utf-8"))
    evaluation = json.loads((run_root / "evaluator" / case_dir.name / "evaluation.json").read_text(encoding="utf-8"))
    execution = json.loads((run_root / "evaluator" / case_dir.name / "execution.json").read_text(encoding="utf-8"))
    canonical = json.dumps(proposal, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    cases.append({
        "case_id": case_dir.name,
        "proposal_digest": hashlib.sha256(canonical.encode()).hexdigest(),
        "execution_digest": execution["execution_digest"],
        "evaluation_digest": evaluation["evaluation_digest"],
        "hard_pass": evaluation["hard_pass"],
        "vds_score": evaluation["vds_score"],
    })
report = {
    "schema_version": 1,
    "status": "engineering_vertical_slice_pass" if all(c["hard_pass"] for c in cases) else "engineering_vertical_slice_fail",
    "scientific_claims_allowed": False,
    "formal_ready": False,
    "model_manifest_digest": model_digest,
    "network_policy": "none",
    "cases": cases,
    "limitations": [
        "Qwen chooses algorithm_family and hypothesis; qwen_solution.py is a fixed target-neutral engineering adapter.",
        "This validates execution/evaluation control flow only, not rediscovery, power, QLoRA, or recursive evolution.",
        "Independent semantic and evaluator-custodian attestations remain required for formal ready preflight.",
    ],
}
(run_root / "MINIMAL_ENGINEERING_PILOT_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
PY

echo "minimal Qwen engineering pilot passed; evidence: $RUN_ROOT/MINIMAL_ENGINEERING_PILOT_REPORT.json"
