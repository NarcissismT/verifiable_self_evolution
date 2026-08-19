#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"

python - "$BUNDLE_ROOT" <<'PY'
import ast
from pathlib import Path
import sys

root = Path(sys.argv[1])
for path in sorted(root.rglob("*.py")):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
PY
python -m json.tool "$BUNDLE_ROOT/admin/targets_admin.json" >/dev/null
python -m json.tool "$BUNDLE_ROOT/admin/pilot_exclusions.json" >/dev/null
python -m json.tool "$BUNDLE_ROOT/schemas/proposal.schema.json" >/dev/null
python -m json.tool "$BUNDLE_ROOT/run_status.json" >/dev/null
if python -c 'import numpy; assert numpy.__version__ == "1.26.4"' >/dev/null 2>&1; then
  VSE_NETWORK_POLICY=none python "$BUNDLE_ROOT/evaluator/trusted_evaluator.py" \
    self-test --reference-module "$BUNDLE_ROOT/student/reference_baseline.py"
else
  echo 'static validation only: host numpy==1.26.4 unavailable; run evaluator self-test in the built image' >&2
fi

if command -v rg >/dev/null 2>&1; then
  forbidden_scan=(rg -n 'zero[_]sum' "$BUNDLE_ROOT")
else
  forbidden_scan=(grep -RInE 'zero[_]sum' "$BUNDLE_ROOT")
fi
if "${forbidden_scan[@]}"; then
  echo 'forbidden legacy stratum term found' >&2
  exit 2
fi

echo 'bundle static validation passed'
