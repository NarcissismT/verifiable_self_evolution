#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${1:-vse-qlora-smoke:0.1}"
docker build --pull=false -f "$ROOT/docker/Dockerfile" -t "$IMAGE" "$ROOT"
DIGEST="$(docker image inspect --format '{{.Id}}' "$IMAGE")"
if [[ ! "$DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "invalid built image digest: $DIGEST" >&2
  exit 2
fi
LOCK="$ROOT/docker/image_digest.lock"
if [[ -f "$LOCK" && "$(tr -d '\n' < "$LOCK")" != "$DIGEST" ]]; then
  echo "refusing to replace different image digest lock: $LOCK" >&2
  exit 2
fi
printf '%s\n' "$DIGEST" > "$LOCK"
echo "$DIGEST"
