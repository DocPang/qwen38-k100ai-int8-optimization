#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_BASE='harbor.sourcefind.cn:5443/dcu/admin/base/custom@sha256:366525b25f452f85eb0ea5813604a64f03c648627bc824bb498b56cf5a325dde'
if [[ -z "${BASE_IMAGE:-}" && -f "$ROOT/.env" ]]; then
  env_base=$(grep -E '^BASE_IMAGE=' "$ROOT/.env" | tail -n1 | cut -d= -f2- || true)
  [[ -n "$env_base" ]] && BASE_IMAGE="$env_base"
fi
BASE_IMAGE=${BASE_IMAGE:-$DEFAULT_BASE}
IMAGE_TAG=${IMAGE_TAG:-qwen38-w8a8-k100ai-dflash2-tp4:local}

export BASE_IMAGE
"$ROOT/build_native.sh"

echo "[image-build] building $IMAGE_TAG from $BASE_IMAGE"
docker build \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  -t "$IMAGE_TAG" \
  "$ROOT"

echo "[image-build] PASS: $IMAGE_TAG"
