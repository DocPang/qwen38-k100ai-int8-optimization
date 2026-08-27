#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_IMAGE="${BASE_IMAGE:-qwen38-k100ai-int8:unified-20260826-fa260728-q8split-rc2}"
OUT_IMAGE="${OUT_IMAGE:-qwen38-k100ai-int8:unified-20260827-v1.2.1}"

if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
  cat >&2 <<EOF
Base image not found locally:
  $BASE_IMAGE

Please import the v1.2.0 RC2 image first, then rerun this script.
This hotfix does not download or modify the original image.
EOF
  exit 2
fi

printf 'Building TP4 v1.2.1 hotfix layer...\n'
printf '  base: %s\n' "$BASE_IMAGE"
printf '  out : %s\n' "$OUT_IMAGE"

docker build \
  --network=none \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  -t "$OUT_IMAGE" \
  -f "$HERE/Dockerfile" \
  "$HERE"

docker run --rm --entrypoint /bin/bash "$OUT_IMAGE" -lc '
set -e
PATCH=/data/qwen38-dflash2-k100ai/runtime_patch_tp4
EXPECTED=/data/qwen38-dflash2-k100ai/runtime_patch_dflash_tp4_fa260728_rawq8_layout0_v1
test "$(readlink -f "$PATCH")" = "$EXPECTED"
grep -q "raw_paged_layout = 0" /data/qwen38-dflash2-k100ai/runtime_patch_dflash_tp4_agent128k_v1/sitecustomize.py
grep -q "SGLANG_DFLASH2_Q8_NATIVE_PAGED" "$EXPECTED/sitecustomize.py"
python3 -m py_compile \
  /data/qwen38-dflash2-k100ai/runtime_patch_dflash_tp4_agent128k_v1/sitecustomize.py \
  "$EXPECTED/sitecustomize.py"
'

cat <<EOF

Hotfix image ready:
  $OUT_IMAGE

Use it exactly like v1.2.0. PROFILE=tp1/tp2/tp4, PORT and MODEL_NAME remain unchanged.
Only TP4 runtime patch selection changes; TP1 and TP2 payloads are untouched.

Rollback: switch IMAGE back to
  $BASE_IMAGE
EOF
