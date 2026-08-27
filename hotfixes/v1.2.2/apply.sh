#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_IMAGE_WAS_SET="${BASE_IMAGE+x}"
BASE_IMAGE="${BASE_IMAGE:-qwen38-k100ai-int8:unified-20260827-v1.2.1}"
OUT_IMAGE="${OUT_IMAGE:-qwen38-k100ai-int8:unified-20260827-v1.2.2}"

if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
  if [[ -z "$BASE_IMAGE_WAS_SET" && "$BASE_IMAGE" == "qwen38-k100ai-int8:unified-20260827-v1.2.1" ]]; then
    V121_APPLY="$HERE/../v1.2.1/apply.sh"
    if [[ -x "$V121_APPLY" ]]; then
      echo "v1.2.1 parent image not found; building prerequisite hotfix first..."
      "$V121_APPLY"
    fi
  fi
fi

if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
  echo "Base image not found locally: $BASE_IMAGE" >&2
  echo "Import the v1.2.0 base image (or provide BASE_IMAGE), then rerun." >&2
  exit 2
fi

echo "Building v1.2.2 DFlash2 non-greedy hotfix..."
echo "  base: $BASE_IMAGE"
echo "  out : $OUT_IMAGE"

docker build --network=none \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  -t "$OUT_IMAGE" \
  -f "$HERE/Dockerfile" \
  "$HERE"

docker run --rm --entrypoint /bin/bash "$OUT_IMAGE" -lc '
set -e
PY=/usr/local/lib/python3.10/dist-packages/sglang/srt/speculative
python3 -m py_compile "$PY/dflash_worker.py" "$PY/dflash_info.py" "$PY/dflash_utils.py"
grep -q "compute_dflash_selector_sampling_correct_drafts_and_bonus" "$PY/dflash_utils.py"
! grep -q "K100AI DFlash2 backport currently supports greedy decoding only" "$PY/dflash_worker.py"
'

echo "Hotfix image ready: $OUT_IMAGE"
echo "Rollback image: $BASE_IMAGE"
