#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_IMAGE=${BASE_IMAGE:-harbor.sourcefind.cn:5443/dcu/admin/base/custom@sha256:366525b25f452f85eb0ea5813604a64f03c648627bc824bb498b56cf5a325dde}
OUT="$ROOT/.build/native"

if [[ ! -d /opt/hyhal ]]; then
  echo "ERROR: /opt/hyhal not found. Stop here; do not modify the host automatically." >&2
  exit 2
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker not found." >&2
  exit 3
fi

mkdir -p "$OUT"
rm -f "$OUT"/*_sglang.so

echo "[safe-build] base image: $BASE_IMAGE"
echo "[safe-build] host mount: /opt/hyhal -> /opt/hyhal (read-only)"
echo "[safe-build] GPU devices: NONE"
echo "[safe-build] network: NONE"
echo "[safe-build] output: $OUT"

docker run --rm \
  --network none \
  -e PYTORCH_ROCM_ARCH=gfx928 \
  -e NATIVE_SRC=/src \
  -e NATIVE_OUT=/out \
  -v "$ROOT/native_ext:/src:ro" \
  -v "$OUT:/out" \
  -v /opt/hyhal:/opt/hyhal:ro \
  "$BASE_IMAGE" \
  python3 /src/build_native.py

required=(
  k100_int8_gemv_v7_sglang.so
  k100_int8_gemv_generic_v2_sglang.so
  k100_int8_gemv_k5120_full5_sglang.so
  k100_int8_gemv_k5120_ldsx_v1_sglang.so
  k100_int8_gemv_deep_v4_sglang.so
  k100_int8_gemv_tp2_row_ldsx_v1_sglang.so
  k100_int8_gemv_tp4_row_ldsx_v1_sglang.so
)
for f in "${required[@]}"; do
  [[ -s "$OUT/$f" ]] || { echo "ERROR: missing build output $OUT/$f" >&2; exit 4; }
done

sha256sum "$OUT"/*_sglang.so

echo "[safe-build] PASS: built seven user-space HIP extensions; host GPU driver was not modified."
