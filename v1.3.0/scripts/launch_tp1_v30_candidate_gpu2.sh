#!/usr/bin/env bash
set -euo pipefail

name=${NAME:-q38-tp1-v30-candidate-gpu2}
served=${SERVED_MODEL_NAME:-Qwen3.8-27B-W8A8-DFlash2-TP1-v30-candidate}
root=/data/qwen38-dflash2-k100ai
target=/data/qwen38-27b-k100ai-int8-opt
port=${PORT:-8042}
render=/dev/dri/renderD130
image=qwen38-k100ai-int8:unified-20260827-v1.2.2
image_id=sha256:ad30e85d745574295921f677054bebebee57f8beb444680205ac5fd5d5e05e0c

required=(
  "$root/scripts/entrypoint_tp1_v30.sh"
  "$root/runtime_patch_dflash_tp1_v30/sitecustomize.py"
  "$root/runtime_patch_dflash_tp124_v30_common/sitecustomize.py"
  "$target/runtime_patch_dflash_tp1_mamba_checkpoint_v2/sitecustomize.py"
  "$target/runtime_patch_dflash_tp1_shortmid_selective_pack_diag/sitecustomize.py"
  "$root/runtime_patch_dflash_tp4_v29_rc3/qwen3_coder_detector_pr25246.py"
  "$root/native_ext/k100_assign_fixed8_i64_v1.so"
)
for file in "${required[@]}"; do
  [[ -s "$file" ]] || { echo "ERROR: missing TP1 v30 asset: $file" >&2; exit 20; }
done
actual_image_id=$(docker image inspect "$image" --format '{{.Id}}' 2>/dev/null || true)
[[ "$actual_image_id" == "$image_id" ]] || {
  echo "ERROR: TP1 verified image mismatch: expected=$image_id actual=$actual_image_id" >&2; exit 21;
}
docker inspect "$name" >/dev/null 2>&1 && { echo "ERROR: container exists: $name" >&2; exit 22; }
ss -ltnH | awk '{print $4}' | grep -Eq "(^|:)${port}$" && { echo "ERROR: port occupied: $port" >&2; exit 23; }
usage=$(/usr/local/hyhal/bin/hy-smi -d 2 --showuse --showmemuse 2>/dev/null || true)
printf '%s\n' "$usage"
grep -q 'HCU use (%): 0.0' <<<"$usage" || { echo 'ERROR: GPU2 compute busy' >&2; exit 24; }
grep -q 'HCU memory use (%): 0' <<<"$usage" || { echo 'ERROR: GPU2 memory busy' >&2; exit 25; }
fuser "$render" >/dev/null 2>&1 && { echo 'ERROR: GPU2 render has users' >&2; exit 26; }

common_mounts=(
  -v "$root/runtime_patch_dflash_tp124_v30_common:$root/runtime_patch_dflash_tp124_v30_common:ro"
  -v "$root/runtime_patch_dflash_tp4_assign_verify_fix_v2:$root/runtime_patch_dflash_tp4_assign_verify_fix_v2:ro"
  -v "$root/runtime_patch_dflash_tp4_retained_alloc_fix_v3:$root/runtime_patch_dflash_tp4_retained_alloc_fix_v3:ro"
  -v "$root/runtime_patch_dflash_tp4_page_reclaim_fix_v4:$root/runtime_patch_dflash_tp4_page_reclaim_fix_v4:ro"
  -v "$root/runtime_patch_dflash_tp4_abort_nocache_v8:$root/runtime_patch_dflash_tp4_abort_nocache_v8:ro"
  -v "$root/runtime_patch_dflash_tp4_verify_stockalloc_v9:$root/runtime_patch_dflash_tp4_verify_stockalloc_v9:ro"
  -v "$root/runtime_patch_dflash_tp4_prepare_native_i64_v29:$root/runtime_patch_dflash_tp4_prepare_native_i64_v29:ro"
  -v "$root/runtime_patch_dflash_tp4_v29_rc2:$root/runtime_patch_dflash_tp4_v29_rc2:ro"
  -v "$root/runtime_patch_dflash_tp4_v29_rc3:$root/runtime_patch_dflash_tp4_v29_rc3:ro"
  -v "$root/native_ext/k100_assign_fixed8_i64_v1.so:$root/native_ext/k100_assign_fixed8_i64_v1.so:ro"
)

docker run -d --name "$name" \
  --network host --ipc host --security-opt label=disable \
  --device /dev/kfd --device "$render" \
  -e PROFILE=tp1 -e HIP_VISIBLE_DEVICES=0 -e PORT="$port" \
  -e SERVED_MODEL_NAME="$served" -e MEM_FRACTION_STATIC=0.84 -e MAX_RUNNING_REQUESTS=1 \
  -e SGLANG_Q38_TP124_V30_DISABLE_COMMON="${SGLANG_Q38_TP124_V30_DISABLE_COMMON:-0}" \
  -e SGLANG_Q38_ASSIGN_FIXED8_I64_SO="$root/native_ext/k100_assign_fixed8_i64_v1.so" \
  -e SGLANG_Q38_QWEN3_CODER_STREAM_STRING_FIX=1 \
  -e SGLANG_Q38_OPENAI_DEFAULT_MAX_TOKENS=8192 \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v /data/my_models/Qwen/Qwen3.8-27B-SmoothQuant-W8A8-INT8:/models/target:ro \
  -v "$root/models/Qwen3.8-27B-DFlash2:/models/draft:ro" \
  -v "$root/scripts/entrypoint_tp1_v30.sh:/opt/qwen38-k100ai/entrypoint.tp1.sh:ro" \
  -v "$root/runtime_patch_dflash_tp1_v30:$root/runtime_patch_dflash_tp1_v30:ro" \
  -v "$target/runtime_patch_dflash_tp1_mamba_checkpoint_v2:$target/runtime_patch_dflash_tp1_mamba_checkpoint_v2:ro" \
  -v "$target/runtime_patch_dflash_tp1_shortmid_selective_pack_diag:$target/runtime_patch_dflash_tp1_shortmid_selective_pack_diag:ro" \
  "${common_mounts[@]}" \
  "$image"

docker inspect -f '{{.State.Status}} restart={{.RestartCount}} oom={{.State.OOMKilled}}' "$name"
