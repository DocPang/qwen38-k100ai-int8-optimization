#!/usr/bin/env bash
# Final TP2 v30 launcher.  TP2 intentionally remains on the verified legacy
# SGLang image: the unified TP1/TP4 image regresses the frozen arithmetic gate.
set -euo pipefail

name=${NAME:-q38-tp2-v30-prod-gpu01}
served=${SERVED_MODEL_NAME:-Qwen3.8-27B-W8A8-DFlash2-TP2}
port=${PORT:-8062}
root=/data/qwen38-dflash2-k100ai
target=/data/qwen38-27b-k100ai-int8-opt
patch=$root/runtime_patch_dflash_tp2_v30
overlay=$root/work/sourcefind_sglang_overlay_tp4/python
control=$target/results/tp2_shortmid_layerdiag_fallback_layers.txt
image=harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-ubuntu22.04-dtk26.04-py3.10-20260620
image_id=sha256:5d6305a6fb1695ebcb3675a7f9b87aca59478aaae21c9eeda3ebb59ddb5f9ad8

required=(
  "$patch/sitecustomize.py"
  "$root/runtime_patch_dflash_tp124_v30_common/sitecustomize.py"
  "$target/runtime_patch_sglang_tp2_longtail_varlen_v1/sitecustomize.py"
  "$target/runtime_patch_sglang_tp2_mamba_checkpoint8k_v1/sitecustomize.py"
  "$target/runtime_patch_sglang_tp2_shortmid_layerdiag_v5/sitecustomize.py"
  "$target/scripts/launch_sglang_require_sitecustomize.py"
  "$target/runtime_assets/qwen38_chat_template.jinja"
  "$root/runtime_patch_dflash_tp4_v29_rc3/qwen3_coder_detector_pr25246.py"
  "$root/native_ext/k100_assign_fixed8_i64_v1.so"
  "$control"
)
for file in "${required[@]}"; do
  [[ -s "$file" ]] || { echo "ERROR: missing TP2 v30 asset: $file" >&2; exit 20; }
done
[[ -d "$overlay" ]] || { echo "ERROR: missing SourceFind SGLang overlay: $overlay" >&2; exit 21; }
[[ "$(tr -d '[:space:]' < "$control")" == 3,23 ]] || {
  echo "ERROR: TP2 v30 requires frozen fallback control 3,23" >&2; exit 22;
}
actual_image_id=$(docker image inspect "$image" --format '{{.Id}}' 2>/dev/null || true)
[[ "$actual_image_id" == "$image_id" ]] || {
  echo "ERROR: TP2 verified image mismatch: expected=$image_id actual=$actual_image_id" >&2; exit 23;
}
docker inspect "$name" >/dev/null 2>&1 && { echo "ERROR: container exists: $name" >&2; exit 24; }
ss -ltnH | awk '{print $4}' | grep -Eq "(^|:)${port}$" && {
  echo "ERROR: port occupied: $port" >&2; exit 25;
}
for gpu in 0 1; do
  usage=$(/usr/local/hyhal/bin/hy-smi -d "$gpu" --showuse --showmemuse 2>/dev/null || true)
  printf '%s\n' "$usage"
  grep -q 'HCU use (%): 0.0' <<<"$usage" || { echo "ERROR: GPU${gpu} compute busy" >&2; exit 26; }
  grep -q 'HCU memory use (%): 0' <<<"$usage" || { echo "ERROR: GPU${gpu} memory busy" >&2; exit 27; }
done
for render in /dev/dri/renderD128 /dev/dri/renderD129; do
  fuser "$render" >/dev/null 2>&1 && { echo "ERROR: $render has users" >&2; exit 28; }
done

warmup_args=(--warmups q38_v30_tp2_shortctx)
if [[ "${SGLANG_Q38_TP124_V30_DISABLE_COMMON:-0}" == 1 ]]; then
  warmup_args=()
elif [[ "${SGLANG_Q38_TP124_V30_DISABLE_COMMON:-0}" != 0 ]]; then
  echo "ERROR: SGLANG_Q38_TP124_V30_DISABLE_COMMON must be 0 or 1" >&2; exit 29
fi

docker run -d --name "$name" \
  --network host --ipc host --shm-size 64m --security-opt label=disable \
  --device /dev/kfd --device /dev/dri/renderD128 --device /dev/dri/renderD129 \
  --mount type=bind,source=/data,target=/data,bind-propagation=rslave \
  --mount type=bind,source=/opt/hyhal,target=/opt/hyhal,readonly \
  --mount type=bind,source=/data/my_models/Qwen/Qwen3.8-27B/preprocessor_config.json,target=/data/my_models/Qwen/Qwen3.8-27B-SmoothQuant-W8A8-INT8/preprocessor_config.json,readonly \
  --mount type=bind,source=/data/my_models/Qwen/Qwen3.8-27B/video_preprocessor_config.json,target=/data/my_models/Qwen/Qwen3.8-27B-SmoothQuant-W8A8-INT8/video_preprocessor_config.json,readonly \
  --mount type=bind,source=$target/cache/torch_extensions_tp2_gpu23,target=/root/.cache/torch_extensions \
  -e HIP_VISIBLE_DEVICES=0,1 -e HSA_FORCE_FINE_GRAIN_PCIE=1 \
  -e PYTHONPATH="$patch:$overlay" \
  -e SGLANG_REQUIRED_SITECUSTOMIZE_PREFIX="$patch" \
  -e SGLANG_Q38_V30_PROFILE=tp2 \
  -e SGLANG_Q38_TP124_V30_DISABLE_COMMON="${SGLANG_Q38_TP124_V30_DISABLE_COMMON:-0}" \
  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  -e SGLANG_KV_LAYOUT_DCU_FA=true \
  -e SGLANG_Q38_COMPACT_HEAD_M1=0 \
  -e SGLANG_Q38_DEEP_DOWN_GEMV_M1=1 \
  -e SGLANG_Q38_DEEP_DOWN_GEMV_SO=$target/native_ext/k100_int8_gemv_deep_v4_sglang.so \
  -e SGLANG_Q38_GDN_BA_FUSED_M1=1 \
  -e SGLANG_Q38_NATIVE_BODY_GEMV_FAMILIES=gate_up,full_qkv \
  -e SGLANG_Q38_NATIVE_BODY_GEMV_M1=1 \
  -e SGLANG_Q38_NATIVE_BODY_GEMV_SO=$target/native_ext/k100_int8_gemv_generic_v2_sglang.so \
  -e SGLANG_Q38_NATIVE_GDN_SPLIT_M1=1 \
  -e SGLANG_Q38_NATIVE_GDN_SPLIT_SO=$target/native_ext/k100_int8_gemv_generic_v2_sglang.so \
  -e SGLANG_Q38_NATIVE_OUT_GEMV_M1=1 \
  -e SGLANG_Q38_NATIVE_OUT_GEMV_SO=$target/native_ext/k100_int8_gemv_v7_sglang.so \
  -e SGLANG_Q38_RMS_GDN_INT8_M1=1 \
  -e SGLANG_Q38_SHARED_LAYER_IDS=32-47 \
  -e SGLANG_Q38_SWIGLU_INT8_M1=1 \
  -e SGLANG_Q38_TP2_COMPACT_HEAD_M1=1 \
  -e SGLANG_Q38_TP2_COMPACT_HEAD_TOPK=1024 \
  -e SGLANG_Q38_TP2_K5120_LDSX_M1=1 \
  -e SGLANG_Q38_TP2_ROW_LDSX_M1=1 \
  -e SGLANG_Q38_TP2_ROW_LDSX_SO=$target/native_ext/k100_int8_gemv_tp2_row_ldsx_v1_sglang.so \
  -e SGLANG_USE_CAUSAL_CONV1D=0 -e SGLANG_USE_FUSED_SILU_MUL_QUANT=1 \
  -e SGLANG_USE_LIGHTOP=0 -e SGLANG_USE_TRITON_VLLM_FA=0 \
  -e SGLANG_Q38_ASSIGN_FIXED8_I64_SO=$root/native_ext/k100_assign_fixed8_i64_v1.so \
  -e SGLANG_Q38_QWEN3_CODER_STREAM_STRING_FIX=1 \
  -e SGLANG_Q38_OPENAI_DEFAULT_MAX_TOKENS=8192 \
  -e TORCHINDUCTOR_CACHE_DIR=$target/cache/torchinductor_tp2_gpu23 \
  -e TRITON_CACHE_DIR=$target/cache/triton_tp2_gpu23 \
  -e TRITON_JSON_DIR=$target/cache/sglang_w8a8_gfx928_tp2_gpu23_rowldsx_v1 \
  -e W8A8_SUPPORT_METHODS=1 -e TZ=Asia/Shanghai \
  "$image" \
  python3 -u "$target/scripts/launch_sglang_require_sitecustomize.py" \
    --model-path /data/my_models/Qwen/Qwen3.8-27B-SmoothQuant-W8A8-INT8 \
    --host 0.0.0.0 --port "$port" --random-seed 0 \
    --served-model-name "$served" "${warmup_args[@]}" \
    --chat-template "$target/runtime_assets/qwen38_chat_template.jinja" \
    --dtype bfloat16 --kv-cache-dtype bfloat16 --tp-size 2 --pp-size 1 \
    --attention-backend fa3 --mm-attention-backend fa3 --page-size 64 \
    --mamba-scheduler-strategy extra_buffer --max-mamba-cache-size 16 --mamba-track-interval 8192 \
    --cuda-graph-bs 1 --disable-piecewise-cuda-graph \
    --context-length 262144 --mem-fraction-static 0.88 \
    --chunked-prefill-size 8192 --max-prefill-tokens 16384 \
    --pack-paged-kv-to-varlen auto \
    --pack-paged-kv-to-varlen-min-kv-tokens 8192 \
    --pack-paged-kv-to-varlen-min-q-tokens 8192 \
    --max-running-requests 4 \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path "$root/models/Qwen3.8-27B-DFlash2" \
    --speculative-draft-model-quantization unquant \
    --speculative-draft-attention-backend triton \
    --speculative-num-steps 1 --speculative-num-draft-tokens 8 \
    --enable-metrics --tool-call-parser qwen3_coder --reasoning-parser qwen3

docker inspect -f '{{.State.Status}} restart={{.RestartCount}} oom={{.State.OOMKilled}} image={{.Image}}' "$name"
