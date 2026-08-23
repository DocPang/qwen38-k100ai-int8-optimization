#!/usr/bin/env bash
set -euo pipefail

TARGET_ROOT=/data/qwen38-27b-k100ai-int8-opt
DFLASH_ROOT=/data/qwen38-dflash2-k100ai
MODEL_SRC=${MODEL:-/models/target}
DRAFT=${DRAFT_MODEL:-/models/draft}
PORT=${PORT:-8062}
PATCH=$TARGET_ROOT/runtime_patch_tp2
TRITON_JSON_DIR=${TRITON_JSON_DIR:-$TARGET_ROOT/cache/tp2}

for f in "$MODEL_SRC/config.json" "$MODEL_SRC/tokenizer.json" "$DRAFT/config.json" "$DRAFT/model.safetensors"; do
  [[ -f "$f" ]] || { echo "ERROR: missing required file: $f" >&2; exit 20; }
done
[[ -s "$PATCH/sitecustomize.py" ]] || { echo "ERROR: missing TP2 runtime patch: $PATCH/sitecustomize.py" >&2; exit 21; }
[[ -d "$TRITON_JSON_DIR" ]] || { echo "ERROR: missing TP2 tune cache: $TRITON_JSON_DIR" >&2; exit 22; }

MODEL=/tmp/q38-target-model
mkdir -p "$MODEL"
for f in "$MODEL_SRC"/*; do
  [[ -e "$f" ]] || continue
  ln -sfn "$f" "$MODEL/$(basename "$f")"
done
ln -sfn /opt/qwen38-k100ai/model_overrides/preprocessor_config.json "$MODEL/preprocessor_config.json"
ln -sfn /opt/qwen38-k100ai/model_overrides/video_preprocessor_config.json "$MODEL/video_preprocessor_config.json"

export PYTHONPATH="$PATCH${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_REQUIRED_SITECUSTOMIZE_PREFIX="$PATCH"
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1}
export HSA_FORCE_FINE_GRAIN_PCIE=${HSA_FORCE_FINE_GRAIN_PCIE:-1}
export W8A8_SUPPORT_METHODS=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_KV_LAYOUT_DCU_FA=true
export SGLANG_USE_TRITON_VLLM_FA=0
export SGLANG_USE_LIGHTOP=0
export SGLANG_USE_CAUSAL_CONV1D=0
export TRITON_JSON_DIR
export SGLANG_Q38_COMPACT_HEAD_M1=0
export SGLANG_Q38_TP2_COMPACT_HEAD_M1=1
export SGLANG_Q38_TP2_COMPACT_HEAD_TOPK=1024
export SGLANG_Q38_TP2_K5120_LDSX_M1=1
export SGLANG_Q38_TP2_ROW_LDSX_M1=1
export SGLANG_Q38_TP2_ROW_LDSX_SO="$TARGET_ROOT/native_ext/k100_int8_gemv_tp2_row_ldsx_v1_sglang.so"
export SGLANG_Q38_SHARED_LAYER_IDS=${SGLANG_Q38_SHARED_LAYER_IDS:-32-47}
export SGLANG_Q38_GDN_BA_FUSED_M1=1
export SGLANG_Q38_SWIGLU_INT8_M1=1
export SGLANG_USE_FUSED_SILU_MUL_QUANT=1
export SGLANG_Q38_RMS_GDN_INT8_M1=1
export SGLANG_Q38_NATIVE_OUT_GEMV_M1=1
export SGLANG_Q38_NATIVE_OUT_GEMV_SO="$TARGET_ROOT/native_ext/k100_int8_gemv_v7_sglang.so"
export SGLANG_Q38_NATIVE_BODY_GEMV_M1=1
export SGLANG_Q38_NATIVE_BODY_GEMV_SO="$TARGET_ROOT/native_ext/k100_int8_gemv_generic_v2_sglang.so"
export SGLANG_Q38_NATIVE_BODY_GEMV_FAMILIES=gate_up,full_qkv
export SGLANG_Q38_NATIVE_GDN_SPLIT_M1=1
export SGLANG_Q38_NATIVE_GDN_SPLIT_SO="$TARGET_ROOT/native_ext/k100_int8_gemv_generic_v2_sglang.so"
export SGLANG_Q38_DEEP_DOWN_GEMV_M1=1
export SGLANG_Q38_DEEP_DOWN_GEMV_SO="$TARGET_ROOT/native_ext/k100_int8_gemv_deep_v4_sglang.so"

ar_args=()
if [[ "${CUSTOM_AR:-1}" == 0 ]]; then
  ar_args=(--disable-custom-all-reduce)
elif [[ "${CUSTOM_AR:-1}" != 1 ]]; then
  echo "ERROR: CUSTOM_AR must be 0 or 1" >&2; exit 23
fi
if [[ "${P2P:-1}" == 0 ]]; then
  export NCCL_P2P_DISABLE=1
elif [[ "${P2P:-1}" != 1 ]]; then
  echo "ERROR: P2P must be 0 or 1" >&2; exit 24
fi

exec python3 -u "$TARGET_ROOT/scripts/launch_sglang_require_sitecustomize.py" \
  --model-path "$MODEL" \
  --host 0.0.0.0 --port "$PORT" --random-seed 0 \
  --served-model-name Qwen3.8-27B-W8A8-DFlash2-TP2 \
  --chat-template "$DFLASH_ROOT/runtime_assets/qwen38_chat_template.jinja" \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --dtype bfloat16 --kv-cache-dtype bfloat16 \
  --tp-size 2 --pp-size 1 \
  --attention-backend fa3 --mm-attention-backend fa3 --page-size 64 \
  --mamba-scheduler-strategy extra_buffer --max-mamba-cache-size 16 \
  --cuda-graph-bs 1 --disable-piecewise-cuda-graph "${ar_args[@]}" \
  --context-length 262144 --mem-fraction-static ${MEM_FRACTION_STATIC:-0.88} \
  --chunked-prefill-size 8192 --max-prefill-tokens 16384 \
  --pack-paged-kv-to-varlen auto --pack-paged-kv-to-varlen-min-q-tokens 8192 --pack-paged-kv-to-varlen-min-kv-tokens 8192 \
  --max-running-requests ${MAX_RUNNING_REQUESTS:-4} \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "$DRAFT" \
  --speculative-draft-model-quantization unquant \
  --speculative-draft-attention-backend triton \
  --speculative-num-steps 1 --speculative-num-draft-tokens 8 \
  --enable-metrics
