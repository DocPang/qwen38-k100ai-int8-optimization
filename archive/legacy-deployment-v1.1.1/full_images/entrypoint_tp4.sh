#!/usr/bin/env bash
set -euo pipefail

TARGET_ROOT=/data/qwen38-27b-k100ai-int8-opt
DFLASH_ROOT=/data/qwen38-dflash2-k100ai
MODEL_SRC=${MODEL:-/models/target}
DRAFT=${DRAFT_MODEL:-/models/draft}
PORT=${PORT:-8068}
PATCH=$DFLASH_ROOT/runtime_patch_tp4
TRITON_JSON_DIR=${TRITON_JSON_DIR:-$TARGET_ROOT/cache/tp4}

for f in "$MODEL_SRC/config.json" "$MODEL_SRC/tokenizer.json" "$DRAFT/config.json" "$DRAFT/model.safetensors"; do
  [[ -f "$f" ]] || { echo "ERROR: missing required file: $f" >&2; exit 20; }
done
[[ -s "$PATCH/sitecustomize.py" ]] || { echo "ERROR: missing TP4 runtime patch: $PATCH/sitecustomize.py" >&2; exit 21; }
[[ -d "$TRITON_JSON_DIR" ]] || { echo "ERROR: missing TP4 tune cache: $TRITON_JSON_DIR" >&2; exit 22; }

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
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3}
export HSA_FORCE_FINE_GRAIN_PCIE=${HSA_FORCE_FINE_GRAIN_PCIE:-1}
export W8A8_SUPPORT_METHODS=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_KV_LAYOUT_DCU_FA=true
export SGLANG_USE_LIGHTOP=0
export SGLANG_USE_CAUSAL_CONV1D=0
export SGLANG_USE_TRITON_VLLM_FA=0
export TRITON_JSON_DIR
export SGLANG_Q38_COMPACT_HEAD_M1=0
export SGLANG_Q38_TP4_COMPACT_HEAD_M1=1
export SGLANG_Q38_TP4_ROW_LDSX_M1=1
export SGLANG_Q38_TP4_ROW_LDSX_SO="$TARGET_ROOT/native_ext/k100_int8_gemv_tp4_row_ldsx_v1_sglang.so"
export SGLANG_Q38_TP4_K5120_LDSX_M1=1
export SGLANG_Q38_GDN_BA_FUSED_M1=1
export SGLANG_Q38_SWIGLU_INT8_M1=1
export SGLANG_USE_FUSED_SILU_MUL_QUANT=1
export SGLANG_Q38_RMS_GDN_INT8_M1=1
export SGLANG_Q38_TP4_RMS_QKVZ_M1=1
export SGLANG_Q38_TP4_BA24_INT8_M1=1
export SGLANG_Q38_NATIVE_OUT_GEMV_M1=1
export SGLANG_Q38_NATIVE_OUT_GEMV_SO="$TARGET_ROOT/native_ext/k100_int8_gemv_v7_sglang.so"
export SGLANG_Q38_NATIVE_BODY_GEMV_M1=1
export SGLANG_Q38_NATIVE_BODY_GEMV_SO="$TARGET_ROOT/native_ext/k100_int8_gemv_generic_v2_sglang.so"
export SGLANG_Q38_NATIVE_BODY_GEMV_FAMILIES=gate_up,full_qkv
export SGLANG_Q38_NATIVE_GDN_SPLIT_M1=1
export SGLANG_Q38_NATIVE_GDN_SPLIT_SO="$TARGET_ROOT/native_ext/k100_int8_gemv_generic_v2_sglang.so"
export SGLANG_Q38_DEEP_DOWN_GEMV_M1=1
export SGLANG_Q38_DEEP_DOWN_GEMV_SO="$TARGET_ROOT/native_ext/k100_int8_gemv_deep_v4_sglang.so"

export SGLANG_Q38_TP4_U036_KV_LENGTHS=${SGLANG_Q38_TP4_U036_KV_LENGTHS:-16384,24576,32768,40960,49152,57344,65536,73728,81920,90112,98304,106496,114688,122880,131072,139264,147456,155648,163840,172032,180224,188416,196608,204800,212992,221184,229376,237568,245760,253952}
export SGLANG_Q38_TP4_U036_BLOCK_M=${SGLANG_Q38_TP4_U036_BLOCK_M:-64}
export SGLANG_Q38_TP4_U036_PROFILE=${SGLANG_Q38_TP4_U036_PROFILE:-ranklocal_bm64_w4_preloadv}
export SGLANG_Q38_TP4_U036_SPLIT_KV=${SGLANG_Q38_TP4_U036_SPLIT_KV:-4}
export SGLANG_Q38_TP4_Q16_KV_LENGTHS=${SGLANG_Q38_TP4_Q16_KV_LENGTHS:-16384,32768,49152,65536,81920,98304,114688,131072,147456,163840,180224,196608,212992,229376,245760}
export SGLANG_Q38_TP4_Q16_SPLIT_KV=${SGLANG_Q38_TP4_Q16_SPLIT_KV:-4}
export SGLANG_Q38_TP4_Q16_QSPLIT2=${SGLANG_Q38_TP4_Q16_QSPLIT2:-1}
export SGLANG_Q38_TP4_Q16_QSPLIT_KV_EXACT=${SGLANG_Q38_TP4_Q16_QSPLIT_KV_EXACT:-131072}
export SGLANG_Q38_TP4_Q16_QSPLIT_KV_MIN=${SGLANG_Q38_TP4_Q16_QSPLIT_KV_MIN:-0}
export SGLANG_Q38_TP4_Q16_QSPLIT_LAYER_IDS=${SGLANG_Q38_TP4_Q16_QSPLIT_LAYER_IDS:-}
export SGLANG_Q38_TP4_TAIL_SPLIT_8K=${SGLANG_Q38_TP4_TAIL_SPLIT_8K:-1}
export SGLANG_Q38_TP4_TAIL_SPLIT_MIN_PREFIX=${SGLANG_Q38_TP4_TAIL_SPLIT_MIN_PREFIX:-131072}
export SGLANG_Q38_TP4_LONG_CHUNK_8K_PREFIX=${SGLANG_Q38_TP4_LONG_CHUNK_8K_PREFIX:-131072}
export SGLANG_Q38_TP4_QTAIL_KV_LENGTHS=${SGLANG_Q38_TP4_QTAIL_KV_LENGTHS:-257900}
export SGLANG_Q38_TP4_QTAIL_SPLIT_KV=${SGLANG_Q38_TP4_QTAIL_SPLIT_KV:-8}
export SGLANG_DFLASH2_SELECTOR_TOPK_OVERRIDE=${SGLANG_DFLASH2_SELECTOR_TOPK_OVERRIDE:-16}
export SGLANG_DFLASH2_Q8_NATIVE_PAGED=${SGLANG_DFLASH2_Q8_NATIVE_PAGED:-1}
export SGLANG_DFLASH2_DEBUG_STATS=${SGLANG_DFLASH2_DEBUG_STATS:-0}
export SGLANG_DFLASH2_STAGE_SYNC=${SGLANG_DFLASH2_STAGE_SYNC:-0}

ar_args=()
if [[ "${CUSTOM_AR:-1}" == 0 ]]; then
  ar_args=(--disable-custom-all-reduce)
elif [[ "${CUSTOM_AR:-1}" != 1 ]]; then
  echo "ERROR: CUSTOM_AR must be 0 or 1" >&2; exit 23
fi

exec python3 -u "$TARGET_ROOT/scripts/launch_sglang_require_sitecustomize.py" \
  --model-path "$MODEL" \
  --host 0.0.0.0 --port "$PORT" --random-seed 0 \
  --served-model-name Qwen3.8-27B-W8A8-DFlash2-TP4 \
  --chat-template "$DFLASH_ROOT/runtime_assets/qwen38_chat_template.jinja" \
  --dtype bfloat16 --kv-cache-dtype bfloat16 \
  --tp-size 4 --pp-size 1 \
  --attention-backend fa3 --mm-attention-backend fa3 --page-size 64 \
  --mamba-scheduler-strategy extra_buffer --max-mamba-cache-size 16 \
  --cuda-graph-bs 1 --disable-piecewise-cuda-graph "${ar_args[@]}" \
  --context-length 262144 --mem-fraction-static ${MEM_FRACTION_STATIC:-0.90} \
  --chunked-prefill-size 16384 --max-prefill-tokens 16384 \
  --pack-paged-kv-to-varlen auto --pack-paged-kv-to-varlen-min-q-tokens 2048 --pack-paged-kv-to-varlen-min-kv-tokens 8192 \
  --max-total-tokens 1048576 --max-running-requests ${MAX_RUNNING_REQUESTS:-4} \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "$DRAFT" \
  --speculative-draft-model-quantization unquant \
  --speculative-draft-attention-backend triton \
  --speculative-num-steps 1 --speculative-num-draft-tokens 8 \
  --enable-metrics --tool-call-parser qwen3_coder --reasoning-parser qwen3
