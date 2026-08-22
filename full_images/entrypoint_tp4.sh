#!/usr/bin/env bash
set -euo pipefail

MODEL_SRC=${MODEL:-/models/target}
DRAFT=${DRAFT_MODEL:-/models/draft}
PORT=${PORT:-8068}

for f in "$MODEL_SRC/config.json" "$MODEL_SRC/tokenizer.json" "$DRAFT/config.json" "$DRAFT/model.safetensors"; do
  [[ -f "$f" ]] || { echo "ERROR: missing required file: $f" >&2; exit 20; }
done

MODEL=/tmp/q38-target-model
mkdir -p "$MODEL"
for f in "$MODEL_SRC"/*; do
  [[ -e "$f" ]] || continue
  ln -sfn "$f" "$MODEL/$(basename "$f")"
done
ln -sfn /opt/qwen38-k100ai/model_overrides/preprocessor_config.json "$MODEL/preprocessor_config.json"
ln -sfn /opt/qwen38-k100ai/model_overrides/video_preprocessor_config.json "$MODEL/video_preprocessor_config.json"

export PYTHONPATH=/data/qwen38-dflash2-k100ai/runtime_patch_dflash_tp4_q16k_agent128k_v1:/data/qwen38-dflash2-k100ai/work/sourcefind_sglang_overlay_tp4/python
export SGLANG_REQUIRED_SITECUSTOMIZE_PREFIX=/data/qwen38-dflash2-k100ai/runtime_patch_dflash_tp4_q16k_agent128k_v1
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3}
export HSA_FORCE_FINE_GRAIN_PCIE=1
export W8A8_SUPPORT_METHODS=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_KV_LAYOUT_DCU_FA=true
export SGLANG_USE_LIGHTOP=0
export SGLANG_USE_CAUSAL_CONV1D=0
export SGLANG_USE_TRITON_VLLM_FA=0
export SGLANG_USE_FUSED_SILU_MUL_QUANT=1
export SGLANG_Q38_COMPACT_HEAD_M1=0
export SGLANG_Q38_TP4_COMPACT_HEAD_M1=1
export SGLANG_Q38_TP4_ROW_LDSX_M1=1
export SGLANG_Q38_TP4_ROW_LDSX_SO=/data/qwen38-27b-k100ai-int8-opt/native_ext/k100_int8_gemv_tp4_row_ldsx_v1_sglang.so
export SGLANG_Q38_TP4_K5120_LDSX_M1=1
export SGLANG_Q38_GDN_BA_FUSED_M1=1
export SGLANG_Q38_SWIGLU_INT8_M1=1
export SGLANG_Q38_RMS_GDN_INT8_M1=1
export SGLANG_Q38_TP4_RMS_QKVZ_M1=1
export SGLANG_Q38_TP4_BA24_INT8_M1=1
export SGLANG_Q38_NATIVE_OUT_GEMV_M1=1
export SGLANG_Q38_NATIVE_OUT_GEMV_SO=/data/qwen38-27b-k100ai-int8-opt/native_ext/k100_int8_gemv_v7_sglang.so
export SGLANG_Q38_NATIVE_BODY_GEMV_M1=1
export SGLANG_Q38_NATIVE_BODY_GEMV_SO=/data/qwen38-27b-k100ai-int8-opt/native_ext/k100_int8_gemv_generic_v2_sglang.so
export SGLANG_Q38_NATIVE_BODY_GEMV_FAMILIES=gate_up,full_qkv
export SGLANG_Q38_NATIVE_GDN_SPLIT_M1=1
export SGLANG_Q38_NATIVE_GDN_SPLIT_SO=/data/qwen38-27b-k100ai-int8-opt/native_ext/k100_int8_gemv_generic_v2_sglang.so
export SGLANG_Q38_DEEP_DOWN_GEMV_M1=1
export SGLANG_Q38_DEEP_DOWN_GEMV_SO=/data/qwen38-27b-k100ai-int8-opt/native_ext/k100_int8_gemv_deep_v4_sglang.so
export SGLANG_Q38_TP4_U036_BLOCK_M=64
export SGLANG_Q38_TP4_U036_PROFILE=ranklocal_bm64_w4_preloadv
export SGLANG_Q38_TP4_U036_SPLIT_KV=4
export SGLANG_Q38_TP4_U036_KV_LENGTHS='16384,24576,32768,40960,49152,57344,65536,73728,81920,90112,98304,106496,114688,122880,131072,139264,147456,155648,163840,172032,180224,188416,196608,204800,212992,221184,229376,237568,245760,253952'
export SGLANG_Q38_TP4_Q16_SPLIT_KV=4
export SGLANG_Q38_TP4_Q16_KV_LENGTHS='16384,32768,49152,65536,81920,98304,114688,131072,147456,163840,180224,196608,212992,229376,245760'
export SGLANG_Q38_TP4_Q16_QSPLIT2=1
export SGLANG_Q38_TP4_Q16_QSPLIT_KV_EXACT=131072
export SGLANG_Q38_TP4_Q16_QSPLIT_KV_MIN=0
export SGLANG_Q38_TP4_Q16_QSPLIT_LAYER_IDS=''
export SGLANG_Q38_TP4_TAIL_SPLIT_8K=1
export SGLANG_Q38_TP4_TAIL_SPLIT_MIN_PREFIX=131072
export SGLANG_Q38_TP4_LONG_CHUNK_8K_PREFIX=131072
export SGLANG_Q38_TP4_QTAIL_KV_LENGTHS=257900
export SGLANG_Q38_TP4_QTAIL_SPLIT_KV=8
export SGLANG_DFLASH2_Q8_NATIVE_PAGED=1
export SGLANG_DFLASH2_SELECTOR_TOPK_OVERRIDE=16
export SGLANG_DFLASH2_DEBUG_STATS=${SGLANG_DFLASH2_DEBUG_STATS:-0}
export SGLANG_DFLASH2_STAGE_SYNC=${SGLANG_DFLASH2_STAGE_SYNC:-0}
export SGLANG_DFLASH2_PROFILE_PREFILL=${SGLANG_DFLASH2_PROFILE_PREFILL:-0}
export TRITON_JSON_DIR=/data/qwen38-27b-k100ai-int8-opt/cache/sglang_w8a8_gfx928_tp4_gpu4567_v4_longtail
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/q38-tp4-torchinductor}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/q38-tp4-triton}
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

exec python3 -u /data/qwen38-dflash2-k100ai/runtime_assets/launch_sglang_require_sitecustomize.py \
  --model-path "$MODEL" \
  --host 0.0.0.0 --port "$PORT" --random-seed 0 \
  --served-model-name Qwen3.8-27B-W8A8-DFlash2-TP4-LongCtx-v8 \
  --chat-template /data/qwen38-dflash2-k100ai/runtime_assets/qwen38_chat_template.jinja \
  --dtype bfloat16 --kv-cache-dtype bfloat16 \
  --tp-size 4 --pp-size 1 \
  --attention-backend fa3 --mm-attention-backend fa3 \
  --page-size 64 --mamba-scheduler-strategy extra_buffer --max-mamba-cache-size 16 \
  --cuda-graph-bs 1 --disable-piecewise-cuda-graph \
  --context-length 262144 --mem-fraction-static ${MEM_FRACTION_STATIC:-0.90} \
  --chunked-prefill-size 16384 --max-prefill-tokens 16384 \
  --pack-paged-kv-to-varlen auto --pack-paged-kv-to-varlen-min-q-tokens 2048 --pack-paged-kv-to-varlen-min-kv-tokens 8192 \
  --max-total-tokens 1048576 --max-running-requests 4 \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "$DRAFT" \
  --speculative-draft-model-quantization unquant \
  --speculative-draft-attention-backend triton \
  --speculative-num-steps 1 --speculative-num-draft-tokens 8 \
  --enable-metrics --tool-call-parser qwen3_coder --reasoning-parser qwen3
