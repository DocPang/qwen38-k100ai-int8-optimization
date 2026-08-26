#!/usr/bin/env bash
set -euo pipefail

TARGET_ROOT=/data/qwen38-27b-k100ai-int8-opt
DFLASH_ROOT=/data/qwen38-dflash2-k100ai
MODEL_SRC=${MODEL:-/models/target}
DRAFT=${DRAFT_MODEL:-/models/draft}
PORT=${PORT:-8090}
PATCH=$TARGET_ROOT/runtime_patch_tp1

for f in "$MODEL_SRC/config.json" "$MODEL_SRC/tokenizer.json" "$DRAFT/config.json" "$DRAFT/model.safetensors"; do
  [[ -f "$f" ]] || { echo "ERROR: missing required file: $f" >&2; exit 20; }
done
[[ -s "$PATCH/sitecustomize.py" ]] || { echo "ERROR: missing TP1 runtime patch: $PATCH/sitecustomize.py" >&2; exit 21; }

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
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
export HSA_FORCE_FINE_GRAIN_PCIE=${HSA_FORCE_FINE_GRAIN_PCIE:-1}
export W8A8_SUPPORT_METHODS=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_KV_LAYOUT_DCU_FA=true
export SGLANG_USE_TRITON_VLLM_FA=0
export SGLANG_USE_LIGHTOP=0
export SGLANG_USE_CAUSAL_CONV1D=0
export SGLANG_USE_FUSED_SILU_MUL_QUANT=1
export SGLANG_Q38_COMPACT_HEAD_M1=1
export SGLANG_Q38_COMPACT_HEAD_TOPK=512
export SGLANG_Q38_DEEP_DOWN_GEMV_M1=1
export SGLANG_Q38_DEEP_DOWN_GEMV_SO="$TARGET_ROOT/native_ext/k100_int8_gemv_deep_v4_sglang.so"
export SGLANG_Q38_GDN_BA_FUSED_M1=1
export SGLANG_Q38_NATIVE_BODY_GEMV_FAMILIES=gate_up,full_qkv
export SGLANG_Q38_NATIVE_BODY_GEMV_M1=1
export SGLANG_Q38_NATIVE_BODY_GEMV_SO="$TARGET_ROOT/native_ext/k100_int8_gemv_generic_v2_sglang.so"
export SGLANG_Q38_NATIVE_GDN_SPLIT_M1=1
export SGLANG_Q38_NATIVE_GDN_SPLIT_SO="$TARGET_ROOT/native_ext/k100_int8_gemv_generic_v2_sglang.so"
export SGLANG_Q38_NATIVE_OUT_GEMV_M1=1
export SGLANG_Q38_NATIVE_OUT_GEMV_SO="$TARGET_ROOT/native_ext/k100_int8_gemv_v7_sglang.so"
export SGLANG_Q38_RMS_GDN_INT8_M1=1
export SGLANG_Q38_SWIGLU_INT8_M1=1
export SGLANG_Q38_U036_KV_LENGTHS=${SGLANG_Q38_U036_KV_LENGTHS:-16384,24576,32768,40960,49152,57344,65536,73728,81920,90112,98304,106496,114688,122880,131072,139264}
export SGLANG_DFLASH_EARLY_TRITON_LAYERS=${SGLANG_DFLASH_EARLY_TRITON_LAYERS:-7,15}
export SGLANG_DFLASH_EARLY_TRITON_ROUNDS=${SGLANG_DFLASH_EARLY_TRITON_ROUNDS:-1}

exec python3 -u "$TARGET_ROOT/scripts/launch_sglang_require_sitecustomize.py" \
  --model-path "$MODEL" \
  --host 0.0.0.0 --port "$PORT" --random-seed 0 \
  --served-model-name Qwen3.8-27B-W8A8-DFlash2-TP1 \
  --chat-template "$DFLASH_ROOT/runtime_assets/qwen38_chat_template.jinja" \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --dtype bfloat16 --kv-cache-dtype bfloat16 \
  --tp-size 1 --pp-size 1 \
  --attention-backend fa3 --mm-attention-backend fa3 --page-size 64 \
  --mamba-scheduler-strategy extra_buffer --max-mamba-cache-size 8 \
  --cuda-graph-bs 1 --disable-piecewise-cuda-graph \
  --context-length 262144 --mem-fraction-static ${MEM_FRACTION_STATIC:-0.84} \
  --chunked-prefill-size 8192 --max-prefill-tokens 16384 \
  --pack-paged-kv-to-varlen auto --pack-paged-kv-to-varlen-min-q-tokens 2048 --pack-paged-kv-to-varlen-min-kv-tokens 8192 \
  --max-running-requests ${MAX_RUNNING_REQUESTS:-1} \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "$DRAFT" \
  --speculative-draft-model-quantization unquant \
  --speculative-draft-attention-backend triton \
  --speculative-num-steps 1 --speculative-num-draft-tokens 8 \
  --enable-metrics
