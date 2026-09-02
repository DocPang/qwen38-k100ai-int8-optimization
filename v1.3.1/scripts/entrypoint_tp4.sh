#!/usr/bin/env bash
set -euo pipefail

TARGET_ROOT=${TARGET_ROOT:-/data/qwen38-27b-k100ai-int8-opt}
DFLASH_ROOT=${DFLASH_ROOT:-/data/qwen38-dflash2-k100ai}
MODEL_SRC=${MODEL:-/models/target}
DRAFT=${DRAFT_MODEL:-/models/draft}
PORT=${PORT:-8068}
HOST=${HOST:-0.0.0.0}
MODEL_NAME=${SERVED_MODEL_NAME:-Qwen3.8-27B-W8A8-DFlash2-TP4}
PATCH=$DFLASH_ROOT/runtime_patch_dflash_tp4_v30
TRITON_JSON_DIR=${TRITON_JSON_DIR:-$TARGET_ROOT/cache/tp4}

CONTEXT_LENGTH=${CONTEXT_LENGTH:-262144}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.95}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-8}
PP_MAX_MICRO_BATCH_SIZE=${PP_MAX_MICRO_BATCH_SIZE:-8}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-16384}
MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-16384}
PAGE_SIZE=${PAGE_SIZE:-64}
MAMBA_TRACK_INTERVAL=${MAMBA_TRACK_INTERVAL:-16384}
MAX_MAMBA_CACHE_SIZE=${MAX_MAMBA_CACHE_SIZE:-32}
PACK_MIN_Q_TOKENS=${PACK_MIN_Q_TOKENS:-2048}
PACK_MIN_KV_TOKENS=${PACK_MIN_KV_TOKENS:-2048}
SPECULATIVE_NUM_STEPS=${SPECULATIVE_NUM_STEPS:-1}
SPECULATIVE_NUM_DRAFT_TOKENS=${SPECULATIVE_NUM_DRAFT_TOKENS:-8}
RANDOM_SEED=${RANDOM_SEED:-0}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-fa3}
MM_ATTENTION_BACKEND=${MM_ATTENTION_BACKEND:-fa3}
DTYPE=${DTYPE:-bfloat16}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-bfloat16}
TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-qwen3_coder}
REASONING_PARSER=${REASONING_PARSER:-qwen3}
CUDA_GRAPH_BS=${CUDA_GRAPH_BS:-"1 2 3 4 5 6 7 8"}

for f in "$MODEL_SRC/config.json" "$MODEL_SRC/tokenizer.json" "$DRAFT/config.json" "$DRAFT/model.safetensors"; do
  [[ -f "$f" ]] || { echo "ERROR: missing required file: $f" >&2; exit 20; }
done
[[ -s "$PATCH/sitecustomize.py" ]] || { echo "ERROR: missing TP4 runtime patch: $PATCH/sitecustomize.py" >&2; exit 21; }
[[ -d "$TRITON_JSON_DIR" ]] || { echo "ERROR: missing TP4 tune cache: $TRITON_JSON_DIR" >&2; exit 22; }

MODEL_LINK=/tmp/q38-target-model
mkdir -p "$MODEL_LINK"
for f in "$MODEL_SRC"/*; do
  [[ -e "$f" ]] || continue
  ln -sfn "$f" "$MODEL_LINK/$(basename "$f")"
done
ln -sfn /opt/qwen38-k100ai/model_overrides/preprocessor_config.json "$MODEL_LINK/preprocessor_config.json"
ln -sfn /opt/qwen38-k100ai/model_overrides/video_preprocessor_config.json "$MODEL_LINK/video_preprocessor_config.json"

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
export SGLANG_Q38_TP4_SHORTQ_ONEPASS_MIN_Q=${SGLANG_Q38_TP4_SHORTQ_ONEPASS_MIN_Q:-2048}
export SGLANG_Q38_TP4_SHORTQ_ONEPASS_MAX_Q=${SGLANG_Q38_TP4_SHORTQ_ONEPASS_MAX_Q:-4095}
export SGLANG_Q38_TP4_SHORTQ_PARENT_LAYERS=${SGLANG_Q38_TP4_SHORTQ_PARENT_LAYERS:-63}
export SGLANG_DFLASH2_SELECTOR_TOPK_OVERRIDE=${SGLANG_DFLASH2_SELECTOR_TOPK_OVERRIDE:-16}
export SGLANG_DFLASH2_Q8_NATIVE_PAGED=${SGLANG_DFLASH2_Q8_NATIVE_PAGED:-1}
export SGLANG_DFLASH2_DEBUG_STATS=${SGLANG_DFLASH2_DEBUG_STATS:-0}
export SGLANG_DFLASH2_STAGE_SYNC=${SGLANG_DFLASH2_STAGE_SYNC:-0}
export SGLANG_Q38_OPENAI_DEFAULT_MAX_TOKENS=${DEFAULT_MAX_TOKENS:-8192}

ar_args=()
if [[ "${CUSTOM_AR:-1}" == 0 ]]; then
  ar_args=(--disable-custom-all-reduce)
elif [[ "${CUSTOM_AR:-1}" != 1 ]]; then
  echo "ERROR: CUSTOM_AR must be 0 or 1" >&2; exit 23
fi

deterministic_args=()
if [[ "${ENABLE_DETERMINISTIC_INFERENCE:-0}" == 1 ]]; then
  deterministic_args=(--enable-deterministic-inference)
elif [[ "${ENABLE_DETERMINISTIC_INFERENCE:-0}" != 0 ]]; then
  echo "ERROR: ENABLE_DETERMINISTIC_INFERENCE must be 0 or 1" >&2; exit 24
fi

spec_args=(
  --speculative-algorithm DFLASH
  --speculative-draft-model-path "$DRAFT"
  --speculative-draft-model-quantization unquant
  --speculative-draft-attention-backend triton
  --speculative-num-steps "$SPECULATIVE_NUM_STEPS"
  --speculative-num-draft-tokens "$SPECULATIVE_NUM_DRAFT_TOKENS"
)
if [[ "${TARGET_ONLY_EXPERIMENT:-0}" == 1 ]]; then
  spec_args=()
elif [[ "${TARGET_ONLY_EXPERIMENT:-0}" != 0 ]]; then
  echo "ERROR: TARGET_ONLY_EXPERIMENT must be 0 or 1" >&2; exit 25
fi

warmup_args=(--warmups q38_v30_shortctx)
if [[ "${DISABLE_WARMUP:-0}" == 1 ]]; then
  warmup_args=()
elif [[ "${DISABLE_WARMUP:-0}" != 0 ]]; then
  echo "ERROR: DISABLE_WARMUP must be 0 or 1" >&2; exit 26
fi

read -r -a cuda_graph_bs <<<"$CUDA_GRAPH_BS"
((${#cuda_graph_bs[@]} > 0)) || { echo "ERROR: CUDA_GRAPH_BS cannot be empty" >&2; exit 27; }

extra_args=()
if [[ -n "${SGLANG_EXTRA_ARGS_B64:-}" ]]; then
  extra_args_tmp=$(mktemp /tmp/q38-extra-args.XXXXXX)
  trap 'rm -f "$extra_args_tmp"' EXIT
  if ! python3 - >"$extra_args_tmp" <<'PY'
import base64, json, os, sys
try:
    raw = base64.b64decode(os.environ["SGLANG_EXTRA_ARGS_B64"], validate=True)
    args = json.loads(raw.decode("utf-8"))
except Exception as exc:
    raise SystemExit(f"invalid SGLANG_EXTRA_ARGS_B64: {exc}")
if not isinstance(args, list) or not all(isinstance(x, str) for x in args):
    raise SystemExit("SGLANG_EXTRA_ARGS_B64 must decode to a JSON string array")
for arg in args:
    if "\x00" in arg:
        raise SystemExit("NUL is not allowed in SGLang arguments")
    sys.stdout.buffer.write(arg.encode("utf-8") + b"\0")
PY
  then
    echo "ERROR: failed to decode SGLang extra args" >&2
    exit 28
  fi
  mapfile -d '' -t extra_args <"$extra_args_tmp"
  rm -f "$extra_args_tmp"
  trap - EXIT
fi

cmd=(
  python3 -u "$TARGET_ROOT/scripts/launch_sglang_require_sitecustomize.py"
  --model-path "$MODEL_LINK"
  --host "$HOST" --port "$PORT" --random-seed "$RANDOM_SEED"
  --served-model-name "$MODEL_NAME" "${warmup_args[@]}"
  --chat-template "$DFLASH_ROOT/runtime_assets/qwen38_chat_template.jinja"
  --dtype "$DTYPE" --kv-cache-dtype "$KV_CACHE_DTYPE"
  --tp-size 4 --pp-size 1
  --attention-backend "$ATTENTION_BACKEND" --mm-attention-backend "$MM_ATTENTION_BACKEND" --page-size "$PAGE_SIZE"
  --mamba-scheduler-strategy extra_buffer --max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE"
  --mamba-track-interval "$MAMBA_TRACK_INTERVAL"
  --cuda-graph-bs "${cuda_graph_bs[@]}" --disable-piecewise-cuda-graph
  "${ar_args[@]}" "${deterministic_args[@]}"
  --context-length "$CONTEXT_LENGTH" --mem-fraction-static "$MEM_FRACTION_STATIC"
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" --max-prefill-tokens "$MAX_PREFILL_TOKENS"
  --pack-paged-kv-to-varlen auto
  --pack-paged-kv-to-varlen-min-q-tokens "$PACK_MIN_Q_TOKENS"
  --pack-paged-kv-to-varlen-min-kv-tokens "$PACK_MIN_KV_TOKENS"
  --max-total-tokens "${MAX_TOTAL_TOKENS:-1048576}"
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --pp-max-micro-batch-size "$PP_MAX_MICRO_BATCH_SIZE"
  "${spec_args[@]}"
  --enable-metrics --tool-call-parser "$TOOL_CALL_PARSER" --reasoning-parser "$REASONING_PARSER"
)
cmd+=("${extra_args[@]}")

printf 'Launching TP4 SGLang:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
