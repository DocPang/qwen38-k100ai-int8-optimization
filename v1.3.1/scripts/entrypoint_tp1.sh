#!/usr/bin/env bash
set -euo pipefail

TARGET_ROOT=${TARGET_ROOT:-/data/qwen38-27b-k100ai-int8-opt}
DFLASH_ROOT=${DFLASH_ROOT:-/data/qwen38-dflash2-k100ai}
MODEL_SRC=${MODEL:-/models/target}
DRAFT=${DRAFT_MODEL:-/models/draft}
PORT=${PORT:-8042}
HOST=${HOST:-0.0.0.0}
MODEL_NAME=${SERVED_MODEL_NAME:-Qwen3.8-27B-W8A8-DFlash2-TP1}
PATCH=$DFLASH_ROOT/runtime_patch_dflash_tp1_v30

CONTEXT_LENGTH=${CONTEXT_LENGTH:-262144}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.84}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-1}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-8192}
MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-16384}
PAGE_SIZE=${PAGE_SIZE:-64}
MAMBA_TRACK_INTERVAL=${MAMBA_TRACK_INTERVAL:-8192}
MAX_MAMBA_CACHE_SIZE=${MAX_MAMBA_CACHE_SIZE:-8}
PACK_MIN_Q_TOKENS=${PACK_MIN_Q_TOKENS:-2048}
PACK_MIN_KV_TOKENS=${PACK_MIN_KV_TOKENS:-8192}
SPECULATIVE_NUM_STEPS=${SPECULATIVE_NUM_STEPS:-1}
SPECULATIVE_NUM_DRAFT_TOKENS=${SPECULATIVE_NUM_DRAFT_TOKENS:-8}
RANDOM_SEED=${RANDOM_SEED:-0}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-fa3}
MM_ATTENTION_BACKEND=${MM_ATTENTION_BACKEND:-fa3}
DTYPE=${DTYPE:-bfloat16}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-bfloat16}
TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-qwen3_coder}
REASONING_PARSER=${REASONING_PARSER:-qwen3}

for f in "$MODEL_SRC/config.json" "$MODEL_SRC/tokenizer.json" "$DRAFT/config.json" "$DRAFT/model.safetensors"; do
  [[ -f "$f" ]] || { echo "ERROR: missing required file: $f" >&2; exit 20; }
done
[[ -s "$PATCH/sitecustomize.py" ]] || { echo "ERROR: missing TP1 v30 patch" >&2; exit 21; }
[[ -s "$DFLASH_ROOT/runtime_patch_dflash_tp124_v30_common/sitecustomize.py" ]] || { echo "ERROR: missing TP124 v30 common patch" >&2; exit 22; }
[[ -s "$TARGET_ROOT/runtime_patch_dflash_tp1_mamba_checkpoint_v2/sitecustomize.py" ]] || { echo "ERROR: missing TP1 checkpoint-v2 patch" >&2; exit 23; }

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
export SGLANG_Q38_V30_PROFILE=tp1
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
export SGLANG_Q38_TP1_SHORTMID_SELECTIVE_PACK_MIN_Q=${SGLANG_Q38_TP1_SHORTMID_SELECTIVE_PACK_MIN_Q:-4096}
export SGLANG_Q38_TP1_SHORTMID_SELECTIVE_PACK_MAX_Q=${SGLANG_Q38_TP1_SHORTMID_SELECTIVE_PACK_MAX_Q:-8191}
export SGLANG_Q38_TP1_SHORTMID_FAST_LAYERS=${SGLANG_Q38_TP1_SHORTMID_FAST_LAYERS:-3,7,11,15,19,23,27,31,43,47,51,55,59,63}
export SGLANG_Q38_ASSIGN_FIXED8_I64_SO=${SGLANG_Q38_ASSIGN_FIXED8_I64_SO:-$DFLASH_ROOT/native_ext/k100_assign_fixed8_i64_v1.so}
export SGLANG_Q38_QWEN3_CODER_STREAM_STRING_FIX=${SGLANG_Q38_QWEN3_CODER_STREAM_STRING_FIX:-1}
export SGLANG_Q38_OPENAI_DEFAULT_MAX_TOKENS=${DEFAULT_MAX_TOKENS:-8192}

warmup_args=(--warmups q38_v30_tp1_shortctx)
if [[ "${DISABLE_WARMUP:-0}" == 1 ]]; then
  warmup_args=()
elif [[ "${DISABLE_WARMUP:-0}" != 0 ]]; then
  echo "ERROR: DISABLE_WARMUP must be 0 or 1" >&2; exit 24
fi
if [[ "${SGLANG_Q38_TP124_V30_DISABLE_COMMON:-0}" == 1 ]]; then
  warmup_args=()
elif [[ "${SGLANG_Q38_TP124_V30_DISABLE_COMMON:-0}" != 0 ]]; then
  echo "ERROR: SGLANG_Q38_TP124_V30_DISABLE_COMMON must be 0 or 1" >&2; exit 25
fi

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
  --model-path "$MODEL_LINK" --host "$HOST" --port "$PORT" --random-seed "$RANDOM_SEED"
  --served-model-name "$MODEL_NAME" "${warmup_args[@]}"
  --chat-template "$DFLASH_ROOT/runtime_assets/qwen38_chat_template.jinja"
  --reasoning-parser "$REASONING_PARSER" --tool-call-parser "$TOOL_CALL_PARSER"
  --dtype "$DTYPE" --kv-cache-dtype "$KV_CACHE_DTYPE" --tp-size 1 --pp-size 1
  --attention-backend "$ATTENTION_BACKEND" --mm-attention-backend "$MM_ATTENTION_BACKEND" --page-size "$PAGE_SIZE"
  --mamba-scheduler-strategy extra_buffer --max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE"
  --mamba-track-interval "$MAMBA_TRACK_INTERVAL"
  --cuda-graph-bs 1 --disable-piecewise-cuda-graph
  --context-length "$CONTEXT_LENGTH" --mem-fraction-static "$MEM_FRACTION_STATIC"
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" --max-prefill-tokens "$MAX_PREFILL_TOKENS"
  --pack-paged-kv-to-varlen auto
  --pack-paged-kv-to-varlen-min-q-tokens "$PACK_MIN_Q_TOKENS"
  --pack-paged-kv-to-varlen-min-kv-tokens "$PACK_MIN_KV_TOKENS"
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --speculative-algorithm DFLASH
  --speculative-draft-model-path "$DRAFT"
  --speculative-draft-model-quantization unquant
  --speculative-draft-attention-backend triton
  --speculative-num-steps "$SPECULATIVE_NUM_STEPS"
  --speculative-num-draft-tokens "$SPECULATIVE_NUM_DRAFT_TOKENS"
  --enable-metrics
)
cmd+=("${extra_args[@]}")

printf 'Launching TP1 SGLang:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
