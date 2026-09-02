#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=_common.sh
source "$script_dir/_common.sh"
q38_collect_sglang_args "$@"

IMAGE=${IMAGE:-${TP1_IMAGE:-$Q38_FINAL_IMAGE_DEFAULT}}
MODEL_PATH=${MODEL_PATH:-${TARGET_MODEL:-}}
DRAFT_MODEL_PATH=${DRAFT_MODEL_PATH:-${DRAFT_MODEL:-}}
NAME=${NAME:-qwen38-tp1}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-Qwen3.8-27B-W8A8-DFlash2-TP1}
PORT=${PORT:-8042}
HOST_GPU_ID=${HOST_GPU_ID:-0}
RENDER_DEVICE=${RENDER_DEVICE:-/dev/dri/renderD128}

for cmd in docker awk grep; do q38_require_cmd "$cmd"; done
[[ -n "$MODEL_PATH" ]] || q38_die "set TARGET_MODEL or MODEL_PATH"
[[ -n "$DRAFT_MODEL_PATH" ]] || q38_die "set DRAFT_MODEL or DRAFT_MODEL_PATH"
q38_check_weights "$MODEL_PATH" "$DRAFT_MODEL_PATH"
q38_check_image "$IMAGE"
q38_check_container_absent "$NAME"
q38_check_port_free "$PORT"
q38_check_gpu_idle "$HOST_GPU_ID"
q38_check_render_idle "$RENDER_DEVICE"
q38_print_extra_args

echo "Profile: TP1"
echo "Image: $IMAGE"
echo "Container: $NAME"
echo "Model: $SERVED_MODEL_NAME"
echo "Host GPU: $HOST_GPU_ID ($RENDER_DEVICE)"
echo "Port: $PORT"

docker_args=(
  run -d --name "$NAME"
  --restart unless-stopped
  --network host --ipc host
  --security-opt label=disable
  --device /dev/kfd:/dev/kfd
  --device "$RENDER_DEVICE:$RENDER_DEVICE"
  -v /opt/hyhal:/opt/hyhal:ro
  -v "$MODEL_PATH:/models/target:ro"
  -v "$DRAFT_MODEL_PATH:/models/draft:ro"
  -e PROFILE=tp1
  -e HIP_VISIBLE_DEVICES=0
  -e PORT="$PORT"
  -e SERVED_MODEL_NAME="$SERVED_MODEL_NAME"
  -e CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
  -e MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.84}"
  -e MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}"
  -e CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
  -e MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-16384}"
  -e MAMBA_TRACK_INTERVAL="${MAMBA_TRACK_INTERVAL:-8192}"
  -e MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-8}"
  -e SPECULATIVE_NUM_DRAFT_TOKENS="${SPECULATIVE_NUM_DRAFT_TOKENS:-8}"
  -e SGLANG_EMPTY_CACHE_INTERVAL="${SGLANG_EMPTY_CACHE_INTERVAL:-60}"
  -e TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
  -e REASONING_PARSER="${REASONING_PARSER:-qwen3}"
  "$IMAGE"
)
if ((${#Q38_SGLANG_ARGS[@]})); then
  docker_args+=(-- "${Q38_SGLANG_ARGS[@]}")
fi

docker "${docker_args[@]}"
docker inspect -f '{{.State.Status}} restart={{.RestartCount}} oom={{.State.OOMKilled}} image={{.Image}}' "$NAME"
