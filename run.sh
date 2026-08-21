#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENV_FILE=${ENV_FILE:-$ROOT/.env}
[[ -f "$ENV_FILE" ]] || { echo "ERROR: missing $ENV_FILE; copy .env.example to .env first" >&2; exit 2; }
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

IMAGE_TAG=${IMAGE_TAG:-qwen38-w8a8-k100ai-dflash2-tp4:local}
CONTAINER_NAME=${CONTAINER_NAME:-qwen38-w8a8-k100ai-dflash2-tp4}
PORT=${PORT:-8068}
U036_PROFILE=${U036_PROFILE:-ranklocal_bm64_w4_preloadv}
U036_SPLIT_KV=${U036_SPLIT_KV:-4}
CUSTOM_AR=${CUSTOM_AR:-1}

for v in TARGET_MODEL DRAFT_MODEL RENDER0 RENDER1 RENDER2 RENDER3; do
  [[ -n "${!v:-}" ]] || { echo "ERROR: $v is not set in $ENV_FILE" >&2; exit 3; }
done
[[ -d "$TARGET_MODEL" ]] || { echo "ERROR: target model directory not found: $TARGET_MODEL" >&2; exit 4; }
[[ -d "$DRAFT_MODEL" ]] || { echo "ERROR: draft model directory not found: $DRAFT_MODEL" >&2; exit 5; }
[[ -e /dev/kfd ]] || { echo "ERROR: /dev/kfd not found; stop and fix the existing host GPU environment first" >&2; exit 6; }
[[ -d /opt/hyhal ]] || { echo "ERROR: /opt/hyhal not found; stop and check the existing host environment" >&2; exit 7; }
for d in "$RENDER0" "$RENDER1" "$RENDER2" "$RENDER3"; do
  [[ -e "$d" ]] || { echo "ERROR: render node not found: $d" >&2; exit 8; }
done

docker image inspect "$IMAGE_TAG" >/dev/null 2>&1 || { echo "ERROR: image $IMAGE_TAG not found; run bash build_image.sh first" >&2; exit 9; }
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "ERROR: container $CONTAINER_NAME already exists; refusing to replace it automatically" >&2
  echo "Stop/remove it manually only after confirming it is safe to do so." >&2
  exit 10
fi

args=(
  docker run -d
  --name "$CONTAINER_NAME"
  --network host
  --ipc host
  --restart unless-stopped
  --security-opt label=disable
  --device /dev/kfd:/dev/kfd
  --device "$RENDER0:$RENDER0"
  --device "$RENDER1:$RENDER1"
  --device "$RENDER2:$RENDER2"
  --device "$RENDER3:$RENDER3"
  -v /opt/hyhal:/opt/hyhal:ro
  -v "$TARGET_MODEL:/models/target:ro"
  -v "$DRAFT_MODEL:/models/draft:ro"
  -e HIP_VISIBLE_DEVICES=0,1,2,3
  -e "PORT=$PORT"
  -e "SGLANG_Q38_TP4_U036_PROFILE=$U036_PROFILE"
  -e "SGLANG_Q38_TP4_U036_SPLIT_KV=$U036_SPLIT_KV"
  -e "CUSTOM_AR=$CUSTOM_AR"
  "$IMAGE_TAG"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${args[@]}"
  printf '\n'
  exit 0
fi

"${args[@]}"
echo "Started $CONTAINER_NAME on port $PORT"
echo "Logs: docker logs -f --tail=100 $CONTAINER_NAME"
