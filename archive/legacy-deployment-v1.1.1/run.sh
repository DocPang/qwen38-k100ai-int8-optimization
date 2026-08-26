#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENV_FILE=${ENV_FILE:-$ROOT/.env}
[[ -f "$ENV_FILE" ]] || { echo "ERROR: missing $ENV_FILE; copy .env.example to .env first" >&2; exit 2; }
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

PROFILE=${PROFILE:-tp4}
IMAGE_TAG=${IMAGE_TAG:-qwen38-k100ai-int8-series:local}
TARGET_MODEL=${TARGET_MODEL:?Set TARGET_MODEL in .env}
DRAFT_MODEL=${DRAFT_MODEL:?Set DRAFT_MODEL in .env}
CUSTOM_AR=${CUSTOM_AR:-1}
P2P=${P2P:-1}

case "$PROFILE" in
  tp1)
    default_port=8090; default_name=qwen38-tp1; hip=0
    renders=("${RENDER0:?Set RENDER0 for TP1}")
    ;;
  tp2)
    default_port=8062; default_name=qwen38-tp2; hip=0,1
    renders=("${RENDER0:?Set RENDER0 for TP2}" "${RENDER1:?Set RENDER1 for TP2}")
    ;;
  tp4)
    default_port=8068; default_name=qwen38-tp4; hip=0,1,2,3
    renders=("${RENDER0:?Set RENDER0 for TP4}" "${RENDER1:?Set RENDER1 for TP4}" "${RENDER2:?Set RENDER2 for TP4}" "${RENDER3:?Set RENDER3 for TP4}")
    ;;
  *)
    echo "ERROR: unsupported PROFILE=$PROFILE; expected tp1, tp2 or tp4" >&2
    exit 3
    ;;
esac
PORT=${PORT:-$default_port}
CONTAINER_NAME=${CONTAINER_NAME:-$default_name}

[[ -d "$TARGET_MODEL" ]] || { echo "ERROR: target model directory not found: $TARGET_MODEL" >&2; exit 4; }
[[ -d "$DRAFT_MODEL" ]] || { echo "ERROR: draft model directory not found: $DRAFT_MODEL" >&2; exit 5; }
[[ -e /dev/kfd ]] || { echo "ERROR: /dev/kfd not found; fix the existing host GPU environment first" >&2; exit 6; }
[[ -d /opt/hyhal ]] || { echo "ERROR: /opt/hyhal not found; check the existing host environment" >&2; exit 7; }
for d in "${renders[@]}"; do
  [[ -e "$d" ]] || { echo "ERROR: render node not found: $d" >&2; exit 8; }
done

docker image inspect "$IMAGE_TAG" >/dev/null 2>&1 || { echo "ERROR: image $IMAGE_TAG not found; for A load the unified image first, for B/C run bash build_image.sh first" >&2; exit 9; }
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "ERROR: container $CONTAINER_NAME already exists; refusing to replace it automatically" >&2
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
)
for d in "${renders[@]}"; do args+=(--device "$d:$d"); done
args+=(
  -v /opt/hyhal:/opt/hyhal:ro
  -v "$TARGET_MODEL:/models/target:ro"
  -v "$DRAFT_MODEL:/models/draft:ro"
  -e "PROFILE=$PROFILE"
  -e "HIP_VISIBLE_DEVICES=$hip"
  -e "PORT=$PORT"
  -e "CUSTOM_AR=$CUSTOM_AR"
  -e "P2P=$P2P"
  "$IMAGE_TAG"
)

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${args[@]}"; printf '\n'; exit 0
fi

"${args[@]}"
echo "Started $CONTAINER_NAME: profile=$PROFILE port=$PORT devices=${renders[*]}"
echo "Logs: docker logs -f --tail=100 $CONTAINER_NAME"
