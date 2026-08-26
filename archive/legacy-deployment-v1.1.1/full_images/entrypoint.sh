#!/usr/bin/env bash
set -euo pipefail

PROFILE=${PROFILE:-tp4}
case "$PROFILE" in
  tp1) exec /opt/qwen38-k100ai/entrypoint.tp1.sh "$@" ;;
  tp2) exec /opt/qwen38-k100ai/entrypoint.tp2.sh "$@" ;;
  tp4) exec /opt/qwen38-k100ai/entrypoint.tp4.sh "$@" ;;
  *)
    echo "ERROR: unsupported PROFILE=$PROFILE; expected tp1, tp2 or tp4" >&2
    exit 64
    ;;
esac
