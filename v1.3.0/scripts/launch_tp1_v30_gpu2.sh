#!/usr/bin/env bash
# Stable-name TP1 v30 deployment entrypoint.  The underlying launcher is shared
# with candidate runs so the validated container contract cannot drift.
set -euo pipefail

export NAME=${NAME:-q38-tp1-v30-prod-gpu2}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-Qwen3.8-27B-W8A8-DFlash2-TP1}
exec /data/qwen38-dflash2-k100ai/scripts/launch_tp1_v30_candidate_gpu2.sh
