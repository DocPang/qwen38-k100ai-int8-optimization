#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage() {
  cat <<'EOF'
Usage:
  ./launch.sh tp1 [-- SGLang extra args...]
  ./launch.sh tp2 [-- SGLang extra args...]
  ./launch.sh tp4 [-- SGLang extra args...]

Examples:
  ./launch.sh tp4
  PORT=9000 MAX_RUNNING_REQUESTS=4 ./launch.sh tp4
  ./launch.sh tp4 -- --enable-deterministic-inference
  CONTEXT_LENGTH=245760 ./launch.sh tp2 -- --log-level info

Common environment overrides are documented in ../README.md.
v1.3.2 reuses qwen38-k100ai-int8:v1.3.1; no new image is required.
TP4 automatically mounts the v1.3.2 DFlash/Mamba boundary hotfix read-only.
TP1/TP2 keep the validated v1.3.1 runtime unchanged.
Arguments after -- are passed through the final-image entrypoint to SGLang without shell re-parsing.
EOF
}

if (($# == 0)); then
  usage
  exit 2
fi

case "$1" in
  -h|--help)
    usage
    exit 0
    ;;
  tp1|tp2|tp4)
    profile=$1
    shift
    exec "$script_dir/launch_${profile}.sh" "$@"
    ;;
  *)
    echo "ERROR: unknown profile: $1" >&2
    usage >&2
    exit 2
    ;;
esac
