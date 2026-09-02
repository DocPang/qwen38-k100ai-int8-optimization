#!/usr/bin/env bash
set -euo pipefail

Q38_FINAL_IMAGE_DEFAULT=${Q38_FINAL_IMAGE_DEFAULT:-qwen38-k100ai-int8:v1.3.1}
Q38_FINAL_IMAGE_ID=${Q38_FINAL_IMAGE_ID:-sha256:534f9512a3d5217c8f65f11183ed75db19e7aa3df1ecc4da1ba993e98793973d}

q38_die() {
  echo "ERROR: $*" >&2
  exit 1
}

q38_require_cmd() {
  command -v "$1" >/dev/null 2>&1 || q38_die "missing command: $1"
}

q38_require_dir() {
  [[ -d "$1" ]] || q38_die "missing required directory: $1"
}

q38_collect_sglang_args() {
  Q38_SGLANG_ARGS=()
  if (($# == 0)); then
    return 0
  fi
  [[ "$1" == "--" ]] || q38_die "unexpected argument '$1'. Put raw SGLang arguments after --"
  shift
  Q38_SGLANG_ARGS=("$@")
}

q38_check_image() {
  local image=$1
  local actual
  actual=$(docker image inspect "$image" --format '{{.Id}}' 2>/dev/null || true)
  if [[ "$actual" == "$Q38_FINAL_IMAGE_ID" ]]; then
    return 0
  fi
  if [[ "${ALLOW_UNVERIFIED_IMAGE:-0}" == 1 ]]; then
    echo "WARNING: image digest mismatch; continuing because ALLOW_UNVERIFIED_IMAGE=1" >&2
    echo "  expected=$Q38_FINAL_IMAGE_ID" >&2
    echo "  actual=${actual:-missing}" >&2
    return 0
  fi
  q38_die "verified image mismatch: image=$image expected=$Q38_FINAL_IMAGE_ID actual=${actual:-missing}"
}

q38_check_container_absent() {
  local name=$1
  if docker inspect "$name" >/dev/null 2>&1; then
    q38_die "container already exists: $name (stop/remove it or set NAME=...)"
  fi
}

q38_check_port_free() {
  local port=$1
  if command -v ss >/dev/null 2>&1 && ss -ltnH | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
    q38_die "port is already occupied: $port"
  fi
}

q38_hysmi() {
  if [[ -x /usr/local/hyhal/bin/hy-smi ]]; then
    /usr/local/hyhal/bin/hy-smi "$@"
  elif command -v hy-smi >/dev/null 2>&1; then
    hy-smi "$@"
  else
    q38_die "hy-smi not found"
  fi
}

q38_check_gpu_idle() {
  local gpu=$1 usage
  usage=$(q38_hysmi -d "$gpu" --showuse --showmemuse 2>/dev/null || true)
  printf '%s\n' "$usage"
  grep -q 'HCU use (%): 0.0' <<<"$usage" || q38_die "GPU${gpu} compute busy"
  grep -q 'HCU memory use (%): 0' <<<"$usage" || q38_die "GPU${gpu} memory busy"
}

q38_check_render_idle() {
  local render=$1
  [[ -e "$render" ]] || q38_die "missing render device: $render"
  if command -v fuser >/dev/null 2>&1 && fuser "$render" >/dev/null 2>&1; then
    q38_die "render device has users: $render"
  fi
}

q38_check_weights() {
  local target=$1 draft=$2
  q38_require_dir "$target"
  q38_require_dir "$draft"
  [[ -f "$target/config.json" ]] || q38_die "Target config.json missing: $target"
  [[ -f "$target/tokenizer.json" ]] || q38_die "Target tokenizer.json missing: $target"
  [[ -f "$draft/config.json" ]] || q38_die "Draft config.json missing: $draft"
  [[ -f "$draft/model.safetensors" ]] || q38_die "Draft model.safetensors missing: $draft"
}

q38_print_extra_args() {
  if ((${#Q38_SGLANG_ARGS[@]} == 0)); then
    echo "SGLang extra args: <none>"
  else
    printf 'SGLang extra args:'
    printf ' %q' "${Q38_SGLANG_ARGS[@]}"
    printf '\n'
  fi
}
