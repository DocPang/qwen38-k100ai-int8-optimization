#!/usr/bin/env bash
# Install the v1.3.0 profile overlay without overwriting divergent local files.
set -euo pipefail

kit=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
dflash_root=${DFLASH_ROOT:-/data/qwen38-dflash2-k100ai}
target_root=${TARGET_ROOT:-/data/qwen38-27b-k100ai-int8-opt}

[[ -d "$dflash_root" ]] || { echo "ERROR: missing DFLASH_ROOT=$dflash_root" >&2; exit 20; }
[[ -d "$target_root" ]] || { echo "ERROR: missing TARGET_ROOT=$target_root" >&2; exit 21; }

install_tree() {
  local src=$1 dest=$2
  if [[ -e "$dest" ]]; then
    diff -qr --exclude='__pycache__' --exclude='*.pyc' "$src" "$dest" >/dev/null || {
      echo "ERROR: refusing to overwrite divergent path: $dest" >&2
      return 30
    }
    echo "IDENTICAL $dest"
    return
  fi
  mkdir -p "$(dirname "$dest")"
  cp -R "$src" "$dest"
  echo "INSTALLED $dest"
}

install_file() {
  local src=$1 dest=$2
  if [[ -e "$dest" ]]; then
    cmp -s "$src" "$dest" || {
      echo "ERROR: refusing to overwrite divergent file: $dest" >&2
      return 31
    }
    echo "IDENTICAL $dest"
    return
  fi
  mkdir -p "$(dirname "$dest")"
  cp "$src" "$dest"
  echo "INSTALLED $dest"
}

while IFS= read -r src; do
  install_tree "$src" "$target_root/$(basename "$src")"
done < <(find "$kit/overlay/target" -mindepth 1 -maxdepth 1 -type d | sort)

while IFS= read -r src; do
  install_tree "$src" "$dflash_root/$(basename "$src")"
done < <(find "$kit/overlay/dflash2" -mindepth 1 -maxdepth 1 -type d | sort)

for src in "$kit/scripts"/*; do
  install_file "$src" "$dflash_root/scripts/$(basename "$src")"
done
install_file "$kit/native_ext/k100_assign_fixed8_i64_v1.so" "$dflash_root/native_ext/k100_assign_fixed8_i64_v1.so"
install_file "$kit/qwen38_chat_template.jinja" "$dflash_root/runtime_assets/qwen38_chat_template.jinja"
install_file "$kit/qwen38_chat_template.jinja" "$target_root/runtime_assets/qwen38_chat_template.jinja"
install_file "$kit/tp2_shortmid_layerdiag_fallback_layers.txt" "$target_root/results/tp2_shortmid_layerdiag_fallback_layers.txt"

echo "Overlay installed without replacing divergent files."
echo "Run the TP1/TP2 launcher preflight; do not remove the previous deployment until the new service passes /v1/models and the release gates."
