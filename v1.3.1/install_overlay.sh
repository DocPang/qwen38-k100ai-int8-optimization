#!/usr/bin/env bash
# Advanced/developer helper: install the v1.3.1 source/runtime assets into an
# existing research tree without overwriting divergent files.
# Final-image users do NOT need this script; import qwen38-k100ai-int8:v1.3.1
# and use scripts/launch.sh instead. This helper does not build/download images.
set -euo pipefail

kit=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
dflash_root=${DFLASH_ROOT:-/data/qwen38-dflash2-k100ai}
target_root=${TARGET_ROOT:-/data/qwen38-27b-k100ai-int8-opt}
sourcefind_base=${SOURCEFIND_BASE_OVERLAY:-$dflash_root/work/sourcefind_sglang_overlay_tp4/python}
sourcefind_v131=${SOURCEFIND_V131_OVERLAY:-$dflash_root/work/sourcefind_sglang_overlay_v131/python}
unified_rootfix_dest=${UNIFIED_ROOTFIX_DEST:-$dflash_root/runtime_assets/idletrim_rootfix_unified_v1}

[[ -d "$dflash_root" ]] || { echo "ERROR: missing DFLASH_ROOT=$dflash_root" >&2; exit 20; }
[[ -d "$target_root" ]] || { echo "ERROR: missing TARGET_ROOT=$target_root" >&2; exit 21; }
[[ -d "$sourcefind_base/sglang" ]] || { echo "ERROR: missing SOURCEFIND_BASE_OVERLAY=$sourcefind_base" >&2; exit 22; }

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
  cp -a "$src" "$dest"
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
  cp -a "$src" "$dest"
  echo "INSTALLED $dest"
}

while IFS= read -r src; do
  install_tree "$src" "$target_root/$(basename "$src")"
done < <(find "$kit/overlay/target" -mindepth 1 -maxdepth 1 -type d | sort)

while IFS= read -r src; do
  install_tree "$src" "$dflash_root/$(basename "$src")"
done < <(find "$kit/overlay/dflash2" -mindepth 1 -maxdepth 1 -type d | sort)

while IFS= read -r src; do
  install_file "$src" "$dflash_root/$(basename "$src")"
done < <(find "$kit/overlay/dflash2" -mindepth 1 -maxdepth 1 -type f | sort)

install_file "$kit/native_ext/k100_assign_fixed8_i64_v1.so" "$dflash_root/native_ext/k100_assign_fixed8_i64_v1.so"
install_file "$kit/scripts/launch_sglang_require_sitecustomize.py" "$target_root/scripts/launch_sglang_require_sitecustomize.py"
install_file "$kit/qwen38_chat_template.jinja" "$dflash_root/runtime_assets/qwen38_chat_template.jinja"
install_file "$kit/qwen38_chat_template.jinja" "$target_root/runtime_assets/qwen38_chat_template.jinja"
install_file "$kit/tp2_shortmid_layerdiag_fallback_layers.txt" "$target_root/results/tp2_shortmid_layerdiag_fallback_layers.txt"

# Unified TP1/TP4 rootfix: a dedicated runtime asset directory.
mkdir -p "$unified_rootfix_dest/sglang/srt/managers"
install_file "$kit/rootfix/unified/sglang/srt/managers/scheduler.py" \
  "$unified_rootfix_dest/sglang/srt/managers/scheduler.py"
install_file "$kit/rootfix/unified/sglang/srt/managers/scheduler_runtime_checker_mixin.py" \
  "$unified_rootfix_dest/sglang/srt/managers/scheduler_runtime_checker_mixin.py"

# TP2 must keep the verified legacy SourceFind overlay. Build a separate v1.3.1
# copy once, then replace only the two scheduler files that implement all-rank
# idle allocator trimming. The original base overlay is never modified.
sourcefind_created=0
if [[ ! -d "$sourcefind_v131/sglang" ]]; then
  mkdir -p "$(dirname "$sourcefind_v131")"
  cp -a "$sourcefind_base" "$sourcefind_v131"
  sourcefind_created=1
  echo "COPIED $sourcefind_base -> $sourcefind_v131"
fi

if [[ "$sourcefind_created" == 1 ]]; then
  cp -a "$kit/rootfix/sourcefind/sglang/srt/managers/scheduler.py" \
    "$sourcefind_v131/sglang/srt/managers/scheduler.py"
  cp -a "$kit/rootfix/sourcefind/sglang/srt/managers/scheduler_runtime_checker_mixin.py" \
    "$sourcefind_v131/sglang/srt/managers/scheduler_runtime_checker_mixin.py"
else
  install_file "$kit/rootfix/sourcefind/sglang/srt/managers/scheduler.py" \
    "$sourcefind_v131/sglang/srt/managers/scheduler.py"
  install_file "$kit/rootfix/sourcefind/sglang/srt/managers/scheduler_runtime_checker_mixin.py" \
    "$sourcefind_v131/sglang/srt/managers/scheduler_runtime_checker_mixin.py"
fi

cmp -s "$kit/rootfix/sourcefind/sglang/srt/managers/scheduler.py" \
  "$sourcefind_v131/sglang/srt/managers/scheduler.py" || { echo "ERROR: TP2 scheduler rootfix verify failed" >&2; exit 32; }
cmp -s "$kit/rootfix/sourcefind/sglang/srt/managers/scheduler_runtime_checker_mixin.py" \
  "$sourcefind_v131/sglang/srt/managers/scheduler_runtime_checker_mixin.py" || { echo "ERROR: TP2 mixin rootfix verify failed" >&2; exit 33; }

python3 -m py_compile \
  "$unified_rootfix_dest/sglang/srt/managers/scheduler.py" \
  "$unified_rootfix_dest/sglang/srt/managers/scheduler_runtime_checker_mixin.py" \
  "$sourcefind_v131/sglang/srt/managers/scheduler.py" \
  "$sourcefind_v131/sglang/srt/managers/scheduler_runtime_checker_mixin.py"

echo
echo "v1.3.1 source/runtime assets installed. No image was built or downloaded."
echo "TP2 research overlay: $sourcefind_v131"
echo "Unified research rootfix: $unified_rootfix_dest"
echo "NOTE: final-image users should not run this helper; use qwen38-k100ai-int8:v1.3.1 directly."
