"""TP2 final-v5 + fail-closed short/mid varlen v2.

Loads the accepted TP2 v5 parent unchanged, then installs a narrow delta:
1) first cold-prefill chunk q==kv in [2048,8191] (v1 behavior), and
2) exact second partial chunk with prefix==8192, i.e. kv-q==8192 and q<8192.

The second gate directly targets the 8K..16K TTFT sawtooth caused by the
upstream auto-pack performance threshold max_seq_len_q>=8192.  All other
shapes fall back to the accepted parent.
"""
from __future__ import annotations

import os
import runpy

_ROOT = "/data/qwen38-27b-k100ai-int8-opt"
_PARENT = f"{_ROOT}/runtime_patch_sglang_tp2_dflash2_longkv_bm128_v5/sitecustomize.py"
_DELTA = f"{_ROOT}/runtime_patch_sglang_tp2_shortmid_layerdiag_v5/delta.py"

try:
    _cmd = open("/proc/self/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "replace")
except Exception:
    _cmd = ""
_helper = (
    "multiprocessing.resource_tracker" in _cmd
    or "/usr/local/bin/ninja" in _cmd
    or " ninja --version" in _cmd
    or "cdll.LoadLibrary" in _cmd
)

runpy.run_path(_PARENT, run_name="__q38_tp2_shortmid_layerdiag_v5_parent__")
if _helper:
    print(f"[K100 DFlash2 TP2 shortmid varlen v2 helper bypass] pid={os.getpid()}", flush=True)
else:
    runpy.run_path(_DELTA, run_name="__q38_tp2_shortmid_layerdiag_v5_delta__")
    print("[K100 DFlash2 TP2 shortmid layerdiag v5 wrapper] parent + dynamic layer-fallback diagnostic active", flush=True)
