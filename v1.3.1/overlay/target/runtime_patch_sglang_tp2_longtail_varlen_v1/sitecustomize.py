"""TP2 champion {3,23} + fail-closed long-context partial-tail varlen v1.

Loads the accepted champion chain (layerdiag v5 -> longkv v5 -> ... ) unchanged,
then installs a narrow delta on top:

Admits the corrected contiguous-varlen consumer for batch1 partial-tail
prefill chunks 512<=q<=8191 with kv>20000, i.e. exactly the region above the
layerdiag v5 short/mid coverage that previously fell to the stock paged path.

Root cause: longkv BM128 v4 admits only exact q=8192 chunks (kv%8192==0,
32K<=KV<=248K) plus the exact q=3948/KV=257900 tail. Any other partial tail at
long KV (e.g. q=8128/KV=122880 for a 131008-token prompt) ran the stock paged
path at ~50-80 tok/s vs ~1400-1600 tok/s on the admitted grid: a single tail
chunk cost more than the entire aligned prefix (measured 131008 TTFT 245.7s
vs 131072 90.8s on the champion container).

The frozen exact qtail shapes (q in {3948,3968} at kv>=250000) stay with the
v4 hook. Everything else falls back to the champion parent unchanged.
"""
from __future__ import annotations

import os
import runpy

_ROOT = "/data/qwen38-27b-k100ai-int8-opt"
_PARENT = f"{_ROOT}/runtime_patch_sglang_tp2_shortmid_layerdiag_v5/sitecustomize.py"
_DELTA = f"{_ROOT}/runtime_patch_sglang_tp2_longtail_varlen_v1/delta.py"

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

runpy.run_path(_PARENT, run_name="__q38_tp2_longtail_v1_parent__")
if _helper:
    print(f"[K100 DFlash2 TP2 longtail varlen v1 helper bypass] pid={os.getpid()}", flush=True)
else:
    runpy.run_path(_DELTA, run_name="__q38_tp2_longtail_v1_delta__")
    print("[K100 DFlash2 TP2 longtail varlen v1 wrapper] champion {3,23} parent + long partial-tail admission active", flush=True)
