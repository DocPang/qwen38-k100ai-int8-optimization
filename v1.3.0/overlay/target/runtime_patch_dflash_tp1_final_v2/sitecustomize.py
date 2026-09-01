"""TP1 DFlash2 Final v2 production overlay.

Parent provides the mature TP1 stack:
- early-Triton semantic repair + proven q8split verifier
- BM128/w8 long-KV q8K attention on the 32K..253952 scheduler grid
- exact q3948/KV257900 ceil-causal BM128 qtail

Raw q8 paged verifier is intentionally NOT enabled on TP1: QH24/KVH4 isolated
validation was bitwise at 64K but drifted at 128K and VMFaulted at 257.9K.
"""
from __future__ import annotations
import runpy
_ROOT="/data/qwen38-27b-k100ai-int8-opt"
_PARENT=f"{_ROOT}/runtime_patch_dflash_tp1_longkv_bm128_v2/sitecustomize.py"
runpy.run_path(_PARENT, run_name="__q38_tp1_final_v2_parent__")
print("[K100 DFlash2 TP1 Final v2] installed: BM128 32K..253952 + exact257900 ceil-qtail; proven q8split + early-Triton preserved; raw-q8 disabled by TP1 gate",flush=True)
