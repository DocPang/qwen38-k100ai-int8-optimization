"""Qwen3.8 TP2 v30 profile.

Parent: TP2 longtail-v1 champion plus its narrow 8192-boundary Mamba
checkpoint-creation repair, including the {3,23} relaxed-quality fallback
contract and validated short/mid/long partial-tail routes.  The TP124 v30
common layer adds release/client/lifecycle semantics while preserving TP2
QH12/KVH2 attention and shared-scale correctness.
"""
from __future__ import annotations

import os
import runpy


_TROOT = "/data/qwen38-27b-k100ai-int8-opt"
_DROOT = "/data/qwen38-dflash2-k100ai"
os.environ["SGLANG_Q38_V30_PROFILE"] = "tp2"
runpy.run_path(
    f"{_TROOT}/runtime_patch_sglang_tp2_mamba_checkpoint8k_v1/sitecustomize.py",
    run_name="__q38_tp2_v30_parent__",
)
if os.environ.get("SGLANG_Q38_TP124_V30_DISABLE_COMMON") == "1":
    print(
        "[K100 DFlash TP2 v30] CONTROL: longtail-v1/{3,23} + checkpoint8192; v30 common disabled",
        flush=True,
    )
else:
    runpy.run_path(
        f"{_DROOT}/runtime_patch_dflash_tp124_v30_common/sitecustomize.py",
        run_name="__q38_tp2_v30_common__",
    )
    print(
        "[K100 DFlash TP2 v30] installed: longtail-v1/{3,23} + checkpoint8192 + v30 common release layer",
        flush=True,
    )
