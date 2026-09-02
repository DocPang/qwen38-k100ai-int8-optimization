"""Qwen3.8 TP1 v30 profile.

Parent: TP1 Final-v2 plus its narrow 8192-boundary Mamba prefix-checkpoint
repair and the validated Hybrid14 selective-pack short/mid incumbent.  The
TP124 v30 common layer adds release/client/lifecycle semantics without enabling
TP1 raw-q8 or changing the frozen long-context q8split path.
"""
from __future__ import annotations

import os
import runpy


_TROOT = "/data/qwen38-27b-k100ai-int8-opt"
_DROOT = "/data/qwen38-dflash2-k100ai"
os.environ["SGLANG_Q38_V30_PROFILE"] = "tp1"
runpy.run_path(
    f"{_TROOT}/runtime_patch_dflash_tp1_mamba_checkpoint_v2/sitecustomize.py",
    run_name="__q38_tp1_v30_checkpoint_parent__",
)
_skip_key = "SGLANG_Q38_TP1_SHORTMID_SKIP_PARENT"
_skip_old = os.environ.get(_skip_key)
os.environ[_skip_key] = "1"
runpy.run_path(
    f"{_TROOT}/runtime_patch_dflash_tp1_shortmid_selective_pack_diag/sitecustomize.py",
    run_name="__q38_tp1_v30_parent__",
)
if _skip_old is None:
    os.environ.pop(_skip_key, None)
else:
    os.environ[_skip_key] = _skip_old
if os.environ.get("SGLANG_Q38_TP124_V30_DISABLE_COMMON") == "1":
    print(
        "[K100 DFlash TP1 v30] CONTROL: Final-v2 + checkpoint8192 + Hybrid14; v30 common disabled",
        flush=True,
    )
else:
    runpy.run_path(
        f"{_DROOT}/runtime_patch_dflash_tp124_v30_common/sitecustomize.py",
        run_name="__q38_tp1_v30_common__",
    )
    print(
        "[K100 DFlash TP1 v30] installed: Final-v2 long/q8split + checkpoint8192 + Hybrid14 short/mid + v30 common release layer",
        flush=True,
    )
