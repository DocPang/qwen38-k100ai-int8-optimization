"""TP4 row-LDSx/compact-head stack + selective gfx928 paged-varlen repair."""
from __future__ import annotations
import runpy

_ROOT = "/data/qwen38-27b-k100ai-int8-opt"
runpy.run_path(
    f"{_ROOT}/runtime_patch_sglang_tp4_row_ldsx_v1/sitecustomize.py",
    run_name="__q38_sglang_tp4_rowldsx_parent__",
)
runpy.run_path(
    f"{_ROOT}/runtime_patch_sglang_gfx928_paged_varlen_repair_v1/repair.py",
    run_name="__q38_sglang_gfx928_paged_varlen_repair__",
)
