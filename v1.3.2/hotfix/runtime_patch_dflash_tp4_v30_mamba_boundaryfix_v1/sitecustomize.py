"""v1.3.2 TP4 runtime composition for the v1.3.1 final image.

Load the frozen v1.3.1 TP4 v30 runtime first, then install the narrow
DFlash/Mamba verify-boundary correctness hotfix. The v1.3.1 base copy is mounted
at a separate path so this wrapper can safely replace the image's original
runtime_patch_dflash_tp4_v30 directory without recursion.
"""
from __future__ import annotations

import runpy

_ROOT = "/data/qwen38-dflash2-k100ai"

runpy.run_path(
    f"{_ROOT}/runtime_patch_dflash_tp4_v30_base_v131/sitecustomize.py",
    run_name="__q38_tp4_v30_v131_base__",
)
runpy.run_path(
    f"{_ROOT}/runtime_patch_dflash_tp4_mamba_verify_boundary_fix_v1/sitecustomize.py",
    run_name="__q38_tp4_v132_mamba_boundaryfix__",
)

print(
    "[K100 TP4 v1.3.2] installed: v1.3.1 v30 base + "
    "DFlash/Mamba verify-boundary correctness hotfix",
    flush=True,
)
