"""TP4 FA260728 raw-q8 compatibility candidate with corrected layout=0 ABI.

This candidate restores the exact TP4 DFlash q=8 verifier on SourceFind
flash-attn 260728 after isolated gates proved that the new paged_attention
final layout selector must be 0 for the production paged KV cache geometry.

Safety contract:
- exact flash-attn build only;
- exact TP4 raw-q8 geometry remains enforced by Agent128K;
- all non-q8/non-exact shapes inherit the existing production parent;
- BF16 KV only;
- no generic q>=5 vendor wrapper enablement.
"""
from __future__ import annotations

import importlib.metadata as _metadata
import os
import runpy

_EXPECTED = "2.8.3+das.opt1.dtk2604.torch290.2607280958.gebb4be"
_actual = _metadata.version("flash-attn")
if _actual != _EXPECTED:
    raise RuntimeError(
        "TP4 FA260728 raw-q8 layout0 candidate is version-locked: "
        f"expected={_EXPECTED}, actual={_actual}"
    )

# Agent128K is itself exact-shape fail-closed.  Enabling this switch only makes
# its audited q=8/QH6/KVH1/D256/page64/BF16 branch eligible.
os.environ["SGLANG_DFLASH2_Q8_NATIVE_PAGED"] = "1"

_PARENT = (
    "/data/qwen38-dflash2-k100ai/"
    "runtime_patch_dflash_tp4_partition_equiv_v3/sitecustomize.py"
)
runpy.run_path(_PARENT, run_name="__q38_tp4_fa260728_rawq8_layout0_parent__")

print(
    "[K100 TP4 FA260728 rawq8-layout0] ACTIVE: exact TP4 q8 raw paged_attention restored; layout=0; all other paths unchanged",
    flush=True,
)
