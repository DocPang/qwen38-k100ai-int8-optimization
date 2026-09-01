"""TP4 DFlash verify-only stock paged allocator v9.

Source-level workaround for a K100AI/SourceFind paged allocator defect isolated on
2026-08-30.  With the stock Triton paged allocator, the 23K+1024 DFlash run
returns to idle with strict pool checks clean; with the SourceFind/DCU
``dcu_alloc_extend_kernel`` path, the same lifecycle can orphan physical KV
pages under speculative verify.

This patch is deliberately narrow:
- inherit abort-nocache-v8 and the accepted TP4 runtime stack;
- only while ``DFlashVerifyInput.prepare_for_verify`` performs paged KV extend
  allocation, temporarily force ``PagedTokenToKVPoolAllocator`` to use the
  stock Triton allocator;
- restore the original allocator mode immediately after verify preparation;
- ordinary prefill, decode allocation, Radix/Mamba, attention, verifier,
  sampling, and strict runtime checks remain unchanged.

No idle reconciliation, cache flush, leak suppression, or post-hoc reclaim is
included.  The strict pool checker remains the acceptance authority.
"""
from __future__ import annotations

import os
import runpy

_ROOT = "/data/qwen38-dflash2-k100ai"
if os.getenv("SGLANG_Q38_TP124_V30_SKIP_TP4_PARENT", "0") != "1":
    runpy.run_path(
        f"{_ROOT}/runtime_patch_dflash_tp4_abort_nocache_v8/sitecustomize.py",
        run_name="__tp4_verify_stockalloc_v9_parent__",
    )

from sglang.srt.speculative.dflash_info import DFlashVerifyInput

_parent_prepare = DFlashVerifyInput.prepare_for_verify
_hits = 0


def _prepare_verify_stockalloc(self, batch, page_size: int, *, build_custom_mask: bool = True):
    global _hits
    # page_size==1 has no paged allocator and must preserve stock behavior.
    if int(page_size) <= 1:
        return _parent_prepare(
            self, batch, page_size, build_custom_mask=build_custom_mask
        )

    allocator = batch.tree_cache.token_to_kv_pool_allocator
    old_mode = getattr(allocator, "sglang_kvalloc_kernel", None)
    if old_mode is None:
        # Fail closed to parent if this runtime does not expose the SourceFind
        # selector; do not guess at allocator internals.
        return _parent_prepare(
            self, batch, page_size, build_custom_mask=build_custom_mask
        )

    _hits += 1
    allocator.sglang_kvalloc_kernel = False
    try:
        out = _parent_prepare(
            self, batch, page_size, build_custom_mask=build_custom_mask
        )
    finally:
        allocator.sglang_kvalloc_kernel = old_mode

    if _hits <= 8 or _hits % 128 == 0:
        print(
            "[K100 TP4 verify-stockalloc-v9] ACTIVE "
            f"hit={_hits} page_size={int(page_size)} restored_dcu={bool(old_mode)}",
            flush=True,
        )
    return out


DFlashVerifyInput.prepare_for_verify = _prepare_verify_stockalloc

print(
    "[K100 TP4 verify-stockalloc-v9] installed: stock Triton paged alloc only inside DFlash verify prepare; strict checker unchanged",
    flush=True,
)
