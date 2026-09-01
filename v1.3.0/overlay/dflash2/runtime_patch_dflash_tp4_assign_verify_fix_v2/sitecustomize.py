"""TP4 DFlash batch>1 post-verify req->KV writeback fix candidate v2.

Root cause proven on 2026-08-25:
SourceFind's ``dcu_assign_req_to_token_pool`` is safe for DFlash's full-width
prepare-for-verify assignment, but it can mis-address later requests when
DFlash verify compacts ``out_cache_loc`` down to the committed tokens before
writing the mapping back.  The failure is not limited to non-uniform commit
lengths: a uniform ``[2, 2, 2, 2]`` verify changed request 2 from page99 to
page98 and orphaned page99.  A prior non-uniform ``[1, 2, 1, 1]`` case changed
page27 to page26.

This candidate is deliberately narrow:
- inherit the current unified-adaptive + page ownership/page-pool diagnostics;
- keep SourceFind/DCU assignment for prepare-for-verify and all batch1 writes;
- only while ``DFlashVerifyInput.verify`` is executing, and only for batch>1,
  route req->KV writeback through stock SGLang's cumulative-offset Triton
  kernel;
- do not change allocator policy, strict checks, Radix cache, DFlash verifier,
  CUDA Graph, or any TP4 long-context specialization.
"""
from __future__ import annotations

import os
import runpy

_ROOT = "/data/qwen38-dflash2-k100ai"
if os.getenv("SGLANG_Q38_TP124_V30_SKIP_TP4_PARENT", "0") != "1":
    runpy.run_path(
        f"{_ROOT}/runtime_patch_dflash_tp4_pagepooldiag_v1/sitecustomize.py",
        run_name="__tp4_assign_verify_fix_v2_parent__",
    )

import torch
from sglang.srt.speculative import dflash_info as _dflash_info
from sglang.srt.speculative import spec_utils as _spec_utils

_parent_assign = _dflash_info.assign_req_to_token_pool_func
_parent_verify = _dflash_info.DFlashVerifyInput.verify
_verify_depth = 0
_verify_assign_hits = 0


def _verify_scoped(self, *args, **kwargs):
    """Mark the synchronous DFlash verify scope without changing its behavior."""
    global _verify_depth
    _verify_depth += 1
    try:
        return _parent_verify(self, *args, **kwargs)
    finally:
        _verify_depth -= 1


def _dflash_assign_verify_safe(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
    batch_size: int,
):
    global _verify_assign_hits
    bs = int(batch_size)

    # prepare_for_verify runs outside DFlashVerifyInput.verify and therefore
    # retains the SourceFind/DCU fast op.  Only the post-verify compact mapping
    # for true multi-request batches takes the cumulative-offset implementation.
    if _verify_depth <= 0 or bs <= 1:
        return _parent_assign(
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
            batch_size,
        )

    lens = end_offset[:bs] - start_offset[:bs]
    _verify_assign_hits += 1
    if _verify_assign_hits <= 16 or _verify_assign_hits % 64 == 0:
        print(
            f"[K100 DFlash TP4 assign-verify-fix-v2] ACTIVE "
            f"hit={_verify_assign_hits} bs={bs} "
            f"lens={lens.detach().to('cpu').tolist()} "
            f"compact={int(out_cache_loc.numel())}",
            flush=True,
        )

    _spec_utils.assign_req_to_token_pool[(bs,)](
        req_pool_indices,
        req_to_token,
        start_offset,
        end_offset,
        out_cache_loc,
        req_to_token.shape[1],
        _spec_utils.next_power_of_2(bs),
    )


_dflash_info.assign_req_to_token_pool_func = _dflash_assign_verify_safe
_dflash_info.DFlashVerifyInput.verify = _verify_scoped

print(
    "[K100 DFlash TP4 assign-verify-fix-v2] installed: DCU assign preserved for "
    "prepare/batch1; Triton cumulative-offset writeback only inside batch>1 "
    "DFlash verify",
    flush=True,
)
