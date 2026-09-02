"""Track paged DFlash verify slots that remain physically allocated.

SourceFind DFlash v1 keeps uncommitted verify slots that share the final
committed page, but then collapses ``req.kv_allocated_len`` to the committed
watermark.  The full verify mapping was already written by
``prepare_for_verify``; retain the matching logical allocation watermark so
normal request cleanup can reclaim those pages.

This patch composes on top of assign-verify-fix-v2 and changes no allocation,
sampling, cache insertion, attention, or long-context policy.
"""
from __future__ import annotations

import os
import runpy

_ROOT = "/data/qwen38-dflash2-k100ai"
if os.getenv("SGLANG_Q38_TP124_V30_SKIP_TP4_PARENT", "0") != "1":
    runpy.run_path(
        f"{_ROOT}/runtime_patch_dflash_tp4_assign_verify_fix_v2/sitecustomize.py",
        run_name="__tp4_retained_alloc_fix_v3_parent__",
    )

from sglang.srt.speculative import dflash_info as _dflash_info

_parent_verify = _dflash_info.DFlashVerifyInput.verify
_events = 0
_LOG_LIMIT = int(os.getenv("SGLANG_DFLASH_RETAINED_ALLOC_LOG_LIMIT", "32"))


def _verify_with_retained_allocation(self, *args, **kwargs):
    global _events
    batch = kwargs.get("batch")
    page_size = int(kwargs.get("page_size", 1))
    if batch is None or page_size <= 1 or batch.forward_mode.is_idle():
        return _parent_verify(self, *args, **kwargs)

    # Snapshot the committed prefixes before the parent advances seq_lens.
    prefix_lens = batch.seq_lens.detach().clone()
    reqs = list(batch.reqs)
    result = _parent_verify(self, *args, **kwargs)
    commit_lens = result[1]

    keep_slots = _dflash_info._compute_paged_keep_slots(
        prefix_lens=prefix_lens,
        commit_lens=commit_lens,
        draft_token_num=int(self.draft_token_num),
        page_size=page_size,
    )
    prefix_cpu = prefix_lens.detach().cpu().tolist()
    keep_cpu = keep_slots.detach().cpu().tolist()
    commit_cpu = commit_lens.detach().cpu().tolist()

    rows = []
    for req, prefix, commit, keep in zip(
        reqs, prefix_cpu, commit_cpu, keep_cpu, strict=True
    ):
        committed = int(req.kv_committed_len)
        expected_committed = int(prefix) + int(commit)
        if committed != expected_committed:
            raise RuntimeError(
                "DFlash retained allocation commit mismatch: "
                f"rid={getattr(req, 'rid', '?')} committed={committed} "
                f"expected={expected_committed}"
            )
        retained_allocated = int(prefix) + int(keep)
        if retained_allocated < committed:
            raise RuntimeError(
                "DFlash retained allocation below committed watermark: "
                f"rid={getattr(req, 'rid', '?')} allocated={retained_allocated} "
                f"committed={committed}"
            )
        req.kv_allocated_len = retained_allocated
        rows.append(
            {
                "rid": str(getattr(req, "rid", "?")),
                "prefix": int(prefix),
                "commit": int(commit),
                "keep": int(keep),
                "allocated": retained_allocated,
            }
        )

    _events += 1
    if _events <= _LOG_LIMIT or _events % 128 == 0:
        print(
            f"[K100 DFlash TP4 retained-alloc-fix-v3] ACTIVE "
            f"event={_events} page_size={page_size} rows={rows}",
            flush=True,
        )
    return result


_dflash_info.DFlashVerifyInput.verify = _verify_with_retained_allocation

print(
    "[K100 DFlash TP4 retained-alloc-fix-v3] installed: paged verify "
    "keep-slot physical ownership retained in req.kv_allocated_len",
    flush=True,
)
