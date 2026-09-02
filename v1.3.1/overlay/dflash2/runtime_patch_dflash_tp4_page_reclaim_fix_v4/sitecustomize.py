"""Reclaim DFlash verify pages lost from a finished request's page table.

SourceFind DFlash can retain a physical page across verify rounds and later
lose that historical page from req_to_token.  Normal finished-request cleanup
then cannot discover the page.  Track only pages actually returned by paged
verify allocation and, after normal Radix cleanup, reclaim tracked pages that
are in neither the allocator free lists nor the Radix tree.

The reconciliation is deliberately fail-closed: a page is never freed when it
is still represented by allocator or cache ownership, and strict pool checks
remain enabled as the final authority.
"""
from __future__ import annotations

import os
import runpy

import torch

_ROOT = "/data/qwen38-dflash2-k100ai"
if os.getenv("SGLANG_Q38_TP124_V30_SKIP_TP4_PARENT", "0") != "1":
    runpy.run_path(
        f"{_ROOT}/runtime_patch_dflash_tp4_retained_alloc_fix_v3/sitecustomize.py",
        run_name="__tp4_page_reclaim_fix_v4_parent__",
    )

from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.srt.speculative import dflash_info as _dflash_info

_parent_verify = _dflash_info.DFlashVerifyInput.verify
_parent_cache_finished = MambaRadixCache.cache_finished_req
_tracked_verify_locs: dict[str, list[torch.Tensor]] = {}
_reclaim_events = 0
_LOG_LIMIT = int(os.getenv("SGLANG_DFLASH_PAGE_RECLAIM_LOG_LIMIT", "64"))


def _cpu_int_set(values) -> set[int]:
    if values is None:
        return set()
    if isinstance(values, torch.Tensor):
        values = values.detach().to("cpu").tolist()
    return {int(value) for value in values}


def _verify_with_page_tracking(self, *args, **kwargs):
    batch = kwargs.get("batch")
    page_size = int(kwargs.get("page_size", 1))
    if batch is None or page_size <= 1 or batch.forward_mode.is_idle():
        return _parent_verify(self, *args, **kwargs)

    bs = batch.batch_size()
    full_verify_locs = batch.out_cache_loc.view(bs, int(self.draft_token_num))
    reqs = list(batch.reqs)
    for row, req in zip(full_verify_locs, reqs, strict=True):
        rid = str(getattr(req, "rid", "?"))
        _tracked_verify_locs.setdefault(rid, []).append(row.detach().clone())

    return _parent_verify(self, *args, **kwargs)


def _cache_finished_with_page_reclaim(self, req, is_insert=True):
    global _reclaim_events
    rid = str(getattr(req, "rid", "?"))
    tracked_locs = _tracked_verify_locs.pop(rid, [])

    result = _parent_cache_finished(self, req, is_insert=is_insert)
    if not tracked_locs:
        return result

    allocator = self.token_to_kv_pool_allocator
    page_size = int(getattr(allocator, "page_size", 1))
    if page_size <= 1:
        return result

    tracked = torch.cat(tracked_locs)
    tracked_pages = _cpu_int_set(torch.unique(tracked // page_size))
    free_pages = _cpu_int_set(getattr(allocator, "free_pages", None))
    release_pages = _cpu_int_set(getattr(allocator, "release_pages", None))
    tree_slots = self.all_values_flatten()
    tree_pages = {
        slot // page_size for slot in _cpu_int_set(tree_slots) if int(slot) > 0
    }
    orphan_pages = tracked_pages - free_pages - release_pages - tree_pages

    if orphan_pages:
        reclaim_locs = torch.tensor(
            [page * page_size for page in sorted(orphan_pages)],
            dtype=tracked.dtype,
            device=tracked.device,
        )
        allocator.free(reclaim_locs)

    _reclaim_events += 1
    if orphan_pages or _reclaim_events <= _LOG_LIMIT:
        print(
            "[K100 DFlash TP4 page-reclaim-fix-v4] ACTIVE "
            f"event={_reclaim_events} rid={rid} "
            f"tracked_pages={len(tracked_pages)} "
            f"reclaimed_pages={sorted(orphan_pages)}",
            flush=True,
        )
    return result


_dflash_info.DFlashVerifyInput.verify = _verify_with_page_tracking
MambaRadixCache.cache_finished_req = _cache_finished_with_page_reclaim

print(
    "[K100 DFlash TP4 page-reclaim-fix-v4] installed: finished requests "
    "reclaim tracked verify pages absent from allocator and Radix ownership",
    flush=True,
)
