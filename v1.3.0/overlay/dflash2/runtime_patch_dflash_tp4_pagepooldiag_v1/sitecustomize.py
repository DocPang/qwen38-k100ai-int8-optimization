"""Diagnostic-only TP4 full-KV page-pool accounting trace.

This composes on top of the current DFlash page-owner diagnostic and leaves the
scheduler's strict memory-check behavior unchanged.  Only when the existing
``_check_all_pools`` reports a leak do we compute page-set authority data:

* allocator free + release page ids;
* Radix tree physical page ids (token slots // page_size);
* free/tree overlap;
* expected - (free union tree) missing pages;
* counter-based evictable tokens versus actual tree token slots/pages.

The diagnostic is fail-open with respect to itself: any tracing exception is
printed, then the original leak result is returned unchanged so the existing
strict checker remains the authority.
"""
from __future__ import annotations

import json
import os
import runpy

_ROOT = "/data/qwen38-dflash2-k100ai"
runpy.run_path(
    f"{_ROOT}/runtime_patch_dflash_tp4_pageownerdiag_v1/sitecustomize.py",
    run_name="__tp4_pagepooldiag_parent__",
)

import torch
from sglang.srt.managers.scheduler_runtime_checker_mixin import (
    SchedulerRuntimeCheckerMixin,
)

_orig_check_all_pools = SchedulerRuntimeCheckerMixin._check_all_pools
_diag_count = 0
_DIAG_LIMIT = int(os.getenv("SGLANG_TP4_PAGEPOOL_DIAG_LIMIT", "32"))


def _cpu_int_list(x):
    if x is None:
        return []
    if isinstance(x, torch.Tensor):
        return [int(v) for v in x.detach().to("cpu").tolist()]
    return [int(v) for v in x]


def _sorted_sample(values, limit=64):
    vals = sorted(int(v) for v in values)
    return vals[:limit], len(vals)


def _check_all_pools_with_page_diag(self, ps, uncached: int = 0):
    global _diag_count
    result = _orig_check_all_pools(self, ps, uncached=uncached)
    has_leak, _messages = result
    if not has_leak or _diag_count >= _DIAG_LIMIT:
        return result

    _diag_count += 1
    try:
        alloc = self.token_to_kv_pool_allocator
        page_size = int(getattr(alloc, "page_size", getattr(self, "page_size", 1)))
        if page_size <= 1:
            print(
                "TP4_PAGEPOOL_DIAG "
                + json.dumps(
                    {
                        "event": _diag_count,
                        "tp_rank": int(getattr(self, "tp_rank", -1)),
                        "page_size": page_size,
                        "note": "non-paged allocator; page-set diagnostic skipped",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return result

        free_pages = set(_cpu_int_list(getattr(alloc, "free_pages", None)))
        release_pages = set(_cpu_int_list(getattr(alloc, "release_pages", None)))
        allocator_free_pages = free_pages | release_pages

        tree_slots_t = self.tree_cache.all_values_flatten()
        tree_slots = _cpu_int_list(tree_slots_t)
        tree_pages = {slot // page_size for slot in tree_slots if int(slot) > 0}

        num_pages = int(
            getattr(alloc, "num_pages", int(getattr(alloc, "size", 0)) // page_size)
        )
        expected_pages = set(range(1, num_pages + 1))
        overlap_pages = allocator_free_pages & tree_pages
        missing_pages = expected_pages - allocator_free_pages - tree_pages
        extra_pages = (allocator_free_pages | tree_pages) - expected_pages

        free_sample, free_count = _sorted_sample(allocator_free_pages)
        tree_sample, tree_count = _sorted_sample(tree_pages)
        overlap_sample, overlap_count = _sorted_sample(overlap_pages)
        missing_sample, missing_count = _sorted_sample(missing_pages)
        extra_sample, extra_count = _sorted_sample(extra_pages)

        counter_evictable = int(self.tree_cache.full_evictable_size())
        counter_protected = int(self.tree_cache.full_protected_size())
        tree_slot_count = len(tree_slots)
        tree_unique_slot_count = len(set(tree_slots))
        actual_tree_page_tokens = tree_count * page_size
        physical_accounted_page_tokens = (free_count + tree_count) * page_size

        payload = {
            "event": _diag_count,
            "tp_rank": int(getattr(self, "tp_rank", -1)),
            "page_size": page_size,
            "num_pages": num_pages,
            "allocator_size_tokens": int(getattr(alloc, "size", 0)),
            "free_pages_count": free_count,
            "release_pages_count": len(release_pages),
            "tree_pages_count": tree_count,
            "tree_slot_count": tree_slot_count,
            "tree_unique_slot_count": tree_unique_slot_count,
            "counter_evictable_tokens": counter_evictable,
            "counter_protected_tokens": counter_protected,
            "actual_tree_page_tokens": actual_tree_page_tokens,
            "physical_accounted_page_tokens": physical_accounted_page_tokens,
            "overlap_pages_count": overlap_count,
            "missing_pages_count": missing_count,
            "extra_pages_count": extra_count,
            "free_pages_sample": free_sample,
            "tree_pages_sample": tree_sample,
            "overlap_pages_sample": overlap_sample,
            "missing_pages_sample": missing_sample,
            "extra_pages_sample": extra_sample,
            "invariant_messages": list(_messages),
        }
        print("TP4_PAGEPOOL_DIAG " + json.dumps(payload, sort_keys=True), flush=True)
    except Exception as exc:
        print(
            "TP4_PAGEPOOL_DIAG_ERROR "
            + json.dumps(
                {
                    "event": _diag_count,
                    "tp_rank": int(getattr(self, "tp_rank", -1)),
                    "error": f"{type(exc).__name__}: {exc}",
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return result


SchedulerRuntimeCheckerMixin._check_all_pools = _check_all_pools_with_page_diag

print(
    "[K100 DFlash TP4 pagepooldiag] installed read-only full-pool page-set trace; strict behavior unchanged",
    flush=True,
)
