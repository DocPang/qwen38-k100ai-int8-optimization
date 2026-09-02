"""Diagnostic-only TP4 DFlash paged-KV ownership trace.

Loads the exact current unified-adaptive parent unchanged, then records the tiny
metadata needed to map an idle strict-memory-check missing physical page back to
its DFlash verify round/request.  No model tensor is modified and no allocator
operation is intercepted.
"""
from __future__ import annotations

import os
import runpy

_ROOT = "/data/qwen38-dflash2-k100ai"
runpy.run_path(
    f"{_ROOT}/runtime_patch_dflash_tp4_shortq_onepass_v2_layerhybrid/sitecustomize.py",
    run_name="__tp4_pageownerdiag_parent__",
)

import torch
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.speculative.dflash_info import DFlashDraftInput, DFlashVerifyInput

_VERIFY_LIMIT = int(os.getenv("SGLANG_DFLASH_PAGEOWNER_VERIFY_ROUNDS", "256"))
_FILTER_LIMIT = int(os.getenv("SGLANG_DFLASH_PAGEOWNER_FILTER_EVENTS", "128"))
_verify_round = 0
_filter_event = 0

_orig_verify = DFlashVerifyInput.verify
_orig_draft_filter = DFlashDraftInput.filter_batch
_orig_batch_filter = ScheduleBatch.filter_batch


def _safe_cpu_list(x):
    if x is None:
        return None
    try:
        if isinstance(x, torch.Tensor):
            return x.detach().to("cpu").tolist()
        return list(x)
    except Exception as exc:
        return [f"<diag-error:{type(exc).__name__}:{exc}>"]


def _req_meta(batch):
    rows = []
    for i, req in enumerate(batch.reqs):
        rows.append(
            {
                "i": i,
                "rid": str(getattr(req, "rid", "?")),
                "req_pool_idx": None
                if getattr(req, "req_pool_idx", None) is None
                else int(req.req_pool_idx),
                "committed": int(getattr(req, "kv_committed_len", -1)),
                "allocated": int(getattr(req, "kv_allocated_len", -1)),
                "cache_protected": int(getattr(req, "cache_protected_len", -1)),
                "finished": bool(req.finished()),
            }
        )
    return rows


def _verify_diag(self, *, batch, logits_output, page_size):
    global _verify_round
    do_log = _verify_round < _VERIFY_LIMIT and not batch.forward_mode.is_idle()
    if do_log:
        bs = int(batch.batch_size())
        draft_n = int(self.draft_token_num)
        loc = batch.out_cache_loc.detach().clone()
        if int(loc.numel()) == bs * draft_n:
            loc2 = loc.view(bs, draft_n)
            pages = (loc2 // int(page_size)).to(torch.int64)
            offsets = (loc2 % int(page_size)).to(torch.int64)
            page_rows = _safe_cpu_list(pages)
            offset_rows = _safe_cpu_list(offsets)
        else:
            page_rows = [f"<unexpected-out-cache-loc-numel:{int(loc.numel())}>"]
            offset_rows = None
        pre = {
            "seq_lens": _safe_cpu_list(batch.seq_lens),
            "req_pool_indices": _safe_cpu_list(batch.req_pool_indices),
            "reqs": _req_meta(batch),
            "pages": page_rows,
            "offsets": offset_rows,
        }
    out = _orig_verify(
        self, batch=batch, logits_output=logits_output, page_size=page_size
    )
    if do_log:
        new_bonus, commit_lens, _next_hidden, accepted = out
        print(
            "TP4_PAGEOWNER_VERIFY "
            f"round={_verify_round + 1} page_size={int(page_size)} "
            f"pre={pre} commit_lens={_safe_cpu_list(commit_lens)} "
            f"accepted={accepted} new_bonus={_safe_cpu_list(new_bonus)} "
            f"post_reqs={_req_meta(batch)} post_seq_lens={_safe_cpu_list(batch.seq_lens)}",
            flush=True,
        )
        _verify_round += 1
    return out


def _draft_filter_diag(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
    global _filter_event
    do_log = _filter_event < _FILTER_LIMIT
    if do_log:
        before = {
            "bonus_len": int(self.bonus_tokens.shape[0]),
            "ctx_lens": _safe_cpu_list(self.ctx_lens),
            "draft_seq_lens": _safe_cpu_list(self.draft_seq_lens),
            "new_indices": _safe_cpu_list(new_indices),
            "has_been_filtered": bool(has_been_filtered),
        }
    out = _orig_draft_filter(
        self, new_indices=new_indices, has_been_filtered=has_been_filtered
    )
    if do_log:
        print(
            "TP4_PAGEOWNER_DRAFT_FILTER "
            f"event={_filter_event + 1} before={before} "
            f"after_bonus_len={int(self.bonus_tokens.shape[0])} "
            f"after_ctx_lens={_safe_cpu_list(self.ctx_lens)} "
            f"after_draft_seq_lens={_safe_cpu_list(self.draft_seq_lens)}",
            flush=True,
        )
        _filter_event += 1
    return out


def _batch_filter_diag(
    self,
    chunked_req_to_exclude=None,
    keep_indices=None,
    v1_spec_info_filtered: bool = False,
):
    is_dflash = isinstance(getattr(self, "spec_info", None), DFlashDraftInput)
    if is_dflash and _filter_event < _FILTER_LIMIT:
        before = {
            "reqs": _req_meta(self),
            "seq_lens": _safe_cpu_list(self.seq_lens),
            "keep_indices_arg": keep_indices,
            "v1_spec_info_filtered": bool(v1_spec_info_filtered),
        }
    else:
        before = None
    out = _orig_batch_filter(
        self,
        chunked_req_to_exclude=chunked_req_to_exclude,
        keep_indices=keep_indices,
        v1_spec_info_filtered=v1_spec_info_filtered,
    )
    if before is not None:
        print(
            "TP4_PAGEOWNER_BATCH_FILTER "
            f"before={before} after_reqs={_req_meta(self)} "
            f"after_seq_lens={_safe_cpu_list(self.seq_lens)}",
            flush=True,
        )
    return out


DFlashVerifyInput.verify = _verify_diag
DFlashDraftInput.filter_batch = _draft_filter_diag
ScheduleBatch.filter_batch = _batch_filter_diag

print(
    "[K100 DFlash TP4 pageownerdiag] installed read-only verify/filter page trace",
    flush=True,
)
