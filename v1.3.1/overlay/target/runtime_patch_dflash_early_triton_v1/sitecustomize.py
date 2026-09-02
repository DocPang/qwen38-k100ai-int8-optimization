"""DFlash2 early-round Triton verifier repair for gfx928 TP1.

Parent: proven DFlash2 q8split stack (q=8 -> 2x native q=4).

Problem: full selective Triton on layers 7/15 restores short greedy semantics,
but long-KV verification becomes prohibitively expensive because every DFlash
round pays long-context Triton attention.  This patch uses corrected Triton only
for the first N TARGET_VERIFY rounds of each real request, then returns to the
already-captured fast native q8split CUDA graph.

Contract:
  SGLANG_DFLASH_EARLY_TRITON_LAYERS=7,15
  SGLANG_DFLASH_EARLY_TRITON_ROUNDS=8
  max_running_requests=1 (runtime fail-closed if a real TARGET_VERIFY batch has
  more than one rid).

CUDA graph capture itself has no real request rid, so it deliberately captures
the parent native q8split path.  Real early rounds are forced eager; later rounds
resume graph replay.
"""
from __future__ import annotations

import os
import runpy

_DFLASH_ROOT = "/data/qwen38-dflash2-k100ai"
runpy.run_path(
    f"{_DFLASH_ROOT}/runtime_patch_dflash_q8split_v1/sitecustomize.py",
    run_name="__dflash_early_triton_parent__",
)

from sglang.srt.layers.attention import flashattention_backend as _fab
from sglang.srt.layers.attention import flashattention_interface as _fai
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.attention.triton_vllm_flash_attn import (
    triton_vllm_flash_attn_varlen_func as _triton_paged_varlen,
)
from sglang.srt.model_executor.model_runner import ModelRunner

_raw_layers = os.getenv("SGLANG_DFLASH_EARLY_TRITON_LAYERS", "").strip()
_raw_rounds = os.getenv("SGLANG_DFLASH_EARLY_TRITON_ROUNDS", "").strip()
if not _raw_layers:
    raise RuntimeError("early-Triton requires SGLANG_DFLASH_EARLY_TRITON_LAYERS")
if not _raw_rounds:
    raise RuntimeError("early-Triton requires SGLANG_DFLASH_EARLY_TRITON_ROUNDS")
try:
    _layers = {int(x.strip()) for x in _raw_layers.split(",") if x.strip()}
    _rounds = int(_raw_rounds)
except ValueError as exc:
    raise RuntimeError("invalid early-Triton layers/rounds") from exc
_allowed = set(range(3, 64, 4))
if not _layers or not _layers.issubset(_allowed):
    raise RuntimeError(
        f"invalid early-Triton layers={sorted(_layers)}; allowed={sorted(_allowed)}"
    )
if _rounds < 1 or _rounds > 64:
    raise RuntimeError(f"early-Triton rounds must be 1..64, got {_rounds}")

_STATE_MAX_RIDS = 4096
_STATE_TTL_FORWARDS = 65536
_STATE_PRUNE_INTERVAL = 256

_parent_varlen = _fab.vllm_flash_attn_varlen_func
_parent_forward_extend = FlashAttentionBackend.forward_extend
_orig_forward_raw = ModelRunner._forward_raw
_seen_layers: set[int] = set()
_seen_rounds: set[int] = set()


def _ensure_state(runner: ModelRunner) -> None:
    if not hasattr(runner, "_q38_dflash_verify_round_by_rid"):
        runner._q38_dflash_verify_round_by_rid = {}
        runner._q38_dflash_rid_last_seen = {}
        runner._q38_dflash_forward_epoch = 0


def _prune(runner: ModelRunner, current_rids: set[str]) -> None:
    epoch = runner._q38_dflash_forward_epoch
    if epoch % _STATE_PRUNE_INTERVAL != 0:
        return
    rounds = runner._q38_dflash_verify_round_by_rid
    last = runner._q38_dflash_rid_last_seen
    cutoff = epoch - _STATE_TTL_FORWARDS
    stale = [rid for rid, seen in last.items() if seen < cutoff and rid not in current_rids]
    for rid in stale:
        last.pop(rid, None)
        rounds.pop(rid, None)
    overflow = len(last) - _STATE_MAX_RIDS
    if overflow > 0:
        candidates = sorted((seen, rid) for rid, seen in last.items() if rid not in current_rids)
        for _, rid in candidates[:overflow]:
            last.pop(rid, None)
            rounds.pop(rid, None)


def _patched_forward_raw(self: ModelRunner, forward_batch, *args, **kwargs):
    _ensure_state(self)
    self._q38_dflash_forward_epoch += 1
    epoch = self._q38_dflash_forward_epoch
    rids = [str(x) for x in list(getattr(forward_batch, "rids", None) or [])]
    current = set(rids)
    _prune(self, current)

    is_verify = bool(forward_batch.forward_mode.is_target_verify())
    early = False
    rid = None
    round_idx = None
    if is_verify and rids:
        if len(rids) != 1:
            raise RuntimeError(
                f"early-Triton TP1 contract requires one real rid in TARGET_VERIFY, got {rids}"
            )
        rid = rids[0]
        round_idx = int(self._q38_dflash_verify_round_by_rid.get(rid, 0))
        early = round_idx < _rounds
        self._q38_dflash_rid_last_seen[rid] = epoch

    # Must be visible before _orig_forward_raw decides whether CUDA graph can run.
    forward_batch._q38_dflash_early_triton = early
    forward_batch._q38_dflash_early_triton_round = round_idx

    success = False
    try:
        out = _orig_forward_raw(self, forward_batch, *args, **kwargs)
        success = True
        return out
    finally:
        if success and is_verify and rid is not None:
            self._q38_dflash_verify_round_by_rid[rid] = int(round_idx) + 1
            self._q38_dflash_rid_last_seen[rid] = epoch


def _early_varlen(
    q,
    k,
    v,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    fa_version,
    q_descale,
    k_descale,
    v_descale,
):
    # Selection is set by the layer wrapper below.  Non-selected layers and all
    # graph-capture/non-real-request paths stay on the parent q8split behavior.
    if getattr(_early_varlen, "_active", False):
        return _triton_paged_varlen(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            block_table=block_table,
            fa_version=fa_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )
    return _parent_varlen(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        seqused_k=seqused_k,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        block_table=block_table,
        fa_version=fa_version,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )


def _forward_extend_early(self, q, k, v, layer, forward_batch, *args, **kwargs):
    lid = int(getattr(layer, "layer_id", -1))
    enabled = bool(
        forward_batch.forward_mode.is_target_verify()
        and getattr(forward_batch, "_q38_dflash_early_triton", False)
        and lid in _layers
    )
    prev = getattr(_early_varlen, "_active", False)
    _early_varlen._active = enabled
    try:
        if enabled:
            if lid not in _seen_layers:
                _seen_layers.add(lid)
                print(f"[K100 DFlash2 early-Triton] ACTIVE layer={lid}", flush=True)
            ri = getattr(forward_batch, "_q38_dflash_early_triton_round", None)
            if isinstance(ri, int) and ri not in _seen_rounds:
                _seen_rounds.add(ri)
                print(f"[K100 DFlash2 early-Triton] ACTIVE verify_round={ri}", flush=True)
        return _parent_forward_extend(self, q, k, v, layer, forward_batch, *args, **kwargs)
    finally:
        _early_varlen._active = prev


ModelRunner._forward_raw = _patched_forward_raw
_fai.vllm_flash_attn_varlen_func = _early_varlen
_fab.vllm_flash_attn_varlen_func = _early_varlen
FlashAttentionBackend.forward_extend = _forward_extend_early

try:
    from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
    _orig_cg_can_run = CudaGraphRunner.can_run

    def _patched_cg_can_run(self, forward_batch):
        if getattr(forward_batch, "_q38_dflash_early_triton", False):
            return False
        return _orig_cg_can_run(self, forward_batch)

    CudaGraphRunner.can_run = _patched_cg_can_run
except Exception as exc:
    raise RuntimeError(f"early-Triton failed to patch CUDA graph can_run: {exc}") from exc

print(
    "[K100 DFlash2 early-Triton] installed layers="
    + ",".join(str(x) for x in sorted(_layers))
    + f" rounds={_rounds}; real early rounds eager+Triton, later rounds native q8split graph",
    flush=True,
)
