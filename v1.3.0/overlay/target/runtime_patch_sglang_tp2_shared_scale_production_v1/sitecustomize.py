"""Production-oriented TP2 shared dynamic token-scale repair v1.

Parent stack:
  runtime_patch_sglang_tp2_row_ldsx_v1_paged_varlen_triton_only

Correctness contract derived from 2026-08-20 causal bisections:
  * both RowParallel W8A8 families Klocal=3072 and Klocal=8704 need a
    TP-equivalent per-token dynamic activation scale during prefill;
  * after prefill, only the first autoregressive DECODE forward needs the
    shared scale (N=0 fails case15, N=1 restores the TP1-minimal failure set);
  * later decode forwards must stay on the existing fast/local-scale path.

Unlike the diagnostic global-sentinel patch, this implementation tracks stable
ForwardBatch.rids per ModelRunner.  Pure DECODE batches use a per-row mask so a
batch may contain both first-decode and already-running requests.  MIXED batches
are conservatively shared for all rows, preserving correctness while avoiding
ambiguous token-layout inference.  First-decode batches bypass CUDA Graph once;
subsequent decode returns to the parent graph/M1 fast path.

Selected shared quantization uses Triton absmax + quant-with-selected-scale
kernels around one TP MAX collective.  The stock/lmslim quantization path is
untouched when no repair is needed.
"""
from __future__ import annotations

import os
import runpy
from typing import Optional

_ROOT = "/data/qwen38-27b-k100ai-int8-opt"
_PARENT = f"{_ROOT}/runtime_patch_sglang_tp2_row_ldsx_v1_paged_varlen_triton_only/sitecustomize.py"
runpy.run_path(_PARENT, run_name="__q38_tp2_shared_scale_production_parent__")

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from triton.language.extra import libdevice

import sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 as _w8
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner

_TARGET_K = frozenset({3072, 8704})
_orig_apply_weights = _w8.CompressedTensorsW8A8Int8.apply_weights
_orig_forward_raw = ModelRunner._forward_raw

# Process-local model-forward context. Each TP rank is a separate process and
# SGLang's model execution in a worker is serialized at this Python boundary.
_ctx_full_shared: bool = False
_ctx_decode_mask: Optional[torch.Tensor] = None
_ctx_label: str = "none"
_seen_diag: set[tuple] = set()


@triton.jit
def _row_absmax_kernel(
    x_ptr,
    absmax_ptr,
    stride_x,
    n_cols: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x)), 1.0e-10)
    tl.store(absmax_ptr + row, absmax)


@triton.jit
def _quant_selected_scale_kernel(
    x_ptr,
    xq_ptr,
    scale_ptr,
    local_absmax_ptr,
    shared_absmax_ptr,
    row_mask_ptr,
    stride_x,
    stride_xq,
    n_cols: tl.constexpr,
    BLOCK: tl.constexpr,
    FULL_SHARED: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    col_mask = cols < n_cols
    x = tl.load(x_ptr + row * stride_x + cols, mask=col_mask, other=0.0).to(tl.float32)
    local_absmax = tl.load(local_absmax_ptr + row)
    shared_absmax = tl.load(shared_absmax_ptr + row)
    if FULL_SHARED:
        absmax = shared_absmax
    elif HAS_MASK:
        use_shared = tl.load(row_mask_ptr + row) != 0
        absmax = tl.where(use_shared, shared_absmax, local_absmax)
    else:
        absmax = local_absmax
    scale = absmax / 127.0
    x_q = x * (127.0 / absmax)
    x_q = libdevice.nearbyint(x_q).to(tl.int8)
    tl.store(xq_ptr + row * stride_xq + cols, x_q, mask=col_mask)
    tl.store(scale_ptr + row, scale)


def _shared_quant_selected(x: torch.Tensor, full_shared: bool, row_mask: Optional[torch.Tensor]):
    if x.ndim != 2:
        raise RuntimeError(f"TP2 production shared-scale expects 2D RowParallel input, got {tuple(x.shape)}")
    m, k = int(x.shape[0]), int(x.shape[1])
    if k not in _TARGET_K:
        raise RuntimeError(f"unexpected shared-scale Klocal={k}")
    if not full_shared:
        if row_mask is None or row_mask.ndim != 1 or int(row_mask.numel()) != m:
            raise RuntimeError(
                f"first-decode row mask mismatch: M={m}, mask={None if row_mask is None else tuple(row_mask.shape)}"
            )
        if row_mask.dtype != torch.uint8:
            row_mask = row_mask.to(torch.uint8)

    local_absmax = torch.empty((m, 1), dtype=torch.float32, device=x.device)
    shared_absmax = torch.empty_like(local_absmax)
    x_q = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty((m, 1), dtype=torch.float32, device=x.device)
    block = triton.next_power_of_2(k)
    num_warps = min(max(block // 256, 1), 8)

    _row_absmax_kernel[(m,)](
        x,
        local_absmax,
        stride_x=x.stride(0),
        n_cols=k,
        BLOCK=block,
        num_warps=num_warps,
        num_stages=1,
    )
    shared_absmax.copy_(local_absmax)
    group = get_tp_group()
    dist.all_reduce(shared_absmax, op=dist.ReduceOp.MAX, group=group.device_group)

    # Triton requires a pointer argument even when HAS_MASK is constexpr false.
    mask_arg = row_mask if row_mask is not None else local_absmax.view(-1).to(torch.uint8)
    _quant_selected_scale_kernel[(m,)](
        x,
        x_q,
        scales,
        local_absmax,
        shared_absmax,
        mask_arg,
        stride_x=x.stride(0),
        stride_xq=x_q.stride(0),
        n_cols=k,
        BLOCK=block,
        FULL_SHARED=bool(full_shared),
        HAS_MASK=bool(row_mask is not None and not full_shared),
        num_warps=num_warps,
        num_stages=1,
    )
    return x_q, scales


def _patched_apply_weights(self, layer, x, bias, input_quant_args=None, silu_quant_args=None):
    global _ctx_full_shared, _ctx_decode_mask, _ctx_label
    selected_layer = (
        isinstance(x, torch.Tensor)
        and x.ndim >= 2
        and layer.__class__.__name__ == "RowParallelLinear"
        and int(getattr(layer, "tp_size", 1)) > 1
        and int(x.shape[-1]) in _TARGET_K
    )
    needs_shared = bool(_ctx_full_shared or _ctx_decode_mask is not None)
    if not selected_layer or not needs_shared:
        return _orig_apply_weights(
            self,
            layer,
            x,
            bias,
            input_quant_args=input_quant_args,
            silu_quant_args=silu_quant_args,
        )

    # For the selected repair calls we intentionally override any fused local
    # quantization args. Later decode forwards never enter this branch and keep
    # the parent's fused/GEMV fast path unchanged.
    x_q, x_scale = _shared_quant_selected(x, _ctx_full_shared, _ctx_decode_mask)
    key = (_ctx_label, int(x.shape[0]), int(x.shape[-1]))
    if key not in _seen_diag:
        _seen_diag.add(key)
        try:
            rank = int(getattr(get_tp_group(), "rank_in_group", -1))
        except Exception:
            rank = -1
        nshared = int(x.shape[0]) if _ctx_full_shared else int(_ctx_decode_mask.sum().item())
        print(
            "[K100 TP2 production shared scale v1] ACTIVE "
            f"rank={rank} mode={_ctx_label} M={x.shape[0]} Klocal={x.shape[-1]} shared_rows={nshared}",
            flush=True,
        )

    # Reuse the original GEMM selection exactly; temporarily substitute the
    # quantizer so apply_weights consumes our shared-scale x_q/x_scale.
    old_quant = _w8.per_token_quant_int8
    _w8.per_token_quant_int8 = lambda _x: (x_q, x_scale)
    try:
        return _orig_apply_weights(
            self,
            layer,
            x,
            bias,
            input_quant_args=None,
            silu_quant_args=None,
        )
    finally:
        _w8.per_token_quant_int8 = old_quant


_w8.CompressedTensorsW8A8Int8.apply_weights = _patched_apply_weights


def _ensure_state(runner: ModelRunner):
    if not hasattr(runner, "_q38_shared_known_rids"):
        runner._q38_shared_known_rids = set()
        runner._q38_shared_pending_first_decode = set()


def _patched_forward_raw(self: ModelRunner, forward_batch, *args, **kwargs):
    global _ctx_full_shared, _ctx_decode_mask, _ctx_label
    _ensure_state(self)
    known = self._q38_shared_known_rids
    pending = self._q38_shared_pending_first_decode
    rids = list(forward_batch.rids or [])
    mode = forward_batch.forward_mode

    full_shared = bool(
        mode == ForwardMode.EXTEND
        or mode == ForwardMode.MIXED
        or mode == ForwardMode.SPLIT_PREFILL
    )
    first_rids: list[str] = []
    decode_mask = None
    if mode == ForwardMode.DECODE:
        first_rids = [rid for rid in rids if rid in pending]
        if first_rids:
            mask_cpu = [1 if rid in pending else 0 for rid in rids]
            decode_mask = torch.tensor(mask_cpu, dtype=torch.uint8, device=forward_batch.input_ids.device)

    # Make graph decisions explicit. Piecewise prefill graph must not capture
    # the repair path, and a batch containing any first-decode request must run
    # eager exactly once. Regular later DECODE batches keep normal graph replay.
    forward_batch._q38_shared_scale_full_batch = full_shared
    forward_batch._q38_force_eager_first_decode = bool(first_rids)

    _ctx_full_shared = full_shared
    _ctx_decode_mask = decode_mask
    _ctx_label = "mixed" if mode == ForwardMode.MIXED else (
        "prefill" if full_shared else ("first_decode" if first_rids else "none")
    )
    success = False
    try:
        out = _orig_forward_raw(self, forward_batch, *args, **kwargs)
        success = True
        return out
    finally:
        _ctx_full_shared = False
        _ctx_decode_mask = None
        _ctx_label = "none"
        if success:
            if mode == ForwardMode.EXTEND or mode == ForwardMode.SPLIT_PREFILL:
                pending.update(rids)
                known.update(rids)
            elif mode == ForwardMode.MIXED:
                # New prefill requests are unseen; already-pending rids are
                # chunked-prefill requests. Known non-pending rids are the
                # running decode side merged into this MIXED batch.
                for rid in rids:
                    if rid not in known or rid in pending:
                        pending.add(rid)
                    known.add(rid)
            elif mode == ForwardMode.DECODE:
                for rid in first_rids:
                    pending.discard(rid)
                known.update(rids)


ModelRunner._forward_raw = _patched_forward_raw

# Full CUDA graph: force eager only for the one first-decode batch.
try:
    from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
    _orig_cg_can_run = CudaGraphRunner.can_run

    def _patched_cg_can_run(self, forward_batch):
        if getattr(forward_batch, "_q38_force_eager_first_decode", False):
            return False
        return _orig_cg_can_run(self, forward_batch)

    CudaGraphRunner.can_run = _patched_cg_can_run
except Exception as exc:
    print(f"[K100 TP2 production shared scale v1] cuda-graph can_run patch skipped: {exc}", flush=True)

# Piecewise graph: repaired prefill/mixed batches must execute the dynamic
# shared-scale kernels rather than replay a captured local-scale graph.
try:
    from sglang.srt.model_executor.piecewise_cuda_graph_runner import PiecewiseCudaGraphRunner
    _orig_pcg_can_run = PiecewiseCudaGraphRunner.can_run

    def _patched_pcg_can_run(self, forward_batch):
        if getattr(forward_batch, "_q38_shared_scale_full_batch", False):
            return False
        return _orig_pcg_can_run(self, forward_batch)

    PiecewiseCudaGraphRunner.can_run = _patched_pcg_can_run
except Exception as exc:
    print(f"[K100 TP2 production shared scale v1] piecewise-graph can_run patch skipped: {exc}", flush=True)

print(
    "[K100 TP2 production shared scale v1] installed: corrected row-LDSx parent; "
    "both-family prefill/mixed shared; request-aware first-decode row mask; "
    "first-decode eager once, later decode graph/local fast path preserved",
    flush=True,
)
