"""TP2 corrected layer-select parent + DFlash2 q8 -> 2x native q4 verifier.

Only batch1 causal q=8 speculative target verification is changed. The first
q4 half sees K-4 keys and the second sees full K, preserving absolute causal
positions. All other attention shapes delegate to the corrected TP2 parent.
"""
from __future__ import annotations

import runpy
import torch

_ROOT = "/data/qwen38-27b-k100ai-int8-opt"
_PARENT = f"{_ROOT}/runtime_patch_sglang_tp2_shared_scale_layerselect_v1/sitecustomize.py"
runpy.run_path(_PARENT, run_name="__q38_tp2_layerselect_parent_for_dflash2_q8split__")

from sglang.srt.layers.attention import flashattention_backend as _fab
from sglang.srt.layers.attention import flashattention_interface as _fai

_parent = _fab.vllm_flash_attn_varlen_func
_native = _fai.vllm_flash_attn_varlen_func_interface
_seen = False
_cuq_cache: dict[tuple[int | None, int], torch.Tensor] = {}


def _cuq(device: torch.device, qlen: int) -> torch.Tensor:
    key = (device.index, qlen)
    t = _cuq_cache.get(key)
    if t is None:
        t = torch.tensor([0, qlen], device=device, dtype=torch.int32)
        _cuq_cache[key] = t
    return t


def _native_call(
    *, q, k, v, qlen, seqused_k, max_seqlen_k, softmax_scale, causal,
    window_size, block_table, fa_version, q_descale, k_descale, v_descale,
):
    return _native(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=_cuq(q.device, qlen),
        max_seqlen_q=qlen,
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


def _q8_split_native_else_parent(
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
    global _seen
    qlen = int(max_seqlen_q)
    batch = int(cu_seqlens_q.numel() - 1)
    eligible = (
        qlen == 8
        and int(q.shape[0]) == 8
        and batch == 1
        and int(seqused_k.numel()) == 1
        and bool(causal)
        and window_size == (-1, -1)
        and block_table is not None
    )
    if not eligible:
        return _parent(
            q=q, k=k, v=v,
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

    if not _seen:
        _seen = True
        print(
            "[K100 DFlash2 TP2 q8split] ACTIVE batch1 causal q=8 -> 2x native q=4",
            flush=True,
        )

    seq_k_first = seqused_k - 4
    out0 = _native_call(
        q=q[:4], k=k, v=v, qlen=4,
        seqused_k=seq_k_first,
        max_seqlen_k=max(1, int(max_seqlen_k) - 4),
        softmax_scale=softmax_scale, causal=causal,
        window_size=window_size, block_table=block_table, fa_version=fa_version,
        q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
    )
    out1 = _native_call(
        q=q[4:], k=k, v=v, qlen=4,
        seqused_k=seqused_k,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale, causal=causal,
        window_size=window_size, block_table=block_table, fa_version=fa_version,
        q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
    )
    if isinstance(out0, (tuple, list)) or isinstance(out1, (tuple, list)):
        raise RuntimeError("TP2 DFlash2 q8split expected tensor outputs")
    return torch.cat((out0, out1), dim=0)


_fai.vllm_flash_attn_varlen_func = _q8_split_native_else_parent
_fab.vllm_flash_attn_varlen_func = _q8_split_native_else_parent
print(
    "[K100 DFlash2 TP2 q8split v1] installed after corrected layer-select parent; "
    "only batch1 causal q8 is split",
    flush=True,
)
