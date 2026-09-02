"""TP2 DFlash2 long-KV v5: v4 prefill/qtail + exact raw q8 paged verifier.

Install the qualified v4 stack first.  Then replace only the batch1 causal
DFlash verifier shape q=8, QH12/KVH2/D256/BF16/page64/fa2 with one raw
flash_attn_2_cuda.paged_attention call.  Isolated 64K/128K/257.9K gates show
bitwise equality versus the existing 2xq4 verifier and ~2.0-2.26x attention
speedup.  Every non-audited shape falls back to the v4/RC1 parent.
"""
from __future__ import annotations

import importlib.metadata as _metadata
import runpy
import torch
import flash_attn_2_cuda as _fa2

_ROOT = "/data/qwen38-27b-k100ai-int8-opt"
_PARENT = f"{_ROOT}/runtime_patch_sglang_tp2_dflash2_longkv_bm128_v4/sitecustomize.py"
runpy.run_path(_PARENT, run_name="__q38_tp2_dflash2_longkv_bm128_v5_parent__")

from sglang.srt.layers.attention import flashattention_backend as _fab
from sglang.srt.layers.attention import flashattention_interface as _fai

_parent = _fab.vllm_flash_attn_varlen_func
_seen = False
_FLASH_ATTN_VERSION = _metadata.version("flash-attn")
_FLASH_ATTN_LEGACY = "2.8.3+das.opt1.dtk2604.torch290.2606021702.ge93bd4"
_FLASH_ATTN_LAYOUT_ABI = "2.8.3+das.opt1.dtk2604.torch290.2607280958.gebb4be"
if _FLASH_ATTN_VERSION == _FLASH_ATTN_LEGACY:
    _RAW_PAGED_LAYOUT = None
elif _FLASH_ATTN_VERSION == _FLASH_ATTN_LAYOUT_ABI:
    # SourceFind 260728 adds a final layout enum to paged_attention():
    # 1=bhsd, 2=bshd.  This exact TP2 cache geometry is B,H,S,D.
    _RAW_PAGED_LAYOUT = 1
else:
    raise RuntimeError(
        "TP2 raw-q8 paged verifier has no audited flash-attn ABI for "
        f"version={_FLASH_ATTN_VERSION}"
    )
print(
    f"[K100 DFlash2 TP2 raw-q8 ABI] flash-attn={_FLASH_ATTN_VERSION} "
    f"layout_arg={'legacy' if _RAW_PAGED_LAYOUT is None else _RAW_PAGED_LAYOUT}",
    flush=True,
)


def _raw_q8_else_parent(
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
        and tuple(q.shape) == (8, 12, 256)
        and batch == 1
        and int(seqused_k.numel()) == 1
        and bool(causal)
        and window_size == (-1, -1)
        and block_table is not None
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and int(k.ndim) == 4
        and int(v.ndim) == 4
        and int(k.shape[1]) == 2
        and int(v.shape[1]) == 2
        and int(k.shape[2]) == 64
        and int(k.shape[3]) == 256
        and int(v.shape[2]) == 256
        and int(v.shape[3]) == 64
        and int(fa_version) == 2
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
            "[K100 DFlash2 TP2 raw-q8 v5] ACTIVE q=8 verifier -> single raw paged_attention "
            "QH12/KVH2/D256/BF16/page64",
            flush=True,
        )

    out = torch.empty_like(q)
    _paged_args = (
        out,
        q.reshape(1, 8, 12, 256),
        k,
        v,
        softmax_scale,
        block_table,
        seqused_k,
        None,
        "auto",
        q_descale,
        k_descale,
        v_descale,
        max_seqlen_k,
        None,
    )
    if _RAW_PAGED_LAYOUT is None:
        _fa2.paged_attention(*_paged_args)
    else:
        _fa2.paged_attention(*_paged_args, _RAW_PAGED_LAYOUT)
    return out


_fai.vllm_flash_attn_varlen_func = _raw_q8_else_parent
_fab.vllm_flash_attn_varlen_func = _raw_q8_else_parent

print(
    "[K100 DFlash2 TP2 longkv BM128 v5] installed: v4 prefill/qtail preserved; "
    "exact batch1 causal q8 verifier uses raw paged_attention; all else parent fallback",
    flush=True,
)
