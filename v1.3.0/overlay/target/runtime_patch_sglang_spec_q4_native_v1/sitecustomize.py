"""Speculative q<=4 native paged-varlen repair for gfx928.

Parent: production INT8 stack with conservative selective paged-varlen Triton repair.
Evidence from the sentinel-write gate shows native gfx928 paged varlen is correct for
q=1..4 and the vendor no-write branch starts at q>=5. EAGLE2/EAGLE4 verification
uses q=2/q=4, so route only q<=4 back to the native interface while preserving the
proven Triton correctness fallback for q>=5.
"""
from __future__ import annotations

import runpy

_ROOT = "/data/qwen38-27b-k100ai-int8-opt"
runpy.run_path(f"{_ROOT}/runtime_patch_sglang_int8_prod_v1/sitecustomize.py", run_name="__q38_prod_parent__")

from sglang.srt.layers.attention import flashattention_backend as _fab
from sglang.srt.layers.attention import flashattention_interface as _fai

_prev = _fab.vllm_flash_attn_varlen_func
_native = _fai.vllm_flash_attn_varlen_func_interface
_seen_native: set[int] = set()
_seen_fallback = False


def _q4_native_else_corrected(
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
    global _seen_fallback
    qlen = int(max_seqlen_q)
    if 1 <= qlen <= 4:
        if qlen not in _seen_native:
            _seen_native.add(qlen)
            print(f"[K100 SGLang spec q4-native] ACTIVE native paged-varlen q={qlen}", flush=True)
        return _native(
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
    if not _seen_fallback:
        _seen_fallback = True
        print(f"[K100 SGLang spec q4-native] q={qlen} stays on corrected Triton fallback", flush=True)
    return _prev(
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


_fai.vllm_flash_attn_varlen_func = _q4_native_else_corrected
_fab.vllm_flash_attn_varlen_func = _q4_native_else_corrected
print("[K100 SGLang spec q4-native] installed: q<=4 native, q>=5 corrected fallback", flush=True)
