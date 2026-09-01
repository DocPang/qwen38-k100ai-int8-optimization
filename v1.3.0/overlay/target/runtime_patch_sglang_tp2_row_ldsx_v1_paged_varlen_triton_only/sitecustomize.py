"""TP2 row-LDSx/compact-head stack + selective gfx928 paged-varlen repair.

This is the TP2 propagation of the validated U022 auto8k consumer mechanism:
  * native gfx928 multi-token paged varlen -> Triton correctness fallback
  * native with-kvcache/decode remains untouched
  * SGLang pack-paged-kv-to-varlen may bypass the fallback for thresholded
    long prefill chunks and use native contiguous varlen attention.

The parent TP2 row-LDSx stack is otherwise unchanged.  Global
SGLANG_USE_TRITON_VLLM_FA must remain off so decode is never silently replaced.
"""
from __future__ import annotations

import os
import runpy

_ROOT = "/data/qwen38-27b-k100ai-int8-opt"
_PARENT = f"{_ROOT}/runtime_patch_sglang_tp2_row_ldsx_v1/sitecustomize.py"

_flag = os.getenv("SGLANG_USE_TRITON_VLLM_FA", "0").strip().lower()
if _flag not in ("", "0", "false", "no", "off"):
    raise RuntimeError(
        "TP2 selective paged-varlen patch requires SGLANG_USE_TRITON_VLLM_FA=0; "
        f"got {_flag!r}. Global Triton would also replace native decode."
    )

runpy.run_path(_PARENT, run_name="__q38_sglang_tp2_row_ldsx_parent__")

from sglang.srt.layers.attention import flashattention_backend as _fab
from sglang.srt.layers.attention import flashattention_interface as _fai
from sglang.srt.layers.attention.triton_vllm_flash_attn import (
    triton_vllm_flash_attn_varlen_func as _triton_paged_varlen,
)

_seen = False


def _q38_tp2_triton_paged_varlen_only(
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
    if not _seen:
        _seen = True
        print(
            "[K100 SGLang TP2 gfx928 paged repair] ACTIVE Triton vLLM varlen fallback only; "
            "native with_kvcache/decode preserved",
            flush=True,
        )
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


_fai.vllm_flash_attn_varlen_func = _q38_tp2_triton_paged_varlen_only
_fab.vllm_flash_attn_varlen_func = _q38_tp2_triton_paged_varlen_only

print(
    "[K100 SGLang TP2 gfx928 paged repair] installed selective consumer: "
    "paged-varlen=Triton fallback, pack path may use native contiguous varlen, "
    "with-kvcache/decode=native",
    flush=True,
)
