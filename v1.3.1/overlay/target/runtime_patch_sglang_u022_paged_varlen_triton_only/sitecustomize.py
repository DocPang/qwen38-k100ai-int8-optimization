"""U022 production stack + selective gfx928 paged-varlen correctness repair.

The native gfx928 vLLM-style paged varlen consumer is catastrophically wrong for
multi-token Q on the DCU_FA physical64 layout, while qLen=1 paged decode is the
fast path we want to preserve.  This child patch therefore changes exactly one
consumer:

  * vllm_flash_attn_varlen_func      -> stride-aware Triton reference
  * vllm_flash_attn_with_kvcache    -> native SourceFind/DCU implementation

The parent U022 W8A8/GEMV/compact-head stack is unchanged.  The global
SGLANG_USE_TRITON_VLLM_FA switch MUST stay off so this patch cannot silently
replace decode as well.
"""
from __future__ import annotations

import os
import runpy

_ROOT = "/data/qwen38-27b-k100ai-int8-opt"
_PARENT = f"{_ROOT}/runtime_patch_sglang_u022_k5120_ldsx/sitecustomize.py"

_flag = os.getenv("SGLANG_USE_TRITON_VLLM_FA", "0").strip().lower()
if _flag not in ("", "0", "false", "no", "off"):
    raise RuntimeError(
        "selective paged-varlen patch requires SGLANG_USE_TRITON_VLLM_FA=0; "
        f"got {_flag!r}. Global Triton would also replace native decode."
    )

runpy.run_path(_PARENT, run_name="__q38_sglang_u022_parent__")

from sglang.srt.layers.attention import flashattention_backend as _fab
from sglang.srt.layers.attention import flashattention_interface as _fai
from sglang.srt.layers.attention.triton_vllm_flash_attn import (
    triton_vllm_flash_attn_varlen_func as _triton_paged_varlen,
)

_seen = False


def _q38_triton_paged_varlen_only(
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
            "[K100 SGLang gfx928 paged repair] ACTIVE Triton vLLM varlen only; "
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


# flashattention_backend imports these symbols by value, so patch both the
# interface module and the already-imported backend binding.
_fai.vllm_flash_attn_varlen_func = _q38_triton_paged_varlen_only
_fab.vllm_flash_attn_varlen_func = _q38_triton_paged_varlen_only

print(
    "[K100 SGLang gfx928 paged repair] installed selective consumer: "
    "paged-varlen=Triton, with-kvcache/decode=native",
    flush=True,
)
