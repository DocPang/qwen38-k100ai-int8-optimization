"""TP4 v29 short-prefill paged-attention runtime block-table stride v2.

Parent: current request-wise q8/v29 chain.

Root cause addressed:
SourceFind/SGLang triton_vllm_flash_attn marks BT_STRIDE_B as tl.constexpr.
For batch1 cold prefill BT_STRIDE_B == ceil(kv_len / page_size), so every new
short prompt page count recompiles the otherwise identical large paged-varlen
attention kernel (~5s first-seen penalty on K100AI).

This patch changes exactly BT_STRIDE_B from compile-time constexpr to a runtime
scalar in a private copy of the same vendor module, and disables Triton value/alignment specialization for that scalar.  The alternate consumer is
used only for exact cold batch1 TP4 prefill with 64 <= q == kv < 2048.
Everything else delegates to the fully composed current parent unchanged.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import runpy

_ROOT = "/data/qwen38-dflash2-k100ai"
runpy.run_path(
    f"{_ROOT}/runtime_patch_dflash_tp4_requestwise_q8_graph_v1/sitecustomize.py",
    run_name="__tp4_paged_btstride_runtime_parent__",
)

try:
    _cmd = open("/proc/self/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "replace")
except Exception:
    _cmd = ""
_helper = (
    "multiprocessing.resource_tracker" in _cmd
    or "/usr/local/bin/ninja" in _cmd
    or " ninja --version" in _cmd
    or ("cdll.LoadLibrary" in _cmd and "torchinductor" in _cmd)
)

if not _helper:
    import torch
    from sglang.srt.layers.attention import flashattention_backend as _fab
    from sglang.srt.layers.attention import flashattention_interface as _fai
    from sglang.srt.layers.attention import triton_vllm_flash_attn as _orig_mod

    _src_path = Path(_orig_mod.__file__)
    _src = _src_path.read_text()
    _anchor = "    BT_STRIDE_B: tl.constexpr,\n"
    if _src.count(_anchor) != 1:
        raise RuntimeError(
            f"paged-btstride-runtime-v2 expected exactly one BT_STRIDE_B constexpr anchor in {_src_path}, "
            f"got {_src.count(_anchor)}"
        )
    _fixed = _src.replace(_anchor, "    BT_STRIDE_B,\n")
    _jit_anchor = "@triton.jit\ndef _paged_varlen_attn_fwd_kernel(\n"
    if _fixed.count(_jit_anchor) != 1:
        raise RuntimeError(f"paged-btstride-runtime-v2 expected one paged kernel jit anchor, got {_fixed.count(_jit_anchor)}")
    _fixed = _fixed.replace(
        _jit_anchor,
        "@triton.jit(do_not_specialize=[\"BT_STRIDE_B\"], "
        "do_not_specialize_on_alignment=[\"BT_STRIDE_B\"])\n"
        "def _paged_varlen_attn_fwd_kernel(\n",
    )
    _tmp = Path(f"/tmp/q38_paged_btstride_runtime_{os.getpid()}.py")
    _tmp.write_text(_fixed)
    _spec = importlib.util.spec_from_file_location(f"q38_paged_btstride_runtime_{os.getpid()}", _tmp)
    if _spec is None or _spec.loader is None:
        raise RuntimeError(f"cannot import generated {_tmp}")
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _runtime_stride = _mod.triton_vllm_flash_attn_varlen_func

    _parent = _fab.vllm_flash_attn_varlen_func
    _logged = False

    def _route(
        q, k, v, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
        softmax_scale, causal, window_size, block_table, fa_version,
        q_descale, k_descale, v_descale,
    ):
        global _logged
        qlen = int(max_seqlen_q)
        klen = int(max_seqlen_k)
        exact = (
            64 <= qlen < 2048
            and qlen == klen
            and int(cu_seqlens_q.numel()) == 2
            and int(q.shape[0]) == qlen
            and int(q.ndim) == 3
            and tuple(q.shape[1:]) == (6, 256)
            and q.dtype == torch.bfloat16
            and k.dtype == torch.bfloat16
            and v.dtype == torch.bfloat16
            and block_table is not None
            and bool(causal)
            and tuple(window_size) == (-1, -1)
        )
        if not exact:
            return _parent(
                q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q, seqused_k=seqused_k,
                max_seqlen_k=max_seqlen_k, softmax_scale=softmax_scale,
                causal=causal, window_size=window_size, block_table=block_table,
                fa_version=fa_version, q_descale=q_descale,
                k_descale=k_descale, v_descale=v_descale,
            )
        if not _logged:
            _logged = True
            print(
                "[K100 TP4 paged-btstride-runtime-v2] ACTIVE cold batch1 q=kv<2048; "
                "BT_STRIDE_B runtime scalar; all other paths parent",
                flush=True,
            )
        return _runtime_stride(
            q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q, seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k, softmax_scale=softmax_scale,
            causal=causal, window_size=window_size, block_table=block_table,
            fa_version=fa_version, q_descale=q_descale,
            k_descale=k_descale, v_descale=v_descale,
        )

    _fai.vllm_flash_attn_varlen_func = _route
    _fab.vllm_flash_attn_varlen_func = _route
    print(
        "[K100 TP4 paged-btstride-runtime-v2] installed: exact cold batch1 64<=q=kv<2048 only",
        flush=True,
    )
