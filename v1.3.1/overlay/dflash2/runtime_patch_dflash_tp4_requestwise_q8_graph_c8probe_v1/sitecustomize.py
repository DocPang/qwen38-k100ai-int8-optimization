"""TP4 DFlash request-wise native-q8 CUDA-graph candidate v1.

Parent: current v1.2.2 runtime_patch_tp4 stack.

Exact target-verify bs2/bs3 q8 batches are decomposed into independent batch1
views and delegated to the already-audited parent native-q8 paged-attention
fast path.  This preserves each request's canonical c1 attention implementation
while avoiding the pathological generic batch paged-varlen verifier.

Graph contract:
- exact TP4 QH6/KVH1/D256 BF16 page64 causal full attention only;
- bs2..bs8 and q=8/request only;
- no GPU->CPU scalar reads, marker checks or dynamic allocations outside normal
  graph capture/replay semantics;
- all other shapes inherit parent unchanged.

This v1 intentionally applies the exact path at all KV lengths.  Promotion
requires both short-context and long-context concurrency gates.
"""
from __future__ import annotations

import runpy

_ROOT = "/data/qwen38-dflash2-k100ai"
runpy.run_path(f"{_ROOT}/runtime_patch_tp4/sitecustomize.py", run_name="__tp4_requestwise_q8_graph_parent__")

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

    _parent = _fab.vllm_flash_attn_varlen_func
    _cuq_cache: dict[int | None, torch.Tensor] = {}
    _active_logged: set[int] = set()

    def _cuq8(device: torch.device) -> torch.Tensor:
        key = device.index
        t = _cuq_cache.get(key)
        if t is None:
            t = torch.tensor([0, 8], device=device, dtype=torch.int32)
            _cuq_cache[key] = t
        return t

    def _route(
        q, k, v, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
        softmax_scale, causal, window_size, block_table, fa_version,
        q_descale, k_descale, v_descale,
    ):
        bs = int(cu_seqlens_q.numel() - 1)
        exact = (
            2 <= bs <= 8
            and int(max_seqlen_q) == 8
            and tuple(q.shape) == (bs * 8, 6, 256)
            and q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16
            and int(k.ndim) == 4 and int(v.ndim) == 4
            and tuple(k.shape[1:]) == (1, 64, 256)
            and tuple(v.shape[1:]) == (1, 256, 64)
            and int(seqused_k.numel()) == bs
            and bool(causal) and window_size == (-1, -1)
            and block_table is not None and int(fa_version) == 2
            and (q_descale is None or int(q_descale.numel()) == 1)
            and (k_descale is None or int(k_descale.numel()) == 1)
            and (v_descale is None or int(v_descale.numel()) == 1)
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

        if bs not in _active_logged:
            _active_logged.add(bs)
            capturing = False
            try:
                capturing = bool(torch.cuda.is_current_stream_capturing())
            except Exception:
                pass
            print(
                f"[K100 TP4 requestwise-q8-c8probe-v1] ACTIVE bs={bs} capturing={int(capturing)}",
                flush=True,
            )

        parts = []
        cuq = _cuq8(q.device)
        for b in range(bs):
            parts.append(_parent(
                q=q[b * 8 : (b + 1) * 8], k=k, v=v,
                cu_seqlens_q=cuq, max_seqlen_q=8,
                seqused_k=seqused_k[b : b + 1], max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale, causal=causal,
                window_size=window_size, block_table=block_table[b : b + 1],
                fa_version=fa_version, q_descale=q_descale,
                k_descale=k_descale, v_descale=v_descale,
            ))
        return torch.cat(parts, dim=0)

    _fai.vllm_flash_attn_varlen_func = _route
    _fab.vllm_flash_attn_varlen_func = _route
    print(
        "[K100 TP4 requestwise-q8-c8probe-v1] installed exact bs2..8 q8 -> per-request parent native-q8; all else parent",
        flush=True,
    )
