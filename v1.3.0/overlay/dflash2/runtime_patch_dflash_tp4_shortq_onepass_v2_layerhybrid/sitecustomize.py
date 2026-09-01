"""TP4 DFlash2 adaptive short-Q packed one-pass Triton GQA v2 layer-hybrid.

Parent is the frozen LongCtx-v8 composition.  This patch changes only exact
batch1 cold-prefill full-attention shapes with q==kv in [2048,4095], TP4
rank-local Q6/KV1/D256/BF16/page64.  It retains SourceFind's audited page-table
gather, but replaces the native contiguous varlen consumer with the vendor
Triton MQA/GQA kernel using a one-pass causal q-block grid.

Why one-pass: the vendor two-pass causal mapping can duplicate or omit blocks
for non-multiples/odd ceil(q/BLOCK_M).  The already-proven ceil reverse-index
fix solves the q3948 even-block case but is insufficient for arbitrary short q.
One-pass keeps IS_CAUSAL=True and assigns each q block exactly once.

Everything outside the exact gate falls back to the frozen parent.
This is a relaxed-numerical candidate until full-model quality promotion.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import runpy

_DROOT = "/data/qwen38-dflash2-k100ai"
_TROOT = "/data/qwen38-27b-k100ai-int8-opt"
_PARENT = f"{_DROOT}/runtime_patch_dflash_tp4_q16k_agent128k_v1/sitecustomize.py"
_VENDOR = f"{_TROOT}/external_images/sglang0.5.12-20260620-src/flash_attn/flash_attn_triton_mqa_gqa.py"

try:
    _cmd = open("/proc/self/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "replace")
except Exception:
    _cmd = ""
_helper = (
    ("cdll.LoadLibrary" in _cmd and "torchinductor_tp4" in _cmd)
    or "multiprocessing.resource_tracker" in _cmd
    or "/usr/local/bin/ninja" in _cmd
    or " ninja --version" in _cmd
)

runpy.run_path(_PARENT, run_name="__q38_dflash_tp4_shortq_onepass_parent__")

if _helper:
    print(f"[K100 DFlash2 TP4 shortQ onepass helper bypass] pid={os.getpid()}", flush=True)
else:
    import torch
    from sglang.srt.layers.attention import pack_paged_kv_to_varlen as _packmod
    from sglang.srt.layers.attention import flashattention_backend as _fabackend

    _MIN_Q = int(os.getenv("SGLANG_Q38_TP4_SHORTQ_ONEPASS_MIN_Q", "2048"))
    _MAX_Q = int(os.getenv("SGLANG_Q38_TP4_SHORTQ_ONEPASS_MAX_Q", "4095"))
    _FALLBACK_LAYERS = {int(x) for x in os.getenv("SGLANG_Q38_TP4_SHORTQ_PARENT_LAYERS", "").split(",") if x.strip()}
    if not (1 <= _MIN_Q <= _MAX_Q < 8192):
        raise RuntimeError(f"shortQ onepass invalid range {_MIN_Q}..{_MAX_Q}")

    _raw = Path(_VENDOR).read_text()
    _old_ret = "            if start_m * BLOCK_M > seqlen_q:\n                return\n"
    _new_ret = "            if start_m * BLOCK_M > seqlen_q:\n                start_m = 0  # one-pass grid makes this unreachable\n"
    _old_rev = "            start_m = MAX_SEQLENS_Q // BLOCK_M -1 - tl.program_id(0)\n"
    _new_rev = "            start_m = (MAX_SEQLENS_Q + BLOCK_M - 1) // BLOCK_M - 1 - tl.program_id(0)\n"
    _kernel_iter = "    if IS_CAUSAL:\n        iter = 2\n    else:\n        iter = 1\n"
    _kernel_iter_one = "    if IS_CAUSAL:\n        iter = 1\n    else:\n        iter = 1\n"
    _wrapper_iter = "        if metadata.causal:\n            iter = 2\n        else:\n            iter = 1\n"
    _wrapper_iter_one = "        if metadata.causal:\n            iter = 1\n        else:\n            iter = 1\n"
    for old, n, label in (
        (_old_ret, 1, "early-return"),
        (_old_rev, 1, "reverse-index"),
        (_kernel_iter, 1, "kernel-iter"),
        (_wrapper_iter, 1, "wrapper-iter"),
    ):
        if _raw.count(old) != n:
            raise RuntimeError(f"shortQ onepass vendor {label} anchor changed")
    _fixed = (_raw.replace(_old_ret, _new_ret)
                  .replace(_old_rev, _new_rev)
                  .replace(_kernel_iter, _kernel_iter_one)
                  .replace(_wrapper_iter, _wrapper_iter_one))
    _tmp = Path(f"/tmp/q38_tp4_shortq_onepass_{os.getpid()}.py")
    _tmp.write_text(_fixed)
    _spec = importlib.util.spec_from_file_location(f"q38_tp4_shortq_onepass_{os.getpid()}", _tmp)
    _tri = importlib.util.module_from_spec(_spec)
    assert _spec.loader is not None
    _spec.loader.exec_module(_tri)

    # Isolated 2026-08-24 sweep winner on Q6/KV1/D256 short-q shapes.
    _winner = []
    for _c in getattr(_tri.attn_fwd, "configs", []):
        _kw = dict(getattr(_c, "kwargs", {}) or {})
        if (
            _kw.get("BLOCK_M") == 64
            and _kw.get("BLOCK_N") == 64
            and _kw.get("waves_per_eu") == 0
            and _kw.get("PRE_LOAD_V") is True
            and int(getattr(_c, "num_stages", -1)) == 1
            and int(getattr(_c, "num_warps", -1)) == 4
        ):
            _winner.append(_c)
    if len(_winner) != 1:
        raise RuntimeError(f"shortQ onepass expected one BM64/BN64/w4/preloadV winner, got {len(_winner)}")
    _tri.attn_fwd.configs = _winner
    if hasattr(_tri.attn_fwd, "cache"):
        _tri.attn_fwd.cache.clear()
    if hasattr(_tri.attn_fwd, "best_config"):
        _tri.attn_fwd.best_config = None
    _tri_fn = _tri.flash_attn_varlen_func

    _parent_try = _packmod.try_pack_paged_kv_to_varlen_attention
    _hits = 0

    def _shortq_try_pack_paged_kv_to_varlen_attention(
        *, q: torch.Tensor, forward_batch, metadata, layer, window_size: tuple,
        sinks, key_cache: torch.Tensor, value_cache: torch.Tensor, page_size: int,
        k_descale, v_descale, **kwargs,
    ):
        global _hits
        qlen = int(q.shape[0])
        layer_id = int(getattr(layer, "layer_id", -1))
        if layer_id in _FALLBACK_LAYERS:
            return _parent_try(
                q=q, forward_batch=forward_batch, metadata=metadata, layer=layer,
                window_size=window_size, sinks=sinks, key_cache=key_cache,
                value_cache=value_cache, page_size=page_size,
                k_descale=k_descale, v_descale=v_descale, **kwargs,
            )
        if not (_MIN_Q <= qlen <= _MAX_Q):
            return _parent_try(
                q=q, forward_batch=forward_batch, metadata=metadata, layer=layer,
                window_size=window_size, sinks=sinks, key_cache=key_cache,
                value_cache=value_cache, page_size=page_size,
                k_descale=k_descale, v_descale=v_descale, **kwargs,
            )
        if int(getattr(forward_batch, "batch_size", -1)) != 1:
            return _parent_try(
                q=q, forward_batch=forward_batch, metadata=metadata, layer=layer,
                window_size=window_size, sinks=sinks, key_cache=key_cache,
                value_cache=value_cache, page_size=page_size,
                k_descale=k_descale, v_descale=v_descale, **kwargs,
            )
        seq = getattr(forward_batch, "seq_lens_cpu", None)
        if seq is None or len(seq) < 1:
            return _parent_try(
                q=q, forward_batch=forward_batch, metadata=metadata, layer=layer,
                window_size=window_size, sinks=sinks, key_cache=key_cache,
                value_cache=value_cache, page_size=page_size,
                k_descale=k_descale, v_descale=v_descale, **kwargs,
            )
        kvlen = int(seq[0])
        eligible = (
            kvlen == qlen  # cold/full prompt only; prefix-resume stays parent
            and int(getattr(metadata, "max_seq_len_q", -1)) == qlen
            and int(getattr(metadata, "max_seq_len_k", -1)) == kvlen
            and q.numel() == qlen * 6 * 256
            and q.is_contiguous()
            and q.dtype == torch.bfloat16
            and key_cache.dtype == torch.bfloat16
            and value_cache.dtype == torch.bfloat16
            and int(getattr(layer, "tp_q_head_num", -1)) == 6
            and int(getattr(layer, "tp_k_head_num", -1)) == 1
            and int(getattr(layer, "tp_v_head_num", -1)) == 1
            and int(getattr(layer, "head_dim", -1)) == 256
            and int(getattr(layer, "v_head_dim", -1)) == 256
            and int(page_size) == 64
            and tuple(window_size) == (-1, -1)
            and sinks is None
            and abs(float(getattr(layer, "scaling", -1.0)) - 0.0625) <= 1e-12
        )
        if not eligible or not _packmod.can_pack_paged_kv_to_varlen(
            forward_batch=forward_batch, metadata=metadata, layer=layer,
            window_size=window_size, sinks=sinks, key_cache=key_cache,
            value_cache=value_cache, page_size=page_size,
        ):
            return _parent_try(
                q=q, forward_batch=forward_batch, metadata=metadata, layer=layer,
                window_size=window_size, sinks=sinks, key_cache=key_cache,
                value_cache=value_cache, page_size=page_size,
                k_descale=k_descale, v_descale=v_descale, **kwargs,
            )

        packed_k, packed_v = _packmod.pack_paged_kv_to_varlen(
            key_cache.view(-1, layer.tp_k_head_num, page_size, layer.head_dim),
            value_cache.view(-1, layer.tp_v_head_num, layer.v_head_dim, page_size),
            metadata.page_table,
            forward_batch.seq_lens_cpu[:1],
            page_size,
        )
        if int(packed_k.shape[0]) != kvlen or int(packed_v.shape[0]) != kvlen:
            raise RuntimeError(f"shortQ onepass gather mismatch q={qlen} kv={kvlen}")
        _hits += 1
        if _hits <= 8 or _hits % 16 == 0:
            print(
                f"[K100 DFlash2 TP4 shortQ onepass] ACTIVE hit={_hits} q=kv={qlen} "
                "Q6/KV1/D256 BM64/BN64/w4/preloadV",
                flush=True,
            )
        out = _tri_fn(
            q.view(-1, layer.tp_q_head_num, layer.head_dim),
            packed_k, packed_v,
            metadata.cu_seqlens_q, metadata.cu_seqlens_k,
            qlen, kvlen,
            dropout_p=0.0, softmax_scale=layer.scaling, causal=True,
            return_attn_probs=False,
        )
        return out[0] if isinstance(out, (tuple, list)) else out

    _packmod.try_pack_paged_kv_to_varlen_attention = _shortq_try_pack_paged_kv_to_varlen_attention
    _fabackend.try_pack_paged_kv_to_varlen_attention = _shortq_try_pack_paged_kv_to_varlen_attention
    print(
        f"[K100 DFlash2 TP4 shortQ onepass] installed range={_MIN_Q}..{_MAX_Q}; "
        f"parent_layers={sorted(_FALLBACK_LAYERS)}; cold q=kv only; exact gather; all non-exact shapes parent fallback",
        flush=True,
    )
