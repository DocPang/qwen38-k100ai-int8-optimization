"""TP1 short/mid selective contiguous-pack diagnostic overlay.

Purpose
-------
Isolate the 4K..8K TP1 prefill opportunity without changing the production
PACK_MIN_KV=8192 policy globally.

Parent is the frozen TP1 Final v2.  For exact batch1 cold/full-prompt TP1
full-attention geometry (q == kv, q in [4096, 8191], QH24/KVH4/D256,
BF16/page64), only layer ids explicitly listed in FAST_LAYERS are routed to
paged-KV -> contiguous varlen FlashAttention.  Every other layer and every
non-exact shape falls back to the frozen parent.

This is diagnostic-only until full quality/performance promotion.  It never
re-enables the known-bad gfx928 q>=5 paged native consumer.
"""
from __future__ import annotations

import os
from pathlib import Path
import runpy

_ROOT = "/data/qwen38-27b-k100ai-int8-opt"
_PARENT = f"{_ROOT}/runtime_patch_dflash_tp1_final_v2/sitecustomize.py"

try:
    _cmd = open("/proc/self/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "replace")
except Exception:
    _cmd = ""
_helper = (
    ("cdll.LoadLibrary" in _cmd and "torchinductor_tp1" in _cmd)
    or "multiprocessing.resource_tracker" in _cmd
    or "/usr/local/bin/ninja" in _cmd
    or " ninja --version" in _cmd
)

_skip_parent = os.getenv("SGLANG_Q38_TP1_SHORTMID_SKIP_PARENT") == "1"
if not _skip_parent:
    runpy.run_path(_PARENT, run_name="__q38_tp1_shortmid_selective_pack_parent__")
else:
    print(
        "[K100 DFlash2 TP1 selective-pack diag] parent skipped by explicit composition",
        flush=True,
    )

if _helper:
    print(f"[K100 DFlash2 TP1 selective-pack helper bypass] pid={os.getpid()}", flush=True)
else:
    import torch
    from sglang.srt.layers.attention import pack_paged_kv_to_varlen as _packmod
    from sglang.srt.layers.attention import flashattention_backend as _fabackend
    from sglang.srt.layers.attention import flashattention_interface as _fai

    _MIN_Q = int(os.getenv("SGLANG_Q38_TP1_SHORTMID_SELECTIVE_PACK_MIN_Q", "4096"))
    _MAX_Q = int(os.getenv("SGLANG_Q38_TP1_SHORTMID_SELECTIVE_PACK_MAX_Q", "8191"))
    _STATIC_FAST_LAYERS = {
        int(x)
        for x in os.getenv("SGLANG_Q38_TP1_SHORTMID_FAST_LAYERS", "").split(",")
        if x.strip()
    }
    _FAST_FILE = Path(
        os.getenv(
            "SGLANG_Q38_TP1_SHORTMID_FAST_LAYERS_FILE",
            f"{_ROOT}/results/tp1_shortmid_fast_layers_manual_20260825.txt",
        )
    )
    _fast_file_mtime_ns = None
    _fast_file_layers: set[int] = set()

    def _get_fast_layers() -> set[int]:
        """Diagnostic-only hot reload of layers allowed to use contiguous pack."""
        global _fast_file_mtime_ns, _fast_file_layers
        try:
            st = _FAST_FILE.stat()
            mtime_ns = int(st.st_mtime_ns)
        except FileNotFoundError:
            mtime_ns = -1
        if mtime_ns != _fast_file_mtime_ns:
            if mtime_ns < 0:
                parsed: set[int] = set()
            else:
                text = _FAST_FILE.read_text().strip()
                parsed = {int(x) for x in text.split(",") if x.strip()}
            invalid = parsed.difference(range(64))
            if invalid:
                raise RuntimeError(f"TP1 selective-pack invalid fast layers: {sorted(invalid)}")
            _fast_file_layers = parsed
            _fast_file_mtime_ns = mtime_ns
            print(
                f"[K100 DFlash2 TP1 selective-pack diag] fast_layers="
                f"{sorted(_STATIC_FAST_LAYERS | _fast_file_layers)}",
                flush=True,
            )
        return _STATIC_FAST_LAYERS | _fast_file_layers

    if not (1 <= _MIN_Q <= _MAX_Q < 8192):
        raise RuntimeError(f"TP1 selective-pack invalid range {_MIN_Q}..{_MAX_Q}")

    _parent_try = _packmod.try_pack_paged_kv_to_varlen_attention
    _hits = 0

    def _tp1_shortmid_selective_pack(
        *, q: torch.Tensor, forward_batch, metadata, layer, window_size: tuple,
        sinks, key_cache: torch.Tensor, value_cache: torch.Tensor, page_size: int,
        k_descale, v_descale, **kwargs,
    ):
        global _hits
        qlen = int(q.shape[0])
        layer_id = int(getattr(layer, "layer_id", -1))
        fast_layers = _get_fast_layers()

        if layer_id not in fast_layers or not (_MIN_Q <= qlen <= _MAX_Q):
            return _parent_try(
                q=q, forward_batch=forward_batch, metadata=metadata, layer=layer,
                window_size=window_size, sinks=sinks, key_cache=key_cache,
                value_cache=value_cache, page_size=page_size,
                k_descale=k_descale, v_descale=v_descale, **kwargs,
            )

        seq = getattr(forward_batch, "seq_lens_cpu", None)
        batch_size = int(getattr(forward_batch, "batch_size", -1))
        if seq is None or batch_size != 1 or len(seq) < 1:
            return _parent_try(
                q=q, forward_batch=forward_batch, metadata=metadata, layer=layer,
                window_size=window_size, sinks=sinks, key_cache=key_cache,
                value_cache=value_cache, page_size=page_size,
                k_descale=k_descale, v_descale=v_descale, **kwargs,
            )

        kvlen = int(seq[0])
        page_table = getattr(metadata, "page_table", None)
        cu_q = getattr(metadata, "cu_seqlens_q", None)
        cu_k = getattr(metadata, "cu_seqlens_k", None)
        max_q = int(getattr(metadata, "max_seq_len_q", -1))
        max_k = int(getattr(metadata, "max_seq_len_k", -1))
        page_cols_ok = (
            page_table is not None
            and page_table.ndim >= 2
            and int(page_table.shape[0]) >= 1
            and int(page_table.shape[1]) >= (max_k + int(page_size) - 1) // int(page_size)
        )
        exact_ok = (
            kvlen == qlen
            and max_q == qlen
            and max_k == kvlen
            and bool(getattr(_packmod, "_kv_layout_dcu_fa", False))
            and not bool(getattr(forward_batch, "mha_return_lse", False))
            and float(getattr(layer, "logit_cap", 0.0)) == 0.0
            and page_cols_ok
            and cu_q is not None and int(cu_q.shape[0]) >= 2
            and cu_k is not None and int(cu_k.shape[0]) >= 2
            and q.numel() == qlen * 24 * 256
            and q.is_contiguous()
            and q.dtype == torch.bfloat16
            and key_cache.dtype == torch.bfloat16
            and value_cache.dtype == torch.bfloat16
            and int(getattr(layer, "tp_q_head_num", -1)) == 24
            and int(getattr(layer, "tp_k_head_num", -1)) == 4
            and int(getattr(layer, "tp_v_head_num", -1)) == 4
            and int(getattr(layer, "head_dim", -1)) == 256
            and int(getattr(layer, "v_head_dim", -1)) == 256
            and int(page_size) == 64
            and tuple(window_size) == (-1, -1)
            and sinks is None
            and abs(float(getattr(layer, "scaling", -1.0)) - 0.0625) <= 1e-12
        )
        if not exact_ok:
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
            raise RuntimeError(f"TP1 selective-pack gather mismatch q={qlen} kv={kvlen}")

        _hits += 1
        if _hits <= 8 or _hits % 16 == 0:
            print(
                f"[K100 DFlash2 TP1 selective-pack] ACTIVE hit={_hits} layer={layer_id} "
                f"q=kv={qlen} QH24/KVH4/D256 contiguous-varlen",
                flush=True,
            )

        return _fai.flash_attn_varlen_func(
            q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
            k=packed_k,
            v=packed_v,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k=metadata.cu_seqlens_k,
            max_seqlen_q=qlen,
            max_seqlen_k=kvlen,
            softmax_scale=layer.scaling,
            causal=True,
            k_descale=k_descale,
            v_descale=v_descale,
            return_softmax_lse=False,
            **kwargs,
        )

    _packmod.try_pack_paged_kv_to_varlen_attention = _tp1_shortmid_selective_pack
    _fabackend.try_pack_paged_kv_to_varlen_attention = _tp1_shortmid_selective_pack
    print(
        f"[K100 DFlash2 TP1 selective-pack diag] installed range={_MIN_Q}..{_MAX_Q}; "
        f"static_fast_layers={sorted(_STATIC_FAST_LAYERS)}; fast_file={_FAST_FILE}; "
        "production PACK_MIN_KV must remain 8192; exact cold q=kv only; all other shapes parent fallback",
        flush=True,
    )
