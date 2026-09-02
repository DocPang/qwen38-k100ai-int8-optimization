"""TP2 DFlash2 RC1 + direct packed long-KV BM128/w8 attention v4 + exact 257900 qtail.

The validated RC1 q8split verifier is installed first.  This overlay intercepts
only SGLang's pack-paged-kv-to-varlen fast path for batch1 q=8192 prefill when
32K <= KV <= 248K on the scheduler 8K grid on TP2 rank-local QH12/KVH2/D256 BF16/page64 geometry.
The packed contiguous attention is evaluated by the SourceFind Triton GQA
kernel frozen at BM128/BN64/w8/no-preload-V, the isolated 64K/128K/~258K winner.
Everything outside that exact contract falls back to the RC1 parent.
"""
from __future__ import annotations

import importlib.util
import os
import runpy
from pathlib import Path

import torch
import triton

_ROOT = "/data/qwen38-27b-k100ai-int8-opt"
_PARENT = f"{_ROOT}/runtime_patch_sglang_tp2_dflash2_q8split_layerselect_v1/sitecustomize.py"
_SRC = f"{_ROOT}/external_images/sglang0.5.12-20260620-src/flash_attn/flash_attn_triton_mqa_gqa.py"

runpy.run_path(_PARENT, run_name="__q38_tp2_dflash2_longkv_v4_parent__")

from sglang.srt.layers.attention import pack_paged_kv_to_varlen as _packmod
from sglang.srt.layers.attention import flashattention_backend as _fabackend

_parent_try = _packmod.try_pack_paged_kv_to_varlen_attention
os.environ["FLASH_ATTENTION_PRINT_PARAM"] = "0"

raw = Path(_SRC).read_text()
old_start = "            start_m = MAX_SEQLENS_Q // BLOCK_M -1 - tl.program_id(0)\n"
new_start = "            start_m = (MAX_SEQLENS_Q + BLOCK_M - 1) // BLOCK_M - 1 - tl.program_id(0)\n"
old_guard = "            if start_m * BLOCK_M > seqlen_q:\n                return\n"
new_guard = "            if start_m * BLOCK_M > seqlen_q:\n                start_m = 0  # admitted q8K/qtail grids; avoids Triton loop-return rejection\n"
if raw.count(old_start) != 1 or raw.count(old_guard) != 1:
    raise RuntimeError("TP2 longkv v4 vendor Triton causal source changed; fail closed")
raw = raw.replace(old_start, new_start).replace(old_guard, new_guard)
tmp_src = f"/tmp/q38_tp2_dflash2_longkv_v4_{os.getpid()}.py"
Path(tmp_src).write_text(raw)
spec = importlib.util.spec_from_file_location(f"q38_tp2_dflash2_longkv_v4_{os.getpid()}", tmp_src)
tri = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tri)
tri.attn_fwd.configs = [
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "waves_per_eu": 0, "PRE_LOAD_V": False},
        num_stages=1,
        num_warps=8,
    )
]
if hasattr(tri.attn_fwd, "cache"):
    tri.attn_fwd.cache.clear()
if hasattr(tri.attn_fwd, "best_config"):
    tri.attn_fwd.best_config = None
_tri_fn = tri.flash_attn_varlen_func
_hits = 0
_qtail_hits = 0
_qtail_probes = 0


def _first(x):
    return x[0] if isinstance(x, (tuple, list)) else x


def _eligible(*, q, forward_batch, metadata, layer, window_size, sinks, key_cache, value_cache, page_size):
    if int(getattr(forward_batch, "batch_size", -1)) != 1:
        return False, None
    sl = getattr(forward_batch, "seq_lens_cpu", None)
    if sl is None or len(sl) < 1:
        return False, None
    kv = int(sl[0])
    if int(q.shape[0]) != 8192 or not (32768 <= kv <= 253952) or kv % 8192 != 0:
        return False, None
    if int(getattr(metadata, "max_seq_len_q", -1)) != 8192:
        return False, None
    if int(getattr(metadata, "max_seq_len_k", -1)) != kv:
        return False, None
    if int(page_size) != 64:
        return False, None
    if int(getattr(layer, "tp_q_head_num", -1)) != 12:
        return False, None
    if int(getattr(layer, "tp_k_head_num", -1)) != 2:
        return False, None
    if int(getattr(layer, "tp_v_head_num", -1)) != 2:
        return False, None
    if int(getattr(layer, "head_dim", -1)) != 256 or int(getattr(layer, "v_head_dim", -1)) != 256:
        return False, None
    if q.dtype != torch.bfloat16 or key_cache.dtype != torch.bfloat16 or value_cache.dtype != torch.bfloat16:
        return False, None
    if tuple(window_size) != (-1, -1) or sinks is not None:
        return False, None
    if abs(float(getattr(layer, "scaling", -1.0)) - 0.0625) > 1e-12:
        return False, None
    if bool(getattr(forward_batch, "mha_return_lse", False)):
        return False, None
    if not q.is_contiguous() or q.numel() != 8192 * 12 * 256:
        return False, None
    if not _packmod.can_pack_paged_kv_to_varlen(
        forward_batch=forward_batch,
        metadata=metadata,
        layer=layer,
        window_size=window_size,
        sinks=sinks,
        key_cache=key_cache,
        value_cache=value_cache,
        page_size=page_size,
    ):
        return False, None
    return True, kv


def _try_qtail(*, q, forward_batch, metadata, layer, window_size, sinks, key_cache, value_cache, page_size):
    global _qtail_hits, _qtail_probes
    qn = int(q.shape[0])
    if qn not in (3948, 3968):
        return None, False
    sl = getattr(forward_batch, "seq_lens_cpu", None)
    if sl is None or len(sl) < 1 or int(getattr(forward_batch, "batch_size", -1)) != 1:
        return None, False
    kv = int(sl[0])
    _qtail_probes += 1
    if _qtail_probes <= 4:
        print(f"[K100 DFlash2 TP2 qtail probe] q={qn} kv={kv} maxq={int(getattr(metadata,'max_seq_len_q',-1))} maxk={int(getattr(metadata,'max_seq_len_k',-1))}", flush=True)
    exact = (
        qn == 3948 and kv == 257900
        and int(getattr(metadata, "max_seq_len_q", -1)) == 3948
        and int(getattr(metadata, "max_seq_len_k", -1)) == 257900
        and int(page_size) == 64
        and int(getattr(layer, "tp_q_head_num", -1)) == 12
        and int(getattr(layer, "tp_k_head_num", -1)) == 2
        and int(getattr(layer, "tp_v_head_num", -1)) == 2
        and int(getattr(layer, "head_dim", -1)) == 256
        and int(getattr(layer, "v_head_dim", -1)) == 256
        and q.dtype == torch.bfloat16 and key_cache.dtype == torch.bfloat16 and value_cache.dtype == torch.bfloat16
        and tuple(window_size) == (-1, -1) and sinks is None
        and abs(float(getattr(layer, "scaling", -1.0)) - 0.0625) <= 1e-12
        and not bool(getattr(forward_batch, "mha_return_lse", False))
        and q.is_contiguous() and q.numel() == 3948 * 12 * 256
        and getattr(metadata, "page_table", None) is not None
        and getattr(metadata, "cu_seqlens_q", None) is not None
        and getattr(metadata, "cu_seqlens_k", None) is not None
    )
    if not exact:
        return None, False
    pk, pv = _packmod.pack_paged_kv_to_varlen(
        key_cache.view(-1, layer.tp_k_head_num, page_size, layer.head_dim),
        value_cache.view(-1, layer.tp_v_head_num, layer.v_head_dim, page_size),
        metadata.page_table, forward_batch.seq_lens_cpu[: forward_batch.batch_size], page_size,
    )
    if int(pk.shape[0]) != 257900 or int(pv.shape[0]) != 257900:
        raise RuntimeError(f"TP2 qtail v4 gather mismatch K={tuple(pk.shape)} V={tuple(pv.shape)}")
    _qtail_hits += 1
    if _qtail_hits <= 4 or _qtail_hits % 16 == 0:
        print(f"[K100 DFlash2 TP2 qtail BM128 v4] ACTIVE hit={_qtail_hits} layer={int(getattr(layer,'layer_id',-1))} q=3948 kv=257900 ceil-causal", flush=True)
    out = _tri_fn(
        q.view(-1, layer.tp_q_head_num, layer.head_dim), pk, pv,
        metadata.cu_seqlens_q, metadata.cu_seqlens_k, 3948, 257900,
        dropout_p=0.0, softmax_scale=layer.scaling, causal=True, return_attn_probs=False,
    )
    return _first(out), True


def _longkv_try(*, q, forward_batch, metadata, layer, window_size, sinks,
                key_cache, value_cache, page_size, k_descale, v_descale, **kwargs):
    global _hits
    qtail_out, qtail_hit = _try_qtail(
        q=q, forward_batch=forward_batch, metadata=metadata, layer=layer, window_size=window_size,
        sinks=sinks, key_cache=key_cache, value_cache=value_cache, page_size=page_size,
    )
    if qtail_hit:
        return qtail_out
    ok, kv = _eligible(
        q=q,
        forward_batch=forward_batch,
        metadata=metadata,
        layer=layer,
        window_size=window_size,
        sinks=sinks,
        key_cache=key_cache,
        value_cache=value_cache,
        page_size=page_size,
    )
    if not ok:
        return _parent_try(
            q=q,
            forward_batch=forward_batch,
            metadata=metadata,
            layer=layer,
            window_size=window_size,
            sinks=sinks,
            key_cache=key_cache,
            value_cache=value_cache,
            page_size=page_size,
            k_descale=k_descale,
            v_descale=v_descale,
            **kwargs,
        )

    pk, pv = _packmod.pack_paged_kv_to_varlen(
        key_cache.view(-1, layer.tp_k_head_num, page_size, layer.head_dim),
        value_cache.view(-1, layer.tp_v_head_num, layer.v_head_dim, page_size),
        metadata.page_table,
        forward_batch.seq_lens_cpu[: forward_batch.batch_size],
        page_size,
    )
    if int(pk.shape[0]) != kv or int(pv.shape[0]) != kv:
        raise RuntimeError(f"TP2 longkv v4 gather mismatch kv={kv} K={tuple(pk.shape)} V={tuple(pv.shape)}")
    if tuple(pk.shape[1:]) != (2, 256) or tuple(pv.shape[1:]) != (2, 256):
        raise RuntimeError(f"TP2 longkv v4 packed geometry drift K={tuple(pk.shape)} V={tuple(pv.shape)}")

    _hits += 1
    if _hits <= 8 or _hits % 16 == 0:
        print(
            "[K100 DFlash2 TP2 longkv BM128 v4] ACTIVE "
            f"hit={_hits} layer={int(getattr(layer, 'layer_id', -1))} q=8192 kv={kv} "
            "qh=12 kvh=2 d=256 BM128/BN64/w8",
            flush=True,
        )

    out = _tri_fn(
        q.view(-1, layer.tp_q_head_num, layer.head_dim),
        pk,
        pv,
        metadata.cu_seqlens_q,
        metadata.cu_seqlens_k,
        8192,
        kv,
        dropout_p=0.0,
        softmax_scale=layer.scaling,
        causal=True,
        return_attn_probs=False,
    )
    return _first(out)


# flashattention_backend imported the function by name at module import time;
# patch both references so the full-model call site cannot retain a stale alias.
_packmod.try_pack_paged_kv_to_varlen_attention = _longkv_try
_fabackend.try_pack_paged_kv_to_varlen_attention = _longkv_try

print(
    "[K100 DFlash2 TP2 longkv BM128 v4] installed after q8split RC1; "
    "direct pack hook q8K 32K<=KV<=248K plus exact q3948/KV257900 ceil-causal tail QH12/KVH2/D256 BF16; "
    "all other shapes parent fallback",
    flush=True,
)
