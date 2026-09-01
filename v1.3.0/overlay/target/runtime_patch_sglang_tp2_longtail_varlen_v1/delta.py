"""TP2 long-context partial-tail varlen v1 delta.

Admission contract (fail-closed):
  - batch_size == 1
  - 512 <= q <= 8191 (partial-tail chunk; q==8192 stays with the longkv v4 hook)
  - 20000 < kv <= 262144 (above the layerdiag v5 short/mid coverage)
  - excludes the frozen exact qtail shapes q in {3948,3968} with kv >= 250000
  - only TP2 rank-local full-attention QH12/KVH2/D256/BF16/page64 causal geometry
  - all remaining guards identical to the accepted longkv BM128 v4 hook

Everything else falls back to the champion parent chain unchanged.
"""
from __future__ import annotations

import importlib.util
import os
import triton
from pathlib import Path

import torch

_ROOT = "/data/qwen38-27b-k100ai-int8-opt"
_VENDOR = f"{_ROOT}/external_images/sglang0.5.12-20260620-src/flash_attn/flash_attn_triton_mqa_gqa.py"

raw = Path(_VENDOR).read_text()
old_start = "            start_m = MAX_SEQLENS_Q // BLOCK_M -1 - tl.program_id(0)\n"
new_start = "            start_m = (MAX_SEQLENS_Q + BLOCK_M - 1) // BLOCK_M - 1 - tl.program_id(0)\n"
old_guard = "            if start_m * BLOCK_M > seqlen_q:\n                return\n"
new_guard = "            if start_m * BLOCK_M > seqlen_q:\n                start_m = 0  # admitted longtail grids; avoids Triton loop-return rejection\n"
if raw.count(old_start) != 1 or raw.count(old_guard) != 1:
    raise RuntimeError("TP2 longtail varlen v1: vendor causal source changed; fail closed")
raw = raw.replace(old_start, new_start).replace(old_guard, new_guard)
tmp_src = f"/tmp/q38_tp2_longtail_v1_{os.getpid()}.py"
Path(tmp_src).write_text(raw)
spec = importlib.util.spec_from_file_location(f"q38_tp2_longtail_v1_{os.getpid()}", tmp_src)
tri = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tri)
# Same hand-constructed winner config as the accepted longkv BM128 v4 hook
# (isolated 64K/128K/~258K winner for long-KV packed attention).
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

from sglang.srt.layers.attention import pack_paged_kv_to_varlen as _packmod
from sglang.srt.layers.attention import flashattention_backend as _fabackend

_parent_try = _packmod.try_pack_paged_kv_to_varlen_attention
os.environ.setdefault("FLASH_ATTENTION_PRINT_PARAM", "0")
_hits = 0


def _first(x):
    return x[0] if isinstance(x, (tuple, list)) else x


def _tail_eligible(*, q, forward_batch, metadata, layer, window_size, sinks,
                   key_cache, value_cache, page_size):
    if int(getattr(forward_batch, "batch_size", -1)) != 1:
        return False, None
    sl = getattr(forward_batch, "seq_lens_cpu", None)
    if sl is None or len(sl) < 1:
        return False, None
    kv = int(sl[0])
    qn = int(q.shape[0])
    if not (512 <= qn <= 8191):
        return False, None
    if not (20000 < kv <= 262144):
        return False, None
    if qn in (3948, 3968) and kv >= 250000:
        # frozen exact qtail shapes stay with the accepted longkv v4 hook
        return False, None
    if int(getattr(metadata, "max_seq_len_q", -1)) != qn:
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
    if float(getattr(layer, "logit_cap", 0.0)) != 0.0:
        return False, None
    if bool(getattr(forward_batch, "mha_return_lse", False)):
        return False, None
    if not q.is_contiguous() or q.numel() != qn * 12 * 256:
        return False, None
    if getattr(metadata, "page_table", None) is None:
        return False, None
    if getattr(metadata, "cu_seqlens_q", None) is None:
        return False, None
    if getattr(metadata, "cu_seqlens_k", None) is None:
        return False, None
    # Same inline page-table guards as the accepted layerdiag v5 delta; the
    # upstream can_pack gate is deliberately NOT consulted because its
    # min-q-tokens threshold (8192) is exactly the partial-tail rejection this
    # patch exists to bypass. total_kv (kv>20000) always exceeds the min-kv
    # threshold, and batch1 satisfies the kv-head batch cap.
    if int(getattr(metadata, "page_table", torch.empty(0, 0)).shape[0]) < 1:
        return False, None
    if int(getattr(metadata, "page_table", torch.empty(0, 0)).shape[1]) < (kv + 63) // 64:
        return False, None
    if int(getattr(metadata, "cu_seqlens_k", torch.empty(0)).shape[0]) < 2:
        return False, None
    return True, kv


def _longtail_try(*, q, forward_batch, metadata, layer, window_size, sinks,
                  key_cache, value_cache, page_size, k_descale, v_descale, **kwargs):
    global _hits
    ok, kv = _tail_eligible(
        q=q, forward_batch=forward_batch, metadata=metadata, layer=layer,
        window_size=window_size, sinks=sinks, key_cache=key_cache,
        value_cache=value_cache, page_size=page_size,
    )
    if not ok:
        return _parent_try(
            q=q, forward_batch=forward_batch, metadata=metadata, layer=layer,
            window_size=window_size, sinks=sinks, key_cache=key_cache,
            value_cache=value_cache, page_size=page_size,
            k_descale=k_descale, v_descale=v_descale, **kwargs,
        )

    qn = int(q.shape[0])
    layer_id = int(getattr(layer, "layer_id", -1))
    if layer_id not in frozenset(range(3, 64, 4)):
        raise RuntimeError(f"TP2 longtail varlen v1 unexpected admitted layer_id={layer_id}")
    pk, pv = _packmod.pack_paged_kv_to_varlen(
        key_cache.view(-1, layer.tp_k_head_num, page_size, layer.head_dim),
        value_cache.view(-1, layer.tp_v_head_num, layer.v_head_dim, page_size),
        metadata.page_table,
        forward_batch.seq_lens_cpu[: forward_batch.batch_size],
        page_size,
    )
    if int(pk.shape[0]) != kv or int(pv.shape[0]) != kv:
        raise RuntimeError(f"TP2 longtail v1 gather mismatch kv={kv} K={tuple(pk.shape)} V={tuple(pv.shape)}")
    if tuple(pk.shape[1:]) != (2, 256) or tuple(pv.shape[1:]) != (2, 256):
        raise RuntimeError(f"TP2 longtail v1 packed geometry drift K={tuple(pk.shape)} V={tuple(pv.shape)}")

    _hits += 1
    if _hits <= 8 or _hits % 16 == 0:
        print(
            "[K100 DFlash2 TP2 longtail varlen v1] ACTIVE "
            f"hit={_hits} layer={int(getattr(layer, 'layer_id', -1))} q={qn} kv={kv} "
            "qh=12 kvh=2 d=256 BM128/BN64/w8",
            flush=True,
        )

    out = _tri_fn(
        q.view(-1, layer.tp_q_head_num, layer.head_dim),
        pk,
        pv,
        metadata.cu_seqlens_q,
        metadata.cu_seqlens_k,
        qn,
        kv,
        dropout_p=0.0,
        softmax_scale=layer.scaling,
        causal=True,
        return_attn_probs=False,
    )
    return _first(out)


# Patch both references so the full-model call site cannot retain a stale alias.
_packmod.try_pack_paged_kv_to_varlen_attention = _longtail_try
_fabackend.try_pack_paged_kv_to_varlen_attention = _longtail_try

print(
    "[K100 DFlash2 TP2 longtail varlen v1] installed after champion {3,23}; "
    "partial tails 512<=q<=8191 with kv>20000 admitted on corrected varlen BM128/BN64/w8; "
    "frozen qtail q3948/3968 at kv>=250000 stays with longkv v4; all other shapes parent fallback",
    flush=True,
)
