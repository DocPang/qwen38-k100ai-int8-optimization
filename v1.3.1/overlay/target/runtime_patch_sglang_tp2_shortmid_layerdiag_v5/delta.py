"""TP2 DFlash2 short/mid varlen v2.

Root cause under the accepted TP2 launch contract:
  --chunked-prefill-size 8192
  --pack-paged-kv-to-varlen auto
  --pack-paged-kv-to-varlen-min-q-tokens 8192

The upstream auto-pack gate therefore rejects a partial second chunk whenever
q<8192 even though all correctness preconditions are satisfied.  This creates
the observed 8K..16K TTFT sawtooth (e.g. 12K q=4096/KV=12288 is slower than
16K q=8192/KV=16384).

This fail-closed delta uses the same corrected contiguous-varlen consumer for:
  A) first cold-prefill chunk: q==kv, 2048<=q<=8191 (v1),
  B) exact second partial chunk: kv-q==8192, 1<=q<=8191.

Only batch1 TP2 full-attention QH12/KVH2/D256/BF16/page64 causal geometry is
admitted. Everything else falls back to the accepted parent unchanged.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch

_ROOT = "/data/qwen38-27b-k100ai-int8-opt"
_VENDOR = f"{_ROOT}/external_images/sglang0.5.12-20260620-src/flash_attn/flash_attn_triton_mqa_gqa.py"

# Load the SourceFind contiguous-varlen Triton consumer with the already-proven
# ceil-causal repair used by TP2 long-KV/qtail. Freeze the actual vendor
# BM128/BN64/w4/no-preload-V config present in this image.
_raw = Path(_VENDOR).read_text()
_old_start = "            start_m = MAX_SEQLENS_Q // BLOCK_M -1 - tl.program_id(0)\n"
_new_start = "            start_m = (MAX_SEQLENS_Q + BLOCK_M - 1) // BLOCK_M - 1 - tl.program_id(0)\n"
_old_guard = "            if start_m * BLOCK_M > seqlen_q:\n                return\n"
_new_guard = "            if start_m * BLOCK_M > seqlen_q:\n                start_m = 0\n"
if _raw.count(_old_start) != 1 or _raw.count(_old_guard) != 1:
    raise RuntimeError("TP2 shortmid layerdiag v5: vendor causal source changed; fail closed")
_raw = _raw.replace(_old_start, _new_start).replace(_old_guard, _new_guard)
_tmp = f"/tmp/q38_tp2_shortmid_varlen_v2_{os.getpid()}.py"
Path(_tmp).write_text(_raw)
_spec = importlib.util.spec_from_file_location(f"q38_tp2_shortmid_varlen_v2_{os.getpid()}", _tmp)
_tri = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_tri)

_winner = []
for _c in getattr(_tri.attn_fwd, "configs", []):
    _kw = dict(getattr(_c, "kwargs", {}) or {})
    if (
        _kw.get("BLOCK_M") == 128
        and _kw.get("BLOCK_N") == 64
        and _kw.get("waves_per_eu") == 0
        and _kw.get("PRE_LOAD_V") is False
        and int(getattr(_c, "num_stages", -1)) == 1
        and int(getattr(_c, "num_warps", -1)) == 4
    ):
        _winner.append(_c)
if len(_winner) != 1:
    raise RuntimeError(f"TP2 shortmid layerdiag v5: expected one BM128/BN64/w4 config, got {len(_winner)}")
_tri.attn_fwd.configs = _winner
if hasattr(_tri.attn_fwd, "cache"):
    _tri.attn_fwd.cache.clear()
if hasattr(_tri.attn_fwd, "best_config"):
    _tri.attn_fwd.best_config = None
_tri_fn = _tri.flash_attn_varlen_func

from sglang.srt.layers.attention import pack_paged_kv_to_varlen as _packmod
from sglang.srt.layers.attention import flashattention_backend as _fabackend

_parent_try = _packmod.try_pack_paged_kv_to_varlen_attention
_hits = 0
_first_hits = 0
_second_hits = 0
_DIAG_CONTROL = Path(f"{_ROOT}/results/tp2_shortmid_layerdiag_fallback_layers.txt")
_FULL_ATTN_LAYERS = frozenset(range(3, 64, 4))
_diag_control_text = None
_diag_fallback_layers = frozenset()


def _read_diag_fallback_layers():
    global _diag_control_text, _diag_fallback_layers
    try:
        text = _DIAG_CONTROL.read_text().strip()
    except FileNotFoundError:
        text = ""
    if text == _diag_control_text:
        return _diag_fallback_layers
    layers = set()
    if text and text.lower() not in {"none", "off"}:
        if text.lower() == "all":
            layers.update(_FULL_ATTN_LAYERS)
        else:
            for raw in text.replace("\n", ",").replace(" ", ",").split(","):
                raw = raw.strip()
                if not raw:
                    continue
                if "-" in raw:
                    a, b = raw.split("-", 1)
                    a, b = int(a), int(b)
                    layers.update(range(a, b + 1, 4))
                else:
                    layers.add(int(raw))
    invalid = layers - _FULL_ATTN_LAYERS
    if invalid:
        raise RuntimeError(f"TP2 shortmid layerdiag v5 invalid fallback layers: {sorted(invalid)}")
    _diag_control_text = text
    _diag_fallback_layers = frozenset(layers)
    print(
        f"[K100 DFlash2 TP2 shortmid layerdiag v5] control update fallback_layers={sorted(layers)}",
        flush=True,
    )
    return _diag_fallback_layers


def _fallback(*, q, forward_batch, metadata, layer, window_size, sinks,
              key_cache, value_cache, page_size, k_descale, v_descale, **kwargs):
    return _parent_try(
        q=q, forward_batch=forward_batch, metadata=metadata, layer=layer,
        window_size=window_size, sinks=sinks, key_cache=key_cache,
        value_cache=value_cache, page_size=page_size,
        k_descale=k_descale, v_descale=v_descale, **kwargs,
    )


def _tp2_shortmid_varlen_v2(
    *, q, forward_batch, metadata, layer, window_size, sinks,
    key_cache, value_cache, page_size, k_descale, v_descale, **kwargs,
):
    global _hits, _first_hits, _second_hits
    qlen = int(q.shape[0])
    if not (1 <= qlen <= 8191):
        return _fallback(
            q=q, forward_batch=forward_batch, metadata=metadata, layer=layer,
            window_size=window_size, sinks=sinks, key_cache=key_cache,
            value_cache=value_cache, page_size=page_size,
            k_descale=k_descale, v_descale=v_descale, **kwargs,
        )
    if int(getattr(forward_batch, "batch_size", -1)) != 1:
        return _fallback(
            q=q, forward_batch=forward_batch, metadata=metadata, layer=layer,
            window_size=window_size, sinks=sinks, key_cache=key_cache,
            value_cache=value_cache, page_size=page_size,
            k_descale=k_descale, v_descale=v_descale, **kwargs,
        )
    seq = getattr(forward_batch, "seq_lens_cpu", None)
    if seq is None or len(seq) < 1:
        return _fallback(
            q=q, forward_batch=forward_batch, metadata=metadata, layer=layer,
            window_size=window_size, sinks=sinks, key_cache=key_cache,
            value_cache=value_cache, page_size=page_size,
            k_descale=k_descale, v_descale=v_descale, **kwargs,
        )
    kvlen = int(seq[0])
    first_chunk = (2048 <= qlen <= 8191 and kvlen == qlen)
    second_partial = (
        512 <= qlen <= 8191
        and (kvlen - qlen) in (8192, 16384)
        and kvlen <= 20000
    )
    if not (first_chunk or second_partial):
        return _fallback(
            q=q, forward_batch=forward_batch, metadata=metadata, layer=layer,
            window_size=window_size, sinks=sinks, key_cache=key_cache,
            value_cache=value_cache, page_size=page_size,
            k_descale=k_descale, v_descale=v_descale, **kwargs,
        )

    layer_id = int(getattr(layer, "layer_id", -1))
    if layer_id not in _FULL_ATTN_LAYERS:
        raise RuntimeError(f"TP2 shortmid layerdiag v5 unexpected admitted layer_id={layer_id}")
    if layer_id in _read_diag_fallback_layers():
        return _fallback(
            q=q, forward_batch=forward_batch, metadata=metadata, layer=layer,
            window_size=window_size, sinks=sinks, key_cache=key_cache,
            value_cache=value_cache, page_size=page_size,
            k_descale=k_descale, v_descale=v_descale, **kwargs,
        )

    eligible = (
        int(getattr(metadata, "max_seq_len_q", -1)) == qlen
        and int(getattr(metadata, "max_seq_len_k", -1)) == kvlen
        and getattr(metadata, "page_table", None) is not None
        and getattr(metadata, "cu_seqlens_q", None) is not None
        and getattr(metadata, "cu_seqlens_k", None) is not None
        and q.numel() == qlen * 12 * 256
        and q.is_contiguous()
        and q.dtype == torch.bfloat16
        and key_cache.dtype == torch.bfloat16
        and value_cache.dtype == torch.bfloat16
        and int(getattr(layer, "tp_q_head_num", -1)) == 12
        and int(getattr(layer, "tp_k_head_num", -1)) == 2
        and int(getattr(layer, "tp_v_head_num", -1)) == 2
        and int(getattr(layer, "head_dim", -1)) == 256
        and int(getattr(layer, "v_head_dim", -1)) == 256
        and float(getattr(layer, "logit_cap", 0.0)) == 0.0
        and int(page_size) == 64
        and tuple(window_size) == (-1, -1)
        and sinks is None
        and abs(float(getattr(layer, "scaling", -1.0)) - 0.0625) <= 1e-12
        and not bool(getattr(forward_batch, "mha_return_lse", False))
        and metadata.page_table.shape[0] >= 1
        and metadata.page_table.shape[1] >= (kvlen + int(page_size) - 1) // int(page_size)
        and metadata.cu_seqlens_k.shape[0] >= 2
    )
    if not eligible:
        return _fallback(
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
        raise RuntimeError(
            f"TP2 shortmid layerdiag v5 gather mismatch q={qlen} kv={kvlen} "
            f"K={tuple(packed_k.shape)} V={tuple(packed_v.shape)}"
        )

    _hits += 1
    if first_chunk:
        _first_hits += 1
        kind = "first"
    else:
        _second_hits += 1
        kind = "second-partial"
    if _hits <= 12 or _hits % 16 == 0:
        print(
            f"[K100 DFlash2 TP2 shortmid layerdiag v5] ACTIVE hit={_hits} kind={kind} "
            f"q={qlen} kv={kvlen} prefix={kvlen-qlen} first_hits={_first_hits} second_hits={_second_hits} "
            "QH12/KVH2/D256 BM128/BN64/w4",
            flush=True,
        )

    out = _tri_fn(
        q.view(-1, layer.tp_q_head_num, layer.head_dim),
        packed_k,
        packed_v,
        metadata.cu_seqlens_q,
        metadata.cu_seqlens_k,
        qlen,
        kvlen,
        dropout_p=0.0,
        softmax_scale=layer.scaling,
        causal=True,
        return_attn_probs=False,
    )
    return out[0] if isinstance(out, (tuple, list)) else out


_packmod.try_pack_paged_kv_to_varlen_attention = _tp2_shortmid_varlen_v2
_fabackend.try_pack_paged_kv_to_varlen_attention = _tp2_shortmid_varlen_v2
print(
    "[K100 DFlash2 TP2 shortmid layerdiag v5] installed: first q==kv 2K-8K + partial tails prefix=8K/16K, "
    "all admitted full-attention layers use corrected varlen; all other shapes parent fallback",
    flush=True,
)
