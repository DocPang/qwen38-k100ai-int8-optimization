"""Qwen3.8-27B K100AI TP4 corrected stack + audited U036 long-KV Triton GQA.

Parent:
  runtime_patch_sglang_tp4_row_ldsx_varlenfix_v1
which supplies TP4 rank-local M1 optimizations plus the proven gfx928 q>=5
paged-varlen correctness repair while retaining native decode/with-kvcache.

This child intercepts only the exact TP4 rank-local full-attention prefill shape:
  batch1, q=8192, QH=6, KVH=1, D=256, BF16, page64, full causal attention.
The accepted paged KV is first gathered by SourceFind's proven exact pack path,
then evaluated by the frozen U036 BM64/BN64/w8 Triton GQA kernel.

Fail-closed env:
  SGLANG_Q38_TP4_U036_KV_LENGTHS=16384,24576,...,253952
Only explicitly audited lengths are enabled. All other shapes fall back to the
parent corrected consumer/packed-contiguous path.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import runpy

_ROOT = "/data/qwen38-27b-k100ai-int8-opt"
_PARENT = f"{_ROOT}/runtime_patch_sglang_tp4_row_ldsx_varlenfix_v1/sitecustomize.py"
_SRC = f"{_ROOT}/external_images/sglang0.5.12-20260620-src/flash_attn/flash_attn_triton_mqa_gqa.py"
_ALLOWED_KV = {
    16384, 24576, 32768, 40960, 49152, 57344, 65536, 73728,
    81920, 90112, 98304, 106496, 114688, 122880, 131072,
    139264, 147456, 155648, 163840, 172032, 180224, 188416,
    196608, 204800, 212992, 221184, 229376, 237568, 245760, 253952,
}

_raw_lengths = os.environ.get("SGLANG_Q38_TP4_U036_KV_LENGTHS", "").strip()
if not _raw_lengths:
    raise RuntimeError(
        "TP4-U036 fail-closed: SGLANG_Q38_TP4_U036_KV_LENGTHS must be explicitly set"
    )
try:
    _KV_LENGTHS = {int(x.strip()) for x in _raw_lengths.split(",") if x.strip()}
except ValueError as exc:
    raise RuntimeError(f"TP4-U036 invalid KV length list: {_raw_lengths!r}") from exc
if not _KV_LENGTHS or not _KV_LENGTHS.issubset(_ALLOWED_KV):
    raise RuntimeError(
        f"TP4-U036 fail-closed: requested={sorted(_KV_LENGTHS)} allowed={sorted(_ALLOWED_KV)}"
    )

runpy.run_path(_PARENT, run_name="__q38_sglang_tp4_corrected_parent__")

import torch
import triton
from sglang.srt.layers.attention import pack_paged_kv_to_varlen as _packmod
from sglang.srt.layers.attention import flashattention_backend as _fabackend

_orig_try = _packmod.try_pack_paged_kv_to_varlen_attention
_BLOCK_M = int(os.environ.get("SGLANG_Q38_TP4_U036_BLOCK_M", "64"))
_PROFILE = os.environ.get("SGLANG_Q38_TP4_U036_PROFILE", "legacy_bm64_w8").strip()
_SPLIT_KV = int(os.environ.get("SGLANG_Q38_TP4_U036_SPLIT_KV", "1"))
_ALLOWED_PROFILES = {"legacy_bm64_w8", "ranklocal_bm64_w4_preloadv", "legacy_bm128_w8"}
if _PROFILE not in _ALLOWED_PROFILES:
    raise RuntimeError(f"TP4-U036 unsupported profile={_PROFILE!r}; allowed={sorted(_ALLOWED_PROFILES)}")
if _PROFILE == "legacy_bm128_w8" and _BLOCK_M != 128:
    raise RuntimeError("TP4-U036 legacy_bm128_w8 requires BLOCK_M=128")
if _PROFILE != "legacy_bm128_w8" and _BLOCK_M != 64:
    raise RuntimeError(f"TP4-U036 profile={_PROFILE} requires BLOCK_M=64")
if _SPLIT_KV not in (1, 4):
    raise RuntimeError(f"TP4-U036 unsupported SPLIT_KV={_SPLIT_KV}; allowed=[1,4]")
if _SPLIT_KV == 4 and _PROFILE != "ranklocal_bm64_w4_preloadv":
    raise RuntimeError("TP4-U036 split4 is qualified only with ranklocal_bm64_w4_preloadv")

os.environ["FLASH_ATTENTION_PRINT_PARAM"] = "0"
_raw = Path(_SRC).read_text()
_old = """            if start_m * BLOCK_M > seqlen_q:\n                return\n"""
_new = """            if start_m * BLOCK_M > seqlen_q:\n                start_m = 0  # TP4-U036 batch1 q8192 compatibility; unreachable on admitted grids\n"""
if _raw.count(_old) != 1:
    raise RuntimeError("TP4-U036 vendor Triton early-return source changed; fail closed")
_compat = f"/tmp/q38_tp4_u036_flash_attn_{os.getpid()}.py"
Path(_compat).write_text(_raw.replace(_old, _new))
_spec = importlib.util.spec_from_file_location(f"q38_tp4_u036_triton_gqa_{os.getpid()}", _compat)
_tri = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_tri)

if _PROFILE == "legacy_bm128_w8":
    # SourceFind's shipped autotune list has BM128/w4 but not the audited
    # BM128/BN64/w8/s1 winner, so inject the exact microbench-qualified config.
    _winner = [triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "waves_per_eu": 0, "PRE_LOAD_V": False},
        num_stages=1,
        num_warps=8,
    )]
elif _PROFILE == "ranklocal_bm64_w4_preloadv":
    # 2026-08-21 TP4 rank-local QH6/KVH1 isolated sweep winner.
    # Stable at KV64K/128K/254K with ~1.20x speedup vs legacy BM64/w8.
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
        raise RuntimeError(f"TP4-U036 expected one ranklocal BM64/w4/preloadV winner, got {len(_winner)}")
else:
    _winner = []
    for _c in getattr(_tri.attn_fwd, "configs", []):
        _kw = dict(getattr(_c, "kwargs", {}) or {})
        if (
            _kw.get("BLOCK_M") == 64
            and _kw.get("BLOCK_N") == 64
            and _kw.get("waves_per_eu") == 0
            and _kw.get("PRE_LOAD_V") is False
            and int(getattr(_c, "num_stages", -1)) == 1
            and int(getattr(_c, "num_warps", -1)) == 8
        ):
            _winner.append(_c)
    if len(_winner) != 1:
        raise RuntimeError(f"TP4-U036 expected one frozen legacy BM64 winner, got {len(_winner)}")
_tri.attn_fwd.configs = _winner
if hasattr(_tri.attn_fwd, "cache"):
    _tri.attn_fwd.cache.clear()
if hasattr(_tri.attn_fwd, "best_config"):
    _tri.attn_fwd.best_config = None
_tri_fn = _tri.flash_attn_varlen_func
_hits: dict[int, int] = {k: 0 for k in sorted(_KV_LENGTHS)}


def _split4_attention(q: torch.Tensor, packed_k: torch.Tensor, packed_v: torch.Tensor, layer):
    """Exact 4-way KV split for admitted TP4 q8192 long-prefill geometry.

    Prefix splits are fully visible; only the final split uses bottom-right causal
    masking. Segment outputs are merged with the exact log-sum-exp identity.
    Enabled only for KV>=64K where each split remains >=q_len.
    """
    q_len = int(q.shape[0])
    kv_len = int(packed_k.shape[0])
    if kv_len < 65536 or kv_len // 4 < q_len:
        return None
    qv = q.view(-1, layer.tp_q_head_num, layer.head_dim)
    qh = int(layer.tp_q_head_num)
    kh = int(layer.tp_k_head_num)
    d = int(layer.head_dim)
    bounds = [(kv_len * i) // 4 for i in range(5)]
    outs = []
    lses = []
    for i in range(4):
        ks = packed_k[bounds[i]:bounds[i + 1]]
        vs = packed_v[bounds[i]:bounds[i + 1]]
        kl = int(ks.shape[0])
        cuq = torch.tensor([0, q_len], device=q.device, dtype=torch.int32)
        cuk = torch.tensor([0, kl], device=q.device, dtype=torch.int32)
        o = torch.empty_like(qv, dtype=vs.dtype)
        m = torch.empty((1, qh, q_len), device=q.device, dtype=torch.float32)
        qs = (0, qv.stride(1), qv.stride(0), qv.stride(2))
        kst = (0, ks.stride(1), ks.stride(0), ks.stride(2))
        vst = (0, vs.stride(1), vs.stride(0), vs.stride(2))
        ost = (0, o.stride(1), o.stride(0), o.stride(2))
        zeros = (0, 0, 0, 0)
        az = (0, 0)
        causal = i == 3
        it = 2 if causal else 1
        grid = lambda META: (triton.cdiv(q_len, META["BLOCK_M"] * it), qh, 1)
        _tri.attn_fwd[grid](
            qv, ks, vs, None, layer.scaling, m, o,
            *qs, *kst, *vst, *ost, *zeros, *az, cuq, cuk,
            dropout_p=0.0, philox_seed=0x1BF52, philox_offset_base=0x1D4B42,
            encoded_softmax=None, alibi_slopes=None,
            HQ=qh, HK=kh, ACTUAL_BLOCK_DMODEL=d,
            MAX_SEQLENS_Q=q_len, MAX_SEQLENS_K=kl,
            VARLEN=True, IS_CAUSAL=causal, BLOCK_DMODEL=256, BIAS_TYPE=0,
            ENABLE_DROPOUT=False, RETURN_ENCODED_SOFTMAX=False,
            USE_ALIBI=False, BATCH_SIZE=q_len,
        )
        outs.append(o)
        lses.append(m[0].transpose(0, 1).contiguous())
    mf = torch.stack(lses, 0)
    mm = mf.max(0).values
    w = torch.exp2(mf - mm.unsqueeze(0))
    of = torch.stack([x.float() for x in outs], 0)
    merged = (of * w.unsqueeze(-1)).sum(0) / w.sum(0).unsqueeze(-1)
    return merged.to(torch.bfloat16)


def _eligible(*, q, forward_batch, metadata, layer, window_size, sinks,
              key_cache, value_cache, page_size) -> tuple[bool, int | None]:
    if int(getattr(forward_batch, "batch_size", -1)) != 1:
        return False, None
    if int(q.shape[0]) != 8192:
        return False, None
    if q.numel() != 8192 * 6 * 256 or not q.is_contiguous():
        return False, None
    if int(getattr(metadata, "max_seq_len_q", -1)) != 8192:
        return False, None
    seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
    if seq_lens_cpu is None or len(seq_lens_cpu) < 1:
        return False, None
    kv_len = int(seq_lens_cpu[0])
    if kv_len not in _KV_LENGTHS:
        return False, None
    if int(getattr(metadata, "max_seq_len_k", -1)) != kv_len:
        return False, None
    if int(page_size) != 64:
        return False, None
    if int(getattr(layer, "tp_q_head_num", -1)) != 6:
        return False, None
    if int(getattr(layer, "tp_k_head_num", -1)) != 1:
        return False, None
    if int(getattr(layer, "tp_v_head_num", -1)) != 1:
        return False, None
    if int(getattr(layer, "head_dim", -1)) != 256:
        return False, None
    if int(getattr(layer, "v_head_dim", -1)) != 256:
        return False, None
    if q.dtype != torch.bfloat16:
        return False, None
    if key_cache.dtype != torch.bfloat16 or value_cache.dtype != torch.bfloat16:
        return False, None
    if tuple(window_size) != (-1, -1) or sinks is not None:
        return False, None
    if abs(float(getattr(layer, "scaling", -1.0)) - 0.0625) > 1e-12:
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
    return True, kv_len


def _tp4_u036_try_pack_paged_kv_to_varlen_attention(
    *, q: torch.Tensor, forward_batch, metadata, layer, window_size: tuple,
    sinks, key_cache: torch.Tensor, value_cache: torch.Tensor, page_size: int,
    k_descale, v_descale, **kwargs,
):
    ok, kv_len = _eligible(
        q=q, forward_batch=forward_batch, metadata=metadata, layer=layer,
        window_size=window_size, sinks=sinks, key_cache=key_cache,
        value_cache=value_cache, page_size=page_size,
    )
    if not ok:
        return _orig_try(
            q=q, forward_batch=forward_batch, metadata=metadata, layer=layer,
            window_size=window_size, sinks=sinks, key_cache=key_cache,
            value_cache=value_cache, page_size=page_size,
            k_descale=k_descale, v_descale=v_descale, **kwargs,
        )

    assert kv_len is not None
    packed_k, packed_v = _packmod.pack_paged_kv_to_varlen(
        key_cache.view(-1, layer.tp_k_head_num, page_size, layer.head_dim),
        value_cache.view(-1, layer.tp_v_head_num, layer.v_head_dim, page_size),
        metadata.page_table,
        forward_batch.seq_lens_cpu[: forward_batch.batch_size],
        page_size,
    )
    if int(packed_k.shape[0]) != kv_len or int(packed_v.shape[0]) != kv_len:
        raise RuntimeError(
            f"TP4-U036 gather shape mismatch kv={kv_len} "
            f"K={tuple(packed_k.shape)} V={tuple(packed_v.shape)}"
        )

    _hits[kv_len] += 1
    h = _hits[kv_len]
    if h <= 4 or h % 16 == 0:
        print(
            "[K100 SGLang TP4 U036 long-KV] ACTIVE "
            f"kv={kv_len} hit={h} q=8192 qh=6 kvh=1 d=256 "
            f"fixed_profile={_PROFILE} split_kv={_SPLIT_KV if kv_len >= 65536 else 1}",
            flush=True,
        )

    if _SPLIT_KV == 4 and kv_len >= 65536:
        split_out = _split4_attention(q, packed_k, packed_v, layer)
        if split_out is None:
            raise RuntimeError(f"TP4-U036 split4 gate unexpectedly rejected admitted kv={kv_len}")
        return split_out

    out = _tri_fn(
        q.view(-1, layer.tp_q_head_num, layer.head_dim),
        packed_k,
        packed_v,
        metadata.cu_seqlens_q,
        metadata.cu_seqlens_k,
        metadata.max_seq_len_q,
        metadata.max_seq_len_k,
        dropout_p=0.0,
        softmax_scale=layer.scaling,
        causal=True,
        return_attn_probs=False,
    )
    return out[0] if isinstance(out, (tuple, list)) else out


_packmod.try_pack_paged_kv_to_varlen_attention = _tp4_u036_try_pack_paged_kv_to_varlen_attention
_fabackend.try_pack_paged_kv_to_varlen_attention = _tp4_u036_try_pack_paged_kv_to_varlen_attention
print(
    "[K100 SGLang TP4 U036 long-KV] installed: "
    f"kv_lengths={sorted(_KV_LENGTHS)}; profile={_PROFILE}; split_kv={_SPLIT_KV}; exact SourceFind gather retained; "
    f"only batch1 q8192 Q6/KV1/d256 BF16 page64 uses frozen U036 BM{_BLOCK_M}",
    flush=True,
)
