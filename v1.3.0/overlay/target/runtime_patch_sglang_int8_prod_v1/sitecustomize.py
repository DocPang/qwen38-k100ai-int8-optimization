"""Qwen3.8-27B K100AI INT8 production v1: corrected paged consumer + audited long-KV Triton GQA.

Production contract: only GPU-audited KV lengths may be enabled. Historical promotion required
isolated numeric/speed gates plus full-model semantic qualification for the enabled profile.

It inherits U022 decode behavior. PFX1 is supplied by the launcher's exact
M=8192 W8A8 JSON cache. SourceFind's own can-pack gate and exact physical64 ->
contiguous gather remain authoritative. Only after those checks may one of the
explicitly enabled single-sequence shapes use the fixed U033 Triton winner.

Fail-closed environment contract:
  SGLANG_Q38_U036_KV_LENGTHS=16384,24576,...,253952
Only the explicitly GPU-audited 8K-spaced candidate lengths from 16K through 248K
are accepted; any other value aborts startup instead of silently broadening the
kernel's semantic surface. The 257900-token benchmark's final 3948-token tail is
handled by the packed contiguous fallback, not by this q=8192-only kernel.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import runpy

_ROOT = "/data/qwen38-27b-k100ai-int8-opt"
_BASE = f"{_ROOT}/runtime_patch_sglang_u022_paged_varlen_triton_only/sitecustomize.py"
_SRC = f"{_ROOT}/external_images/sglang0.5.12-20260620-src/flash_attn/flash_attn_triton_mqa_gqa.py"
_ALLOWED_KV = {16384, 24576, 32768, 40960, 49152, 57344, 65536, 73728, 81920, 90112, 98304, 106496, 114688, 122880, 131072, 139264, 147456, 155648, 163840, 172032, 180224, 188416, 196608, 204800, 212992, 221184, 229376, 237568, 245760, 253952}
_raw_lengths = os.environ.get("SGLANG_Q38_U036_KV_LENGTHS", "").strip()
if not _raw_lengths:
    raise RuntimeError(
        "U036 fail-closed: SGLANG_Q38_U036_KV_LENGTHS must be explicitly set; "
        "do not enable unvalidated long-KV shapes implicitly"
    )
try:
    _KV_LENGTHS = {int(x.strip()) for x in _raw_lengths.split(",") if x.strip()}
except ValueError as exc:
    raise RuntimeError(f"U036 invalid KV length list: {_raw_lengths!r}") from exc
if not _KV_LENGTHS or not _KV_LENGTHS.issubset(_ALLOWED_KV):
    raise RuntimeError(
        f"U036 fail-closed: requested KV lengths={sorted(_KV_LENGTHS)}; "
        f"allowed candidates={sorted(_ALLOWED_KV)}"
    )

# Decode/head/GDN/GEMV stack remains U022, while the gfx928 q>=5 paged-varlen correctness repair is inherited from the corrected parent. PFX1 W8A8 cache is external.
runpy.run_path(_BASE, run_name="__q38_sglang_u022_corrected_paged__")

import torch
from sglang.srt.layers.attention import pack_paged_kv_to_varlen as _packmod
from sglang.srt.layers.attention import flashattention_backend as _fabackend

_orig_try = _packmod.try_pack_paged_kv_to_varlen_attention

# Vendor source predates the current Triton parser. For the admitted batch1
# q=8192 grids, grid construction makes this early-return branch unreachable.
os.environ["FLASH_ATTENTION_PRINT_PARAM"] = "0"
_raw = Path(_SRC).read_text()
_old = """            if start_m * BLOCK_M > seqlen_q:\n                return\n"""
_new = """            if start_m * BLOCK_M > seqlen_q:\n                start_m = 0  # U036 batch1 q8192 compatibility; unreachable on admitted grids\n"""
if _raw.count(_old) != 1:
    raise RuntimeError("U036 vendor Triton early-return source changed; fail closed")
_compat = "/tmp/q38_u036_flash_attn_triton_mqa_gqa.py"
Path(_compat).write_text(_raw.replace(_old, _new))
_spec = importlib.util.spec_from_file_location("q38_u036_triton_gqa", _compat)
_tri = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_tri)

# Frozen U033 winner. Do not re-autotune eleven configs in a fresh server.
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
    raise RuntimeError(f"U036 expected one frozen U033 winner, got {len(_winner)}")
_tri.attn_fwd.configs = _winner
if hasattr(_tri.attn_fwd, "cache"):
    _tri.attn_fwd.cache.clear()
if hasattr(_tri.attn_fwd, "best_config"):
    _tri.attn_fwd.best_config = None
_tri_fn = _tri.flash_attn_varlen_func
_hits: dict[int, int] = {k: 0 for k in sorted(_KV_LENGTHS)}


def _eligible(*, q, forward_batch, metadata, layer, window_size, sinks,
              key_cache, value_cache, page_size) -> tuple[bool, int | None]:
    if int(getattr(forward_batch, "batch_size", -1)) != 1:
        return False, None
    # SGLang passes the pre-view Q tensor into this hook. Depending on the model/runtime
    # boundary it may be represented as [T, H*D] or already [T, H, D].  U034 proved
    # the semantic authority is the same contiguous buffer because the accepted path
    # immediately does q.view(-1, H, D).  Gate on exact token count + storage size
    # rather than an unproven ndim representation, while still failing closed on any
    # incompatible/non-contiguous tensor.
    if int(q.shape[0]) != 8192:
        return False, None
    if q.numel() != 8192 * 24 * 256 or not q.is_contiguous():
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
    if int(getattr(layer, "tp_q_head_num", -1)) != 24:
        return False, None
    if int(getattr(layer, "tp_k_head_num", -1)) != 4:
        return False, None
    if int(getattr(layer, "tp_v_head_num", -1)) != 4:
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
    # Qwen3.8 d=256 attention scale is exactly 1/sqrt(256)=1/16.
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


def _u036_try_pack_paged_kv_to_varlen_attention(
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
    # Fail closed if the gather result does not exactly match the admitted shape.
    if int(packed_k.shape[0]) != kv_len or int(packed_v.shape[0]) != kv_len:
        raise RuntimeError(
            f"U036 gather shape mismatch: admitted kv={kv_len}, "
            f"packed_k={tuple(packed_k.shape)}, packed_v={tuple(packed_v.shape)}"
        )
    _hits[kv_len] += 1
    h = _hits[kv_len]
    if h <= 4 or h % 16 == 0:
        print(
            "[K100 SGLang U036 long-KV relaxed Triton GQA] ACTIVE "
            f"kv={kv_len} hit={h} q=8192 qh=24 kvh=4 d=256 "
            "fixed=BM64/BN64/w8/s1",
            flush=True,
        )
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


# flashattention_backend imported the symbol by value; patch both references.
_packmod.try_pack_paged_kv_to_varlen_attention = _u036_try_pack_paged_kv_to_varlen_attention
_fabackend.try_pack_paged_kv_to_varlen_attention = _u036_try_pack_paged_kv_to_varlen_attention
print(
    "[K100 SGLang U036 long-KV relaxed Triton GQA] installed: "
    f"explicit kv_lengths={sorted(_KV_LENGTHS)}; SourceFind gather retained; "
    "only batch1 q8192 Q24/KV4/d256 BF16 page64 uses frozen U033 winner",
    flush=True,
)
