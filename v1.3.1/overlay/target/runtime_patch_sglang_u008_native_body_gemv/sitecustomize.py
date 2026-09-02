"""U008: SGLang TP1/M1 native INT8 GEMV for remaining dense body shapes.

Parent: U004 winner (N7 RMS->GDN INT8 + exact SwiGLU->INT8 + compact head +
GDN QKVZ+BA fused INT8 + specialized K6144->N5120 output GEMV v7).

This patch redirects only the remaining plain lmslim M=1 W8A8 matmuls:
  gate_up:        K5120  -> N34816  generic-v2 waves=4 pipe=3
  down:           K17408 -> N5120   generic-v2 waves=1 pipe=3
  full-attn QKV:  K5120  -> N14336  generic-v2 waves=2 pipe=3

GDN input stays on the accepted fused QKVZ+BA kernel. Output projection stays on
U004's specialized v7 kernel. Prefill/non-M1/other shapes fall back unchanged.
Isolated rotating/cache-polluted gates were bitwise exact vs current lmslim for
all three shapes.
"""
from __future__ import annotations

import importlib.util
import os
import runpy
import torch

_BASE = (
    "/data/qwen38-27b-k100ai-int8-opt/"
    "runtime_patch_sglang_u004_native_out_gemv/sitecustomize.py"
)
runpy.run_path(_BASE, run_name="__q38_sglang_u004_native_out_gemv__")

if os.getenv("SGLANG_Q38_NATIVE_BODY_GEMV_M1", "0") == "1":
    from lmslim import quant_ops as _lmslim_quant_ops

    _SO = os.getenv(
        "SGLANG_Q38_NATIVE_BODY_GEMV_SO",
        "/data/qwen38-27b-k100ai-int8-opt/native_ext/k100_int8_gemv_generic_v2_sglang.so",
    )
    _spec = importlib.util.spec_from_file_location("k100_int8_gemv_generic_v2_sglang", _SO)
    if _spec is None or _spec.loader is None:
        raise RuntimeError(f"cannot load native body GEMV: {_SO}")
    _gemv = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_gemv)

    _prev = _lmslim_quant_ops.triton_scaled_mm
    _ALL_SHAPES = {
        (5120, 34816): (4, 3, "gate_up"),
        (17408, 5120): (1, 3, "down"),
        (5120, 14336): (2, 3, "full_qkv"),
    }
    _families_raw = os.getenv(
        "SGLANG_Q38_NATIVE_BODY_GEMV_FAMILIES", "gate_up,down,full_qkv"
    )
    _FAMILIES = {x.strip() for x in _families_raw.split(",") if x.strip()}
    _unknown = _FAMILIES - {v[2] for v in _ALL_SHAPES.values()}
    if _unknown:
        raise RuntimeError(f"unknown native body GEMV families: {sorted(_unknown)}")
    _SHAPES = {k: v for k, v in _ALL_SHAPES.items() if v[2] in _FAMILIES}
    _seen: set[tuple[int, int]] = set()

    def _native_body_scaled_mm(
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        out_dtype: torch.dtype,
        bias=None,
        best_config=None,
    ) -> torch.Tensor:
        if (
            bias is None
            and out_dtype is torch.bfloat16
            and isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and a.dtype is torch.int8
            and b.dtype is torch.int8
            and a.ndim == 2
            and b.ndim == 2
            and a.shape[0] == 1
            and a.is_contiguous()
            and b.t().is_contiguous()
            and scale_a.dtype is torch.float32
            and scale_b.dtype is torch.float32
            and scale_a.numel() >= 1
        ):
            k = int(a.shape[1])
            if int(b.shape[0]) == k:
                n = int(b.shape[1])
                cfg = _SHAPES.get((k, n))
                if cfg is not None and scale_b.numel() == n:
                    waves, pipe, label = cfg
                    key = (k, n)
                    if key not in _seen:
                        _seen.add(key)
                        print(
                            f"[K100 SGLang native body GEMV] ACTIVE {label} "
                            f"M1 K{k}->N{n} zero-copy b.t(); waves={waves} pipe={pipe}",
                            flush=True,
                        )
                    return _gemv.gemv(
                        a,
                        b.t(),
                        scale_a.reshape(-1),
                        scale_b.reshape(-1),
                        waves,
                        pipe,
                    )
        return _prev(a, b, scale_a, scale_b, out_dtype, bias, best_config)

    _lmslim_quant_ops.triton_scaled_mm = _native_body_scaled_mm
    print(
        f"[K100 SGLang native body GEMV] installed from {_SO}; "
        f"families={sorted(_FAMILIES)}; other shapes inherit U004/stock",
        flush=True,
    )
