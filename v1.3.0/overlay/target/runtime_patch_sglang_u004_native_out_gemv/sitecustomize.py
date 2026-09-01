"""U004: SGLang TP1/M1 native K100AI INT8 output-projection GEMV.

Single-variable parent: accepted N6 exact SwiGLU speed-first stack.
Only dynamic W8A8 calls with M=1, K=6144, N=5120, bias=None are redirected
from lmslim Triton matmul to the already-proven K100AI native GEMV v7.
All prefill/non-M1/other shapes fall back unchanged.

The checkpoint's channelwise INT8 weight is stored contiguous [N,K] and SGLang
exposes it post-load as a transpose [K,N] view. Therefore b.t() is a zero-copy
contiguous [N,K] view, exactly matching the native extension ABI.
"""
from __future__ import annotations

import importlib.util
import os
import runpy
import torch

_BASE = (
    "/data/qwen38-27b-k100ai-int8-opt/"
    "runtime_patch_sglang_n7_rms_gdn_exact/sitecustomize.py"
)
runpy.run_path(_BASE, run_name="__q38_sglang_n7_rms_gdn_exact__")

if os.getenv("SGLANG_Q38_NATIVE_OUT_GEMV_M1", "0") == "1":
    from lmslim import quant_ops as _lmslim_quant_ops

    _SO = os.getenv(
        "SGLANG_Q38_NATIVE_OUT_GEMV_SO",
        "/data/qwen38-27b-k100ai-int8-opt/native_ext/k100_int8_gemv_v7_sglang.so",
    )
    _spec = importlib.util.spec_from_file_location("k100_int8_gemv_v7_sglang", _SO)
    if _spec is None or _spec.loader is None:
        raise RuntimeError(f"cannot load native GEMV: {_SO}")
    _gemv = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_gemv)

    _orig = _lmslim_quant_ops.triton_scaled_mm
    _seen = False

    def _native_out_scaled_mm(
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        out_dtype: torch.dtype,
        bias=None,
        best_config=None,
    ) -> torch.Tensor:
        global _seen
        if (
            bias is None
            and out_dtype is torch.bfloat16
            and isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and a.dtype is torch.int8
            and b.dtype is torch.int8
            and a.ndim == 2
            and b.ndim == 2
            and tuple(a.shape) == (1, 6144)
            and tuple(b.shape) == (6144, 5120)
            and a.is_contiguous()
            and b.t().is_contiguous()
            and scale_a.dtype is torch.float32
            and scale_b.dtype is torch.float32
            and scale_a.numel() >= 1
            and scale_b.numel() == 5120
        ):
            if not _seen:
                _seen = True
                print(
                    "[K100 SGLang native out GEMV] ACTIVE M1 K6144->N5120 "
                    "zero-copy b.t(); waves=2 pipe=3",
                    flush=True,
                )
            return _gemv.gemv(
                a,
                b.t(),
                scale_a.reshape(-1),
                scale_b.reshape(-1),
                2,
                3,
            )
        return _orig(a, b, scale_a, scale_b, out_dtype, bias, best_config)

    _lmslim_quant_ops.triton_scaled_mm = _native_out_scaled_mm
    print(f"[K100 SGLang native out GEMV] installed from {_SO}; non-M1/other shapes stock", flush=True)
