"""U016: stable U010 GDN split + dedicated deep-pipeline down GEMV.

Parent: U010 native GDN split stack, but deployment must set the U008 body-family
gate to `gate_up` only. This outer wrapper intercepts only the remaining M=1
MLP down projection K17408->N5120 and routes it to deep-v4 waves=4,depth=8.
Full-attention QKV therefore stays on the stock/tuned lmslim path, avoiding the
intermittent long CUDA-Graph replay hang observed when native full_qkv was enabled.

U015 rotating/cache-polluted same-runtime gate:
  generic-v2 best: 163.84 us/site
  deep-v4 w4d8:    149.28 us/site (~8.9% faster)
  1000 graph replays + 20 random-input rewrites: all bitwise exact.
"""
from __future__ import annotations

import importlib.util
import os
import runpy
import torch

_BASE = (
    "/data/qwen38-27b-k100ai-int8-opt/"
    "runtime_patch_sglang_u010_native_gdn_split/sitecustomize.py"
)
runpy.run_path(_BASE, run_name="__q38_sglang_u010_native_gdn_split__")

if os.getenv("SGLANG_Q38_DEEP_DOWN_GEMV_M1", "0") == "1":
    from lmslim import quant_ops as _lmslim_quant_ops

    _SO = os.getenv(
        "SGLANG_Q38_DEEP_DOWN_GEMV_SO",
        "/data/qwen38-27b-k100ai-int8-opt/native_ext/k100_int8_gemv_deep_v4_sglang.so",
    )
    _spec = importlib.util.spec_from_file_location("k100_int8_gemv_deep_v4_sglang", _SO)
    if _spec is None or _spec.loader is None:
        raise RuntimeError(f"cannot load deep down GEMV: {_SO}")
    _deep = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_deep)

    _prev = _lmslim_quant_ops.triton_scaled_mm
    _seen = False

    def _deep_down_scaled_mm(
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
            and tuple(a.shape) == (1, 17408)
            and tuple(b.shape) == (17408, 5120)
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
                    "[K100 SGLang deep down GEMV] ACTIVE M1 K17408->N5120 "
                    "zero-copy b.t(); waves=4 depth=8",
                    flush=True,
                )
            return _deep.gemv(
                a,
                b.t(),
                scale_a.reshape(-1),
                scale_b.reshape(-1),
                4,
                8,
            )
        return _prev(a, b, scale_a, scale_b, out_dtype, bias, best_config)

    _lmslim_quant_ops.triton_scaled_mm = _deep_down_scaled_mm
    print(
        f"[K100 SGLang deep down GEMV] installed from {_SO}; "
        "only M1 K17408->N5120 intercepted",
        flush=True,
    )
