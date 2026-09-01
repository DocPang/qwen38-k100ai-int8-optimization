"""TP2-V1: extend accepted U022 K5120 LDS-x to TP2 rank-local output shapes.

Parent: U022/PFX1 stack. This patch changes only M=1 BF16-output W8A8 matmuls
whose input K remains 5120 under TP2 but whose column-parallel N is halved:
  gate_up      K5120 -> N17408
  GDN QKVZ     K5120 -> N8192
  full QKV     K5120 -> N7168
All other shapes fall through to U022/stock. The existing U022 .so accepts
arbitrary N; isolated TP2 gates proved eager + CUDAGraph bitwise equality
against the production fallback for all three shapes.
"""
from __future__ import annotations
import importlib.util
import os
import runpy
import torch

_BASE = (
    "/data/qwen38-27b-k100ai-int8-opt/"
    "runtime_patch_sglang_u022_k5120_ldsx/sitecustomize.py"
)
runpy.run_path(_BASE, run_name="__q38_sglang_u022_parent_for_tp2__")

if os.getenv("SGLANG_Q38_TP2_K5120_LDSX_M1", "0") == "1":
    from lmslim import quant_ops as _lmslim_quant_ops

    _SO = os.getenv(
        "SGLANG_Q38_K5120_LDSX_SO",
        "/data/qwen38-27b-k100ai-int8-opt/native_ext/k100_int8_gemv_k5120_ldsx_v1_sglang.so",
    )
    _spec = importlib.util.spec_from_file_location("k100_int8_gemv_k5120_ldsx_v1_sglang", _SO)
    if _spec is None or _spec.loader is None:
        raise RuntimeError(f"cannot load TP2 K5120 LDS-x: {_SO}")
    _gemv = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_gemv)

    _prev = _lmslim_quant_ops.triton_scaled_mm
    _SHAPES = {
        (5120, 17408): (2, "gate_up_tp2"),
        (5120, 8192): (2, "gdn_qkvz_tp2"),
        (5120, 7168): (2, "full_qkv_tp2"),
    }
    _seen: set[tuple[int, int]] = set()

    def _tp2_k5120_ldsx_scaled_mm(
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
            and int(b.shape[0]) == int(a.shape[1])
        ):
            k = int(a.shape[1]); n = int(b.shape[1])
            cfg = _SHAPES.get((k, n))
            if cfg is not None and scale_b.numel() == n:
                waves, label = cfg
                key = (k, n)
                if key not in _seen:
                    _seen.add(key)
                    print(
                        f"[K100 SGLang TP2 K5120 LDS-x] ACTIVE {label} "
                        f"M1 K{k}->N{n}; waves={waves}",
                        flush=True,
                    )
                return _gemv.gemv(
                    a, b.t(), scale_a.reshape(-1), scale_b.reshape(-1), waves
                )
        return _prev(a, b, scale_a, scale_b, out_dtype, bias, best_config)

    _lmslim_quant_ops.triton_scaled_mm = _tp2_k5120_ldsx_scaled_mm
    print(
        f"[K100 SGLang TP2 K5120 LDS-x] installed from {_SO}; "
        "rank-local shapes only; all others inherit U022/stock",
        flush=True,
    )
