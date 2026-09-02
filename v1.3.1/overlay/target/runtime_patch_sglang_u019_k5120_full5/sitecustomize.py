"""U019: U017 winner stack + K5120 five-segment full-prefetch GEMV.

Parent is U016 deep-down stack. Deployment keeps U008 native body families
`gate_up,full_qkv` enabled, then this outer layer supersedes those two exact
K5120 M=1 shapes with the U018 full-five-prefetch kernel. It also supersedes
the U010 GDN QKVZ direct native call while retaining generic-v2 for tiny BA.

Only exact TP1/M1 INT8 shapes are intercepted:
  gate_up: K5120 -> N34816, full5 waves=4
  full_qkv: K5120 -> N14336, full5 waves=2
  GDN QKVZ: K5120 -> N16384, full5 waves=4
Everything else falls back to U016/U010/U008 unchanged.
"""
from __future__ import annotations

import importlib.util
import os
import runpy
import torch

_BASE = (
    "/data/qwen38-27b-k100ai-int8-opt/"
    "runtime_patch_sglang_u016_deep_down/sitecustomize.py"
)
runpy.run_path(_BASE, run_name="__q38_sglang_u016_deep_down__")

from lmslim import quant_ops as _lmslim_quant_ops
from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet

_ROOT = "/data/qwen38-27b-k100ai-int8-opt"
_FULL5_SO = os.getenv(
    "SGLANG_Q38_K5120_FULL5_SO",
    f"{_ROOT}/native_ext/k100_int8_gemv_k5120_full5_sglang.so",
)
_GENERIC_SO = os.getenv(
    "SGLANG_Q38_NATIVE_GDN_SPLIT_SO",
    f"{_ROOT}/native_ext/k100_int8_gemv_generic_v2_sglang.so",
)

_spec = importlib.util.spec_from_file_location(
    "k100_int8_gemv_k5120_full5_sglang", _FULL5_SO
)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load U019 K5120 full5 GEMV: {_FULL5_SO}")
_full5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_full5)

_gspec = importlib.util.spec_from_file_location(
    "k100_int8_gemv_generic_v2_sglang", _GENERIC_SO
)
if _gspec is None or _gspec.loader is None:
    raise RuntimeError(f"cannot load U019 generic GEMV for GDN BA: {_GENERIC_SO}")
_generic = importlib.util.module_from_spec(_gspec)
_gspec.loader.exec_module(_generic)

# Supersede U008's generic-v2 body interception for the two deployed K5120 body
# shapes. U016 down remains underneath this wrapper and is reached via _prev_mm.
_prev_mm = _lmslim_quant_ops.triton_scaled_mm
_BODY = {
    (5120, 34816): (4, "gate_up"),
    (5120, 14336): (2, "full_qkv"),
}
_body_seen: set[tuple[int, int]] = set()


def _u019_scaled_mm(
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
        and int(b.shape[0]) == int(a.shape[1])
        and b.t().is_contiguous()
        and scale_a.dtype is torch.float32
        and scale_b.dtype is torch.float32
        and scale_a.numel() >= 1
    ):
        k = int(a.shape[1])
        n = int(b.shape[1])
        cfg = _BODY.get((k, n))
        if cfg is not None and scale_b.numel() == n:
            waves, label = cfg
            key = (k, n)
            if key not in _body_seen:
                _body_seen.add(key)
                print(
                    f"[K100 SGLang U019 K5120 full5] ACTIVE {label} "
                    f"M1 K{k}->N{n}; waves={waves}",
                    flush=True,
                )
            return _full5.gemv(
                a,
                b.t(),
                scale_a.reshape(-1),
                scale_b.reshape(-1),
                waves,
            )
    return _prev_mm(a, b, scale_a, scale_b, out_dtype, bias, best_config)


_lmslim_quant_ops.triton_scaled_mm = _u019_scaled_mm

# U010 bypasses lmslim.triton_scaled_mm for prequantized GDN input and calls its
# native module directly, so override that exact high-value path separately.
_prev_gdn_input_proj = Qwen3_5GatedDeltaNet._forward_input_proj
_gdn_seen = False
_gdn_diag_seen = False
K = 5120
NQ = 16384
NB = 96


def _u019_gdn_input_proj(self, hidden_states):
    global _gdn_seen, _gdn_diag_seen
    if isinstance(hidden_states, tuple) and len(hidden_states) == 3:
        carrier, q, s = hidden_states
        qkvz = self.in_proj_qkvz
        ba = self.in_proj_ba
        qkvz_w = getattr(qkvz, "weight", None)
        qkvz_s = getattr(qkvz, "weight_scale", None)
        ba_w = getattr(ba, "k100_q38_ba_weight_int8_nk", None)
        ba_s = getattr(ba, "k100_q38_ba_weight_scale", None)
        if (
            isinstance(carrier, torch.Tensor)
            and tuple(carrier.shape) == (1, K)
            and isinstance(q, torch.Tensor)
            and tuple(q.shape) == (1, K)
            and q.dtype is torch.int8
            and q.is_contiguous()
            and isinstance(s, torch.Tensor)
            and tuple(s.shape) == (1, 1)
            and s.dtype is torch.float32
            and isinstance(qkvz_w, torch.Tensor)
            and qkvz_w.dtype is torch.int8
            and tuple(qkvz_w.shape) == (K, NQ)
            and qkvz_w.t().is_contiguous()
            and isinstance(qkvz_s, torch.Tensor)
            and qkvz_s.dtype is torch.float32
            and qkvz_s.numel() == NQ
            and isinstance(ba_w, torch.Tensor)
            and ba_w.dtype is torch.int8
            and tuple(ba_w.shape) == (NB, K)
            and ba_w.is_contiguous()
            and isinstance(ba_s, torch.Tensor)
            and ba_s.dtype is torch.float32
            and ba_s.numel() == NB
        ):
            q_out = _full5.gemv(
                q,
                qkvz_w.t(),
                s.reshape(-1),
                qkvz_s.reshape(-1),
                4,
            )
            # Tiny BA is not a U018 target; retain U010's accepted generic-v2.
            ba_out = _generic.gemv(
                q,
                ba_w,
                s.reshape(-1),
                ba_s.reshape(-1),
                2,
                3,
            )
            if not _gdn_seen:
                _gdn_seen = True
                print(
                    "[K100 SGLang U019 K5120 full5] ACTIVE GDN QKVZ "
                    "full5(w4) + BA generic-v2(w2,p3); shared q/scale",
                    flush=True,
                )
            return q_out, ba_out
        if not _gdn_diag_seen:
            _gdn_diag_seen = True
            print("[K100 SGLang U019 K5120 full5] GDN GUARD_MISS -> parent U010", flush=True)
    return _prev_gdn_input_proj(self, hidden_states)


Qwen3_5GatedDeltaNet._forward_input_proj = _u019_gdn_input_proj
print(
    f"[K100 SGLang U019 K5120 full5] installed from {_FULL5_SO}; "
    "gate_up/full_qkv/GDN-QKVZ exact M1 only",
    flush=True,
)
