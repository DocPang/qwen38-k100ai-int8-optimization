"""U010: native split GDN QKVZ + BA consumers for TP1/M1 decode.

Parent: U008 winner. N7 already fuses input RMSNorm -> INT8 q/scale and carries
(hidden,q,scale) into Qwen3_5GatedDeltaNet._forward_input_proj.  The parent then
uses one Triton fused QKVZ+BA kernel.  U010 intercepts only that exact prequantized
M=1 tuple and launches two native generic-v2 GEMVs sharing the same q/scale:
  QKVZ K5120 -> N16384: waves=4, pipe=3
  BA   K5120 -> N96:    waves=2, pipe=3
All other paths fall back to U008/N7 unchanged.

Real-weight isolated U010 gate showed both outputs bitwise equal to the accepted
fused INT8 consumer. Full-model A/B remains the promotion authority.
"""
from __future__ import annotations

import importlib.util
import os
import runpy
import torch

_BASE = (
    "/data/qwen38-27b-k100ai-int8-opt/"
    "runtime_patch_sglang_u008_native_body_gemv/sitecustomize.py"
)
runpy.run_path(_BASE, run_name="__q38_sglang_u008_native_body_gemv__")

if os.getenv("SGLANG_Q38_NATIVE_GDN_SPLIT_M1", "0") == "1":
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet

    _SO = os.getenv(
        "SGLANG_Q38_NATIVE_GDN_SPLIT_SO",
        "/data/qwen38-27b-k100ai-int8-opt/native_ext/k100_int8_gemv_generic_v2_sglang.so",
    )
    _spec = importlib.util.spec_from_file_location("k100_int8_gemv_generic_v2_sglang", _SO)
    if _spec is None or _spec.loader is None:
        raise RuntimeError(f"cannot load native GDN split GEMV: {_SO}")
    _gemv = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_gemv)

    K = 5120
    NQ = 16384
    NB = 96

    # N5 creates a KN-contiguous BA INT8 shadow for the fused Triton kernel.
    # Native GEMV wants NK-contiguous weight, so build that second view ONCE at
    # model-load time. This is ~0.47MiB/layer (~22.5MiB for 48 layers), not a
    # per-token transpose/copy.
    _prev_unquant_process = UnquantizedLinearMethod.process_weights_after_loading
    _ba_nk_layers = 0

    def _process_with_ba_nk_shadow(self, layer: torch.nn.Module) -> None:
        global _ba_nk_layers
        _prev_unquant_process(self, layer)
        kn = getattr(layer, "k100_q38_ba_weight_int8_kn", None)
        if (
            not isinstance(kn, torch.Tensor)
            or tuple(kn.shape) != (K, NB)
            or kn.dtype is not torch.int8
            or hasattr(layer, "k100_q38_ba_weight_int8_nk")
        ):
            return
        nk = kn.t().contiguous()
        layer.register_buffer("k100_q38_ba_weight_int8_nk", nk, persistent=False)
        _ba_nk_layers += 1
        if _ba_nk_layers in (1, 48):
            print(
                "[K100 SGLang native GDN split] BA NK shadow ready "
                f"layers={_ba_nk_layers} shape=({NB},{K})",
                flush=True,
            )

    UnquantizedLinearMethod.process_weights_after_loading = _process_with_ba_nk_shadow

    _prev_input_proj = Qwen3_5GatedDeltaNet._forward_input_proj
    _seen = False
    _diag_seen = False

    def _native_gdn_split_input_proj(self, hidden_states):
        global _seen, _diag_seen
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
                q_out = _gemv.gemv(
                    q,
                    qkvz_w.t(),
                    s.reshape(-1),
                    qkvz_s.reshape(-1),
                    4,
                    3,
                )
                ba_out = _gemv.gemv(
                    q,
                    ba_w,
                    s.reshape(-1),
                    ba_s.reshape(-1),
                    2,
                    3,
                )
                if not _seen:
                    _seen = True
                    print(
                        "[K100 SGLang native GDN split] ACTIVE M1 prequantized "
                        "QKVZ native(w4,p3) + BA native(w2,p3); no second quant",
                        flush=True,
                    )
                return q_out, ba_out
            if not _diag_seen:
                _diag_seen = True
                def _meta(x):
                    if not isinstance(x, torch.Tensor):
                        return repr(type(x))
                    return {
                        "shape": tuple(x.shape), "dtype": str(x.dtype),
                        "stride": tuple(x.stride()), "contiguous": x.is_contiguous(),
                        "t_contiguous": x.t().is_contiguous() if x.ndim == 2 else None,
                        "numel": x.numel(),
                    }
                print(
                    "[K100 SGLang native GDN split] GUARD_MISS "
                    f"carrier={_meta(carrier)} q={_meta(q)} s={_meta(s)} "
                    f"qkvz_w={_meta(qkvz_w)} qkvz_s={_meta(qkvz_s)} "
                    f"ba_w={_meta(ba_w)} ba_s={_meta(ba_s)}",
                    flush=True,
                )
        return _prev_input_proj(self, hidden_states)

    Qwen3_5GatedDeltaNet._forward_input_proj = _native_gdn_split_input_proj
    print(
        f"[K100 SGLang native GDN split] installed from {_SO}; "
        "only exact TP1/M1 prequantized GDN path intercepted",
        flush=True,
    )
