"""N5 speed-first SGLang stack: compact M1 head + fused GDN QKVZ+BA W8A8.

The base N5 compact-head patch remains authoritative. This layer adds only the
M=1 GDN input-projection fast path proven by the isolated GPU0 gate:
  one dynamic per-token INT8 quant -> one fused Triton kernel for
  QKVZ (already checkpoint INT8) + BA (runtime row-wise INT8 shadow).

BA BF16 weights are preserved for every non-M1 path, because the generic lmslim
96x5120 W8A8 GEMM is slower than BF16 on K100AI. Thus prefill/unsupported shapes
fail closed to stock SGLang instead of pursuing "INT8 everywhere" at a loss.
"""
from __future__ import annotations

import os
import runpy

import torch
import triton
import triton.language as tl

_BASE = (
    "/data/qwen38-27b-k100ai-int8-opt/"
    "runtime_patch_sglang_n5_compact_head/sitecustomize.py"
)
runpy.run_path(_BASE, run_name="__q38_sglang_n5_compact_head__")

if os.getenv("SGLANG_Q38_GDN_BA_FUSED_M1", "0") == "1":
    from lmslim.layers.gemm.int8_utils import per_token_quant_int8
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet

    N_QKVZ = 16384
    N_BA = 96
    K = 5120
    _seen_active = False
    _quantized_ba_layers = 0
    # U010 opt-in only; default keeps the accepted N5/N7 geometry unchanged.
    # Rotating/cache-polluted gfx928 gate found BM32/warps8/waves2 exact and
    # materially faster while retaining kpack2/mmac_layout_force1/ds6.
    _GDN_U010_CFG = os.getenv("SGLANG_Q38_GDN_U010_CFG", "0") == "1"

    @triton.jit
    def _gdn_qkvz_ba_kernel(
        a_ptr,
        qkvz_ptr,
        ba_ptr,
        scale_a_ptr,
        qkvz_scale_ptr,
        ba_scale_ptr,
        c_ptr,
        M: tl.constexpr,
        NQ: tl.constexpr,
        NB: tl.constexpr,
        KK: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_qkvz_k: tl.constexpr,
        stride_qkvz_n: tl.constexpr,
        stride_ba_k: tl.constexpr,
        stride_ba_n: tl.constexpr,
        stride_cm: tl.constexpr,
        stride_cn: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        total_n = NQ + NB
        num_pid_n = tl.cdiv(total_n, BN)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
        offsets_m = pid_m * BM + tl.arange(0, BM)
        offsets_n = pid_n * BN + tl.arange(0, BN)
        offsets_k = tl.arange(0, BK)
        mask_m = offsets_m < M
        mask_n = offsets_n < total_n
        a_ptrs = (
            a_ptr
            + offsets_m[:, None] * stride_am
            + offsets_k[None, :] * stride_ak
        )
        acc = tl.zeros((BM, BN), dtype=tl.int32)
        output_n_base = pid_n * BN
        if output_n_base < NQ:
            weight_n = offsets_n
            weight_ptrs = (
                qkvz_ptr
                + offsets_k[:, None] * stride_qkvz_k
                + weight_n[None, :] * stride_qkvz_n
            )
            for _ in range(0, tl.cdiv(KK, BK)):
                mask_k = offsets_k < KK
                av = tl.load(
                    a_ptrs,
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0,
                )
                wv = tl.load(
                    weight_ptrs,
                    mask=mask_k[:, None] & mask_n[None, :],
                    other=0,
                )
                acc = tl.dot(av, wv, acc, out_dtype=tl.int32)
                offsets_k += BK
                a_ptrs += BK * stride_ak
                weight_ptrs += BK * stride_qkvz_k
            scale_b = tl.load(
                qkvz_scale_ptr + weight_n, mask=mask_n, other=0.0
            )[None, :]
        else:
            weight_n = offsets_n - NQ
            mask_weight_n = weight_n < NB
            weight_ptrs = (
                ba_ptr
                + offsets_k[:, None] * stride_ba_k
                + weight_n[None, :] * stride_ba_n
            )
            for _ in range(0, tl.cdiv(KK, BK)):
                mask_k = offsets_k < KK
                av = tl.load(
                    a_ptrs,
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0,
                )
                wv = tl.load(
                    weight_ptrs,
                    mask=mask_k[:, None] & mask_weight_n[None, :],
                    other=0,
                )
                acc = tl.dot(av, wv, acc, out_dtype=tl.int32)
                offsets_k += BK
                a_ptrs += BK * stride_ak
                weight_ptrs += BK * stride_ba_k
            scale_b = tl.load(
                ba_scale_ptr + weight_n, mask=mask_weight_n, other=0.0
            )[None, :]
        scale_a = tl.load(
            scale_a_ptr + offsets_m, mask=mask_m, other=0.0
        )[:, None]
        result = (acc.to(tl.float32) * scale_a * scale_b).to(tl.bfloat16)
        c_ptrs = (
            c_ptr
            + offsets_m[:, None] * stride_cm
            + offsets_n[None, :] * stride_cn
        )
        tl.store(
            c_ptrs,
            result,
            mask=mask_m[:, None] & mask_n[None, :],
        )

    @torch.library.custom_op(
        "k100_q38::sglang_gdn_qkvz_ba_fused_m1",
        mutates_args=(),
        device_types="cuda",
    )
    def _gdn_qkvz_ba_fused_m1(
        hidden_states: torch.Tensor,
        qkvz_weight_kn: torch.Tensor,
        qkvz_scale: torch.Tensor,
        ba_weight_kn: torch.Tensor,
        ba_scale: torch.Tensor,
    ) -> torch.Tensor:
        x = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
        if tuple(x.shape) != (1, K):
            raise RuntimeError(f"GDN fused M1 shape drift: {tuple(x.shape)}")
        q, s = per_token_quant_int8(x)
        out = torch.empty(
            (1, N_QKVZ + N_BA),
            dtype=torch.bfloat16,
            device=x.device,
        )
        grid = (triton.cdiv(1, 16) * triton.cdiv(N_QKVZ + N_BA, 32),)
        _gdn_qkvz_ba_kernel[grid](
            q,
            qkvz_weight_kn,
            ba_weight_kn,
            s,
            qkvz_scale,
            ba_scale,
            out,
            1,
            N_QKVZ,
            N_BA,
            K,
            q.stride(0),
            q.stride(1),
            qkvz_weight_kn.stride(0),
            qkvz_weight_kn.stride(1),
            ba_weight_kn.stride(0),
            ba_weight_kn.stride(1),
            out.stride(0),
            out.stride(1),
            BM=32 if _GDN_U010_CFG else 16,
            BN=32,
            BK=512,
            num_warps=8 if _GDN_U010_CFG else 4,
            num_stages=2,
            waves_per_eu=2 if _GDN_U010_CFG else 1,
            kpack=2,
            mmac_layout_force=1,
            sched_latency="mmac5-ds6",
        )
        return out

    @_gdn_qkvz_ba_fused_m1.register_fake
    def _gdn_qkvz_ba_fused_m1_fake(
        hidden_states: torch.Tensor,
        qkvz_weight_kn: torch.Tensor,
        qkvz_scale: torch.Tensor,
        ba_weight_kn: torch.Tensor,
        ba_scale: torch.Tensor,
    ) -> torch.Tensor:
        del qkvz_weight_kn, qkvz_scale, ba_weight_kn, ba_scale
        return hidden_states.new_empty((1, N_QKVZ + N_BA), dtype=torch.bfloat16)

    @torch.library.custom_op(
        "k100_q38::sglang_gdn_qkvz_ba_fused_prequant_m1",
        mutates_args=(),
        device_types="cuda",
    )
    def _gdn_qkvz_ba_fused_prequant_m1(
        q: torch.Tensor,
        s: torch.Tensor,
        qkvz_weight_kn: torch.Tensor,
        qkvz_scale: torch.Tensor,
        ba_weight_kn: torch.Tensor,
        ba_scale: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(q.shape) != (1, K) or q.dtype is not torch.int8:
            raise RuntimeError(f"GDN prequant M1 q shape/dtype drift: {tuple(q.shape)} {q.dtype}")
        if tuple(s.shape) != (1, 1):
            raise RuntimeError(f"GDN prequant M1 scale shape drift: {tuple(s.shape)}")
        out = torch.empty(
            (1, N_QKVZ + N_BA), dtype=torch.bfloat16, device=q.device
        )
        grid = (triton.cdiv(1, 16) * triton.cdiv(N_QKVZ + N_BA, 32),)
        _gdn_qkvz_ba_kernel[grid](
            q,
            qkvz_weight_kn,
            ba_weight_kn,
            s,
            qkvz_scale,
            ba_scale,
            out,
            1,
            N_QKVZ,
            N_BA,
            K,
            q.stride(0),
            q.stride(1),
            qkvz_weight_kn.stride(0),
            qkvz_weight_kn.stride(1),
            ba_weight_kn.stride(0),
            ba_weight_kn.stride(1),
            out.stride(0),
            out.stride(1),
            BM=32 if _GDN_U010_CFG else 16,
            BN=32,
            BK=512,
            num_warps=8 if _GDN_U010_CFG else 4,
            num_stages=2,
            waves_per_eu=2 if _GDN_U010_CFG else 1,
            kpack=2,
            mmac_layout_force=1,
            sched_latency="mmac5-ds6",
        )
        return out

    @_gdn_qkvz_ba_fused_prequant_m1.register_fake
    def _gdn_qkvz_ba_fused_prequant_m1_fake(
        q: torch.Tensor,
        s: torch.Tensor,
        qkvz_weight_kn: torch.Tensor,
        qkvz_scale: torch.Tensor,
        ba_weight_kn: torch.Tensor,
        ba_scale: torch.Tensor,
    ) -> torch.Tensor:
        del s, qkvz_weight_kn, qkvz_scale, ba_weight_kn, ba_scale
        return q.new_empty((1, N_QKVZ + N_BA), dtype=torch.bfloat16)

    _orig_unquant_process = UnquantizedLinearMethod.process_weights_after_loading

    def _process_unquant_with_ba_shadow(self, layer: torch.nn.Module) -> None:
        global _quantized_ba_layers
        _orig_unquant_process(self, layer)
        w = getattr(layer, "weight", None)
        if (
            not isinstance(w, torch.Tensor)
            or tuple(w.shape) != (N_BA, K)
            or w.dtype not in (torch.bfloat16, torch.float16)
            or hasattr(layer, "k100_q38_ba_weight_int8_kn")
        ):
            return
        with torch.no_grad():
            w32 = w.float()
            scale = w32.abs().amax(dim=1).clamp_min_(1e-12).div_(127.0)
            q_nk = torch.round(w32 / scale[:, None]).clamp_(-127, 127).to(torch.int8)
            q_kn = q_nk.t().contiguous()
        layer.register_buffer(
            "k100_q38_ba_weight_int8_kn", q_kn, persistent=False
        )
        layer.register_buffer(
            "k100_q38_ba_weight_scale", scale[:, None].float().contiguous(), persistent=False
        )
        del w32, q_nk, q_kn, scale
        _quantized_ba_layers += 1
        if _quantized_ba_layers in (1, 48):
            print(
                "[K100 SGLang GDN BA W8A8] runtime shadow ready "
                f"layers={_quantized_ba_layers} shape=({K},{N_BA}); BF16 fallback preserved",
                flush=True,
            )

    UnquantizedLinearMethod.process_weights_after_loading = _process_unquant_with_ba_shadow

    _orig_input_proj = Qwen3_5GatedDeltaNet._forward_input_proj

    def _input_proj_fused_m1(self, hidden_states: torch.Tensor):
        global _seen_active
        qkvz = self.in_proj_qkvz
        ba = self.in_proj_ba
        if (
            hidden_states.ndim == 2
            and tuple(hidden_states.shape) == (1, K)
            and getattr(qkvz, "weight", None) is not None
            and qkvz.weight.dtype is torch.int8
            and tuple(qkvz.weight.shape) == (K, N_QKVZ)
            and hasattr(qkvz, "weight_scale")
            and hasattr(ba, "k100_q38_ba_weight_int8_kn")
            and tuple(ba.k100_q38_ba_weight_int8_kn.shape) == (K, N_BA)
        ):
            if not _seen_active:
                _seen_active = True
                print(
                    "[K100 SGLang GDN QKVZ+BA W8A8] ACTIVE M=1 fused one-quant path",
                    flush=True,
                )
            out = torch.ops.k100_q38.sglang_gdn_qkvz_ba_fused_m1(
                hidden_states,
                qkvz.weight,
                qkvz.weight_scale,
                ba.k100_q38_ba_weight_int8_kn,
                ba.k100_q38_ba_weight_scale,
            )
            return out[:, :N_QKVZ], out[:, N_QKVZ:]
        return _orig_input_proj(self, hidden_states)

    Qwen3_5GatedDeltaNet._forward_input_proj = _input_proj_fused_m1
    print(
        "[K100 SGLang GDN QKVZ+BA W8A8] installed: M1 fused; non-M1 stock fallback",
        flush=True,
    )
