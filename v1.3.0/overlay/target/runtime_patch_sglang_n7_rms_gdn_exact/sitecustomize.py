"""N7: TP1/M1 input Gemma RMSNorm -> prequantized fused GDN QKVZ+BA.

Parent: N6 exact SwiGLU speed-first stack.  Only the 48 linear-attention
(Qwen3_5LinearDecoderLayer) input norms are specialized. Full-attention layers
stay stock so this remains a single-variable GDN producer/consumer A/B.

For a GDN M=1 layer:
  stock: residual-add + Gemma RMSNorm -> BF16 hidden -> dynamic INT8 quant
         -> fused QKVZ+BA W8A8
  N7:    residual-add + Gemma RMSNorm -> directly (INT8 q, scale)
         -> fused QKVZ+BA consumes prequantized q/scale
The residual buffer is updated in-place exactly as stock when present. The first
layer (residual=None) preserves stock residual semantics by retaining the
original hidden tensor as residual while quantizing the normalized output.
"""
from __future__ import annotations
import os, runpy
import torch

_BASE=("/data/qwen38-27b-k100ai-int8-opt/"
      "runtime_patch_sglang_n6_swiglu_exact/sitecustomize.py")
runpy.run_path(_BASE, run_name="__q38_sglang_n6_swiglu_exact__")

if os.getenv("SGLANG_Q38_RMS_GDN_INT8_M1","0")=="1":
    from lmslim.quantize.quant_ops import lm_faster_rmsquant
    from sglang.srt.layers.communicator import LayerCommunicator
    from sglang.srt.layers.layernorm import GemmaRMSNorm
    from sglang.srt.models.qwen3_5 import (
        Qwen3_5GatedDeltaNet,
        Qwen3_5LinearDecoderLayer,
    )

    K=5120; NQ=16384; NB=96
    _seen_norm=False; _seen_proj=False

    # Mark only linear-attention decoder communicators. AttentionDecoderLayer
    # instances are untouched and therefore keep ordinary BF16 input norm.
    _orig_linear_init=Qwen3_5LinearDecoderLayer.__init__
    def _linear_init(self,*args,**kwargs):
        _orig_linear_init(self,*args,**kwargs)
        self.layer_communicator.k100_q38_rms_gdn_m1=True
    Qwen3_5LinearDecoderLayer.__init__=_linear_init

    _orig_prepare_attn=LayerCommunicator.prepare_attn
    def _prepare_attn_rms_gdn(
        self,
        hidden_states,
        residual,
        forward_batch,
        quant_format="",
        post_residual_addition=None,
    ):
        global _seen_norm
        norm=self.input_layernorm
        if (
            getattr(self,"k100_q38_rms_gdn_m1",False)
            and self._context.tp_size==1 and self._context.attn_tp_size==1
            and post_residual_addition is None
            and isinstance(norm,GemmaRMSNorm)
            and isinstance(hidden_states,torch.Tensor)
            and hidden_states.ndim==2 and tuple(hidden_states.shape)==(1,K)
            and hidden_states.dtype is torch.bfloat16
            and (residual is None or (
                isinstance(residual,torch.Tensor)
                and residual.ndim==2 and tuple(residual.shape)==(1,K)
                and residual.dtype is torch.bfloat16
            ))
        ):
            if residual is None:
                # Stock semantics: residual becomes the pre-norm hidden input.
                new_residual=hidden_states
                q,s=lm_faster_rmsquant(
                    input=hidden_states,
                    rms_weight=norm.gemma_weight,
                    epsilon=norm.variance_epsilon,
                    quant_dtype=torch.int8,
                    residual=None,
                    update_input=False,
                )
            else:
                q,s=lm_faster_rmsquant(
                    input=hidden_states,
                    rms_weight=norm.gemma_weight,
                    epsilon=norm.variance_epsilon,
                    quant_dtype=torch.int8,
                    residual=residual,
                    update_input=True,
                )
                new_residual=residual
            if not _seen_norm:
                _seen_norm=True
                print("[K100 SGLang RMS->GDN INT8] ACTIVE M1 input Gemma RMS directly produces q/scale",flush=True)
            # carrier is a dtype/device/shape token only; the GDN projection
            # consumes q/s and does not read its numerical values.
            return (hidden_states,q,s),new_residual
        return _orig_prepare_attn(
            self,hidden_states,residual,forward_batch,
            quant_format=quant_format,
            post_residual_addition=post_residual_addition,
        )
    LayerCommunicator.prepare_attn=_prepare_attn_rms_gdn

    _prev_input_proj=Qwen3_5GatedDeltaNet._forward_input_proj
    def _input_proj_prequant(self,hidden_states):
        global _seen_proj
        if isinstance(hidden_states,tuple) and len(hidden_states)==3:
            carrier,q,s=hidden_states
            qkvz=self.in_proj_qkvz; ba=self.in_proj_ba
            if (
                isinstance(carrier,torch.Tensor) and tuple(carrier.shape)==(1,K)
                and isinstance(q,torch.Tensor) and tuple(q.shape)==(1,K) and q.dtype is torch.int8
                and isinstance(s,torch.Tensor) and tuple(s.shape)==(1,1)
                and getattr(qkvz,"weight",None) is not None
                and qkvz.weight.dtype is torch.int8
                and tuple(qkvz.weight.shape)==(K,NQ)
                and hasattr(qkvz,"weight_scale")
                and hasattr(ba,"k100_q38_ba_weight_int8_kn")
                and tuple(ba.k100_q38_ba_weight_int8_kn.shape)==(K,NB)
            ):
                out=torch.ops.k100_q38.sglang_gdn_qkvz_ba_fused_prequant_m1(
                    q,s,qkvz.weight,qkvz.weight_scale,
                    ba.k100_q38_ba_weight_int8_kn,ba.k100_q38_ba_weight_scale,
                )
                if not _seen_proj:
                    _seen_proj=True
                    print("[K100 SGLang RMS->GDN INT8] ACTIVE prequantized QKVZ+BA fused consumer; no second dynamic quant",flush=True)
                return out[:,:NQ],out[:,NQ:]
        return _prev_input_proj(self,hidden_states)
    Qwen3_5GatedDeltaNet._forward_input_proj=_input_proj_prequant

    print("[K100 SGLang RMS->GDN INT8] installed on 48 linear-attention layers; full-attention input norms stock",flush=True)
