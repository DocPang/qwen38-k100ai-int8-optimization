"""N6: exact M=1 SwiGLU -> INT8 producer fusion on the accepted SGLang speed-first stack.

Chain: N5 compact head + fused GDN QKVZ/BA, then specialize Qwen2MLP M=1:
  gate_up W8A8 -> fused exact BF16-semantics SiLU*Mul + dynamic INT8 quant
  -> down_proj consumes prequantized (q, scale), avoiding BF16 activation
  materialization followed by a separate per-token quant kernel.

Non-M1, TP>1, unsupported shapes/quant schemes, bias, or missing weights fall
back to stock Qwen2MLP.forward. The producer is bitwise identical to the current
SGLang stock BF16 SiLU*Mul + per_token_quant_int8 gate on real M1 data.
"""
from __future__ import annotations
import os, runpy
import torch
import triton
import triton.language as tl

_BASE=("/data/qwen38-27b-k100ai-int8-opt/"
      "runtime_patch_sglang_n5_compact_head_gdnint8/sitecustomize.py")
runpy.run_path(_BASE, run_name="__q38_sglang_n5_compact_gdn__")

if os.getenv("SGLANG_Q38_SWIGLU_INT8_M1", "0") == "1":
    from sglang.srt.models.qwen2_moe import Qwen2MoeMLP

    D=17408; K=5120; BA=512; BQ=256
    NP=triton.cdiv(D,BA); NQ=triton.cdiv(D,BQ)
    _seen=False

    @triton.jit
    def _partial_amax_rows(x_ptr, partial_ptr, DD:tl.constexpr, NPP:tl.constexpr, BS:tl.constexpr):
        pid=tl.program_id(0); row=pid//NPP; part=pid-row*NPP
        offs=part*BS+tl.arange(0,BS); mask=offs<DD; base=row*(2*DD)
        gate=tl.load(x_ptr+base+offs,mask=mask,other=0.0).to(tl.float32)
        up=tl.load(x_ptr+base+DD+offs,mask=mask,other=0.0).to(tl.float32)
        y=((gate*tl.sigmoid(gate))*up).to(tl.bfloat16)
        tl.store(partial_ptr+row*NPP+part,tl.max(tl.abs(y.to(tl.float32)),axis=0))

    @triton.jit
    def _quant_rows_exact(x_ptr,q_ptr,scale_ptr,partial_ptr,DD:tl.constexpr,NPP:tl.constexpr,NQQ:tl.constexpr,BS:tl.constexpr):
        pid=tl.program_id(0); row=pid//NQQ; qpart=pid-row*NQQ
        poffs=tl.arange(0,128)
        partial=tl.load(partial_ptr+row*NPP+poffs,mask=poffs<NPP,other=0.0)
        amax=tl.max(partial,axis=0); scale=amax/127.0
        inv=tl.where(amax>0.0,127.0/amax,0.0)
        if qpart==0: tl.store(scale_ptr+row,scale)
        offs=qpart*BS+tl.arange(0,BS); mask=offs<DD; base=row*(2*DD)
        gate=tl.load(x_ptr+base+offs,mask=mask,other=0.0).to(tl.float32)
        up=tl.load(x_ptr+base+DD+offs,mask=mask,other=0.0).to(tl.float32)
        y=((gate*tl.sigmoid(gate))*up).to(tl.bfloat16).to(tl.float32)
        value=y*inv; av=tl.abs(value); floored=tl.floor(av); frac=av-floored
        fi=floored.to(tl.int32)
        inc=(frac>0.5)|((frac==0.5)&((fi&1)!=0))
        rounded=floored+inc.to(tl.float32); rounded=tl.where(value<0.0,-rounded,rounded)
        tl.store(q_ptr+row*DD+offs,rounded.to(tl.int8),mask=mask)

    @torch.library.custom_op("k100_q38::sglang_swiglu_int8_m1_exact",mutates_args=(),device_types="cuda")
    def _swiglu_int8_m1_exact(gate_up:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
        x=gate_up.contiguous()
        if tuple(x.shape)!=(1,2*D):
            raise RuntimeError(f"SwiGLU exact M1 shape drift: {tuple(x.shape)}")
        partial=torch.empty((1,NP),device=x.device,dtype=torch.float32)
        q=torch.empty((1,D),device=x.device,dtype=torch.int8)
        scale=torch.empty((1,1),device=x.device,dtype=torch.float32)
        _partial_amax_rows[(NP,)](x,partial,DD=D,NPP=NP,BS=BA,num_warps=4)
        _quant_rows_exact[(NQ,)](x,q,scale,partial,DD=D,NPP=NP,NQQ=NQ,BS=BQ,num_warps=4)
        return q,scale

    @_swiglu_int8_m1_exact.register_fake
    def _fake(gate_up:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
        return (gate_up.new_empty((1,D),dtype=torch.int8),gate_up.new_empty((1,1),dtype=torch.float32))

    _orig=Qwen2MoeMLP.forward
    def _forward(self,x:torch.Tensor,should_allreduce_fusion:bool=False,use_reduce_scatter:bool=False):
        global _seen
        down=self.down_proj
        if (
            x.ndim==2 and tuple(x.shape)==(1,K)
            and int(getattr(down,"tp_size",1))==1
            and getattr(down,"bias",None) is None
            and getattr(down,"quant_method",None) is not None
            and getattr(down,"weight",None) is not None
            and down.weight.dtype is torch.int8
            and tuple(down.weight.shape)==(D,K)
        ):
            gate_up,_=self.gate_up_proj(x)
            if tuple(gate_up.shape)==(1,2*D):
                q,s=torch.ops.k100_q38.sglang_swiglu_int8_m1_exact(gate_up)
                if not _seen:
                    _seen=True
                    print("[K100 SGLang SwiGLU->INT8] ACTIVE exact M1 producer; down consumes prequantized q/scale",flush=True)
                # CompressedTensorsLinearMethod passes silu_quant_args to W8A8 scheme.
                out=down.quant_method.apply(down,gate_up,None,silu_quant_args=[q,s])
                return out
        return _orig(self,x,should_allreduce_fusion,use_reduce_scatter)
    Qwen2MoeMLP.forward=_forward
    print("[K100 SGLang SwiGLU->INT8] installed exact M1; non-M1 stock fallback",flush=True)
