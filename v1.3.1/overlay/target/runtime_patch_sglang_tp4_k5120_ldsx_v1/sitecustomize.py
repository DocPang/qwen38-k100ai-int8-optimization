"""TP4-V1: extend accepted U022 K5120 LDS-x to TP4 rank-local output shapes.

Parent: U022/PFX1 stack. Changes only M=1 BF16-output W8A8 matmuls whose
input K remains 5120 under TP4 but whose column-parallel N is quartered:
  gate_up      K5120 -> N8704
  GDN QKVZ     K5120 -> N4096
  full QKV     K5120 -> N3584
All other shapes fall through to U022/stock. Isolated GPU4 gates on 2026-08-19
proved eager + CUDAGraph bitwise equality for all three shapes; waves=2 won.
"""
from __future__ import annotations
import importlib.util
import os
import runpy
import torch

_BASE=("/data/qwen38-27b-k100ai-int8-opt/"
      "runtime_patch_sglang_u022_k5120_ldsx/sitecustomize.py")
runpy.run_path(_BASE,run_name="__q38_sglang_u022_parent_for_tp4__")

if os.getenv("SGLANG_Q38_TP4_K5120_LDSX_M1","0")=="1":
    from lmslim import quant_ops as _lmslim_quant_ops
    _SO=os.getenv("SGLANG_Q38_K5120_LDSX_SO","/data/qwen38-27b-k100ai-int8-opt/native_ext/k100_int8_gemv_k5120_ldsx_v1_sglang.so")
    _spec=importlib.util.spec_from_file_location("k100_int8_gemv_k5120_ldsx_v1_sglang",_SO)
    if _spec is None or _spec.loader is None: raise RuntimeError(f"cannot load TP4 K5120 LDS-x: {_SO}")
    _gemv=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_gemv)
    _prev=_lmslim_quant_ops.triton_scaled_mm
    _SHAPES={(5120,8704):(2,"gate_up_tp4"),(5120,4096):(2,"gdn_qkvz_tp4"),(5120,3584):(2,"full_qkv_tp4")}
    _seen=set()
    def _tp4_k5120_ldsx_scaled_mm(a,b,scale_a,scale_b,out_dtype,bias=None,best_config=None):
        if (bias is None and out_dtype is torch.bfloat16 and isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor)
            and a.dtype is torch.int8 and b.dtype is torch.int8 and a.ndim==2 and b.ndim==2 and a.shape[0]==1
            and a.is_contiguous() and b.t().is_contiguous() and scale_a.dtype is torch.float32 and scale_b.dtype is torch.float32
            and scale_a.numel()>=1 and int(b.shape[0])==int(a.shape[1])):
            k=int(a.shape[1]);n=int(b.shape[1]);cfg=_SHAPES.get((k,n))
            if cfg is not None and scale_b.numel()==n:
                waves,label=cfg;key=(k,n)
                if key not in _seen:
                    _seen.add(key);print(f"[K100 SGLang TP4 K5120 LDS-x] ACTIVE {label} M1 K{k}->N{n}; waves={waves}",flush=True)
                return _gemv.gemv(a,b.t(),scale_a.reshape(-1),scale_b.reshape(-1),waves)
        return _prev(a,b,scale_a,scale_b,out_dtype,bias,best_config)
    _lmslim_quant_ops.triton_scaled_mm=_tp4_k5120_ldsx_scaled_mm
    print(f"[K100 SGLang TP4 K5120 LDS-x] installed from {_SO}; rank-local shapes only; all others inherit U022/stock",flush=True)
