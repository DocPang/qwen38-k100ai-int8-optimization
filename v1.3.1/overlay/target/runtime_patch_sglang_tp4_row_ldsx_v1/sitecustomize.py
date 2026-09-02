"""TP4 row-parallel native LDS-x candidate on top of TP4 compact-head stack.

Intercepts only true-M1 W8A8 row-parallel rank-local shapes proven bitwise
exact in isolated eager+CUDAGraph gates on K100AI:
  down family         K4352 -> N5120, waves=2 (stable multi-rank choice)
  attn/GDN output     K1536 -> N5120, waves=4
All other shapes inherit the TP4 compact-head parent unchanged.
"""
from __future__ import annotations
import importlib.util
import os
import runpy
import torch

_BASE=("/data/qwen38-27b-k100ai-int8-opt/"
      "runtime_patch_sglang_tp4_compact_head_v1/sitecustomize.py")
runpy.run_path(_BASE,run_name="__q38_sglang_tp4_compact_parent__")

if os.getenv("SGLANG_Q38_TP4_ROW_LDSX_M1","0")=="1":
    from lmslim import quant_ops as _lmslim_quant_ops
    _SO=os.getenv("SGLANG_Q38_TP4_ROW_LDSX_SO","/data/qwen38-27b-k100ai-int8-opt/native_ext/k100_int8_gemv_tp4_row_ldsx_v1_sglang.so")
    _spec=importlib.util.spec_from_file_location("k100_int8_gemv_tp4_row_ldsx_v1_sglang",_SO)
    if _spec is None or _spec.loader is None: raise RuntimeError(f"cannot load TP4 row LDS-x: {_SO}")
    _gemv=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_gemv)
    _prev=_lmslim_quant_ops.triton_scaled_mm
    _SHAPES={(4352,5120):(2,"down_tp4"),(1536,5120):(4,"out_tp4")}
    _seen=set()
    def _tp4_row_ldsx_scaled_mm(a,b,scale_a,scale_b,out_dtype,bias=None,best_config=None):
        if (bias is None and out_dtype is torch.bfloat16 and isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor)
            and a.dtype is torch.int8 and b.dtype is torch.int8 and a.ndim==2 and b.ndim==2 and a.shape[0]==1
            and a.is_contiguous() and b.t().is_contiguous() and scale_a.dtype is torch.float32 and scale_b.dtype is torch.float32
            and scale_a.numel()>=1 and int(b.shape[0])==int(a.shape[1])):
            k=int(a.shape[1]); n=int(b.shape[1]); cfg=_SHAPES.get((k,n))
            if cfg is not None and scale_b.numel()==n:
                waves,label=cfg; key=(k,n)
                if key not in _seen:
                    _seen.add(key); print(f"[K100 SGLang TP4 row LDS-x] ACTIVE {label} M1 K{k}->N{n}; waves={waves}",flush=True)
                return _gemv.gemv(a,b.t(),scale_a.reshape(-1),scale_b.reshape(-1),waves)
        return _prev(a,b,scale_a,scale_b,out_dtype,bias,best_config)
    _lmslim_quant_ops.triton_scaled_mm=_tp4_row_ldsx_scaled_mm
    print(f"[K100 SGLang TP4 row LDS-x] installed from {_SO}; exact rank-local shapes only",flush=True)
