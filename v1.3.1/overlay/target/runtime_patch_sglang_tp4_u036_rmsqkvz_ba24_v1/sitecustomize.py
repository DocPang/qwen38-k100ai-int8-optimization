"""TP4 corrected U036 + RMS->prequant QKVZ + TP4 BA24 INT8 native GEMV.

Helpers spawned by TorchInductor/resource_tracker bypass the model patch stack.
Model processes inherit rmsqkvz_v1 and additionally quantize only local GDN BA
weights [24,5120] once after loading.  During true-M1 tuple consumption, QKVZ
uses the exact prequantized path while BA reuses the same q/scale in generic-v2
native GEMV (waves=2, pipe=3).  All non-exact shapes fall back to parent.
"""
from __future__ import annotations
import importlib.util
import os
import runpy

_ROOT = "/data/qwen38-27b-k100ai-int8-opt"
try:
    _cmd = open("/proc/self/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "replace")
except Exception:
    _cmd = ""
_skip = False; _reason = ""
if "cdll.LoadLibrary" in _cmd and "torchinductor_tp4_gpu4567" in _cmd:
    _skip=True; _reason="torchinductor-so-loader"
elif "multiprocessing.resource_tracker" in _cmd:
    _skip=True; _reason="resource-tracker"
elif "/usr/local/bin/ninja" in _cmd or " ninja --version" in _cmd:
    _skip=True; _reason="ninja-helper"

if _skip:
    print(f"[K100 TP4 sitecustomize helper bypass] reason={_reason} pid={os.getpid()}", flush=True)
else:
    runpy.run_path(
        f"{_ROOT}/runtime_patch_sglang_tp4_u036_rmsqkvz_v1/sitecustomize.py",
        run_name="__q38_tp4_rmsqkvz_parent_for_ba24__",
    )
    if os.getenv("SGLANG_Q38_TP4_BA24_INT8_M1", "0") == "1":
        import torch
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
        from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet

        K=5120; NQ=4096; NB=24
        _SO=os.getenv("SGLANG_Q38_NATIVE_GDN_SPLIT_SO", f"{_ROOT}/native_ext/k100_int8_gemv_generic_v2_sglang.so")
        _spec=importlib.util.spec_from_file_location("k100_int8_gemv_generic_v2_sglang", _SO)
        if _spec is None or _spec.loader is None:
            raise RuntimeError(f"cannot load TP4 BA24 native GEMV: {_SO}")
        _gemv=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_gemv)

        _prev_process=UnquantizedLinearMethod.process_weights_after_loading
        _shadow_layers=0
        def _process_ba24_shadow(self, layer):
            global _shadow_layers
            _prev_process(self, layer)
            w=getattr(layer,"weight",None)
            if (
                not isinstance(w,torch.Tensor)
                or tuple(w.shape)!=(NB,K)
                or w.dtype not in (torch.bfloat16,torch.float16)
                or hasattr(layer,"k100_q38_tp4_ba24_int8_nk")
            ):
                return
            with torch.no_grad():
                w32=w.float(); ws=w32.abs().amax(1).clamp_min_(1e-12).div_(127.0)
                qw=torch.round(w32/ws[:,None]).clamp_(-127,127).to(torch.int8).contiguous()
            layer.register_buffer("k100_q38_tp4_ba24_int8_nk",qw,persistent=False)
            layer.register_buffer("k100_q38_tp4_ba24_scale",ws.float().contiguous(),persistent=False)
            _shadow_layers+=1
            if _shadow_layers in (1,48):
                print(f"[K100 SGLang TP4 BA24 INT8] shadow ready layers={_shadow_layers} shape=({NB},{K})",flush=True)

        UnquantizedLinearMethod.process_weights_after_loading=_process_ba24_shadow

        _prev_input=Qwen3_5GatedDeltaNet._forward_input_proj
        _hits=0
        def _input_proj_ba24(self, hidden_states):
            global _hits
            if isinstance(hidden_states,tuple) and len(hidden_states)==3:
                carrier,q,s=hidden_states
                qkvz=self.in_proj_qkvz; ba=self.in_proj_ba
                qw=getattr(qkvz,"weight",None); qs=getattr(qkvz,"weight_scale",None)
                bw=getattr(ba,"k100_q38_tp4_ba24_int8_nk",None); bs=getattr(ba,"k100_q38_tp4_ba24_scale",None)
                qm=getattr(qkvz,"quant_method",None)
                if (
                    isinstance(carrier,torch.Tensor) and carrier.dtype is torch.bfloat16 and tuple(carrier.shape)==(1,K)
                    and isinstance(q,torch.Tensor) and q.dtype is torch.int8 and tuple(q.shape)==(1,K) and q.is_contiguous()
                    and isinstance(s,torch.Tensor) and s.dtype is torch.float32 and tuple(s.shape)==(1,1)
                    and qm is not None and isinstance(qw,torch.Tensor) and qw.dtype is torch.int8 and tuple(qw.shape)==(K,NQ)
                    and isinstance(qs,torch.Tensor) and qs.dtype is torch.float32 and qs.numel()==NQ
                    and isinstance(bw,torch.Tensor) and bw.dtype is torch.int8 and tuple(bw.shape)==(NB,K) and bw.is_contiguous()
                    and isinstance(bs,torch.Tensor) and bs.dtype is torch.float32 and bs.numel()==NB
                ):
                    q_out=qm.apply(qkvz,carrier,None,input_quant_args=[q,s])
                    ba_out=_gemv.gemv(q,bw,s.reshape(-1),bs.reshape(-1),2,3)
                    _hits+=1
                    if _hits<=4 or _hits in (16,32,64,128):
                        print(f"[K100 SGLang TP4 BA24 INT8] ACTIVE hit={_hits} shared-q native(w2,p3)",flush=True)
                    return q_out,ba_out
            return _prev_input(self,hidden_states)
        Qwen3_5GatedDeltaNet._forward_input_proj=_input_proj_ba24
        print("[K100 SGLang TP4 BA24 INT8] installed; exact TP4 M1 tuple only, relaxed BA weight shadow",flush=True)
