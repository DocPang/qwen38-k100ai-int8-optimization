"""v30 short-Q compile bucket v2: MAX_Q/K=4096 + VARLEN BATCH_SIZE key fixed to 1.

Parent is paged-btstride-runtime-v2, which inherits the requestwise-q8/v29-RC3 chain.
This layer replaces only the exact short-Q one-pass consumer for cold batch1 q==kv. Runtime cu_seqlens and launch
grid remain exact; only Triton compile constants and LSE scratch stride are
bucketed to 4096 so arbitrary 2K..4K prompt lengths reuse one kernel.
"""
from __future__ import annotations
import importlib.util, os, runpy
from pathlib import Path

_DROOT='/data/qwen38-dflash2-k100ai'
_TROOT='/data/qwen38-27b-k100ai-int8-opt'
runpy.run_path(f'{_DROOT}/runtime_patch_dflash_tp4_paged_btstride_runtime_v2/sitecustomize.py', run_name='__v30_shortq_bucket_parent__')
try:
    _cmd=open('/proc/self/cmdline','rb').read().replace(b'\0',b' ').decode('utf-8','replace')
except Exception:
    _cmd=''
_helper=('multiprocessing.resource_tracker' in _cmd or '/usr/local/bin/ninja' in _cmd or ' ninja --version' in _cmd or ('cdll.LoadLibrary' in _cmd and 'torchinductor' in _cmd))
if not _helper:
    import torch
    from sglang.srt.layers.attention import pack_paged_kv_to_varlen as _packmod
    from sglang.srt.layers.attention import flashattention_backend as _fab

    _VENDOR=f'{_TROOT}/external_images/sglang0.5.12-20260620-src/flash_attn/flash_attn_triton_mqa_gqa.py'
    raw=Path(_VENDOR).read_text()
    reps=[
      ("            if start_m * BLOCK_M > seqlen_q:\n                return\n", "            if start_m * BLOCK_M > seqlen_q:\n                start_m = 0  # one-pass grid makes this unreachable\n", 'early-return'),
      ("            start_m = MAX_SEQLENS_Q // BLOCK_M -1 - tl.program_id(0)\n", "            start_m = (MAX_SEQLENS_Q + BLOCK_M - 1) // BLOCK_M - 1 - tl.program_id(0)\n", 'reverse-index'),
      ("    if IS_CAUSAL:\n        iter = 2\n    else:\n        iter = 1\n", "    if IS_CAUSAL:\n        iter = 1\n    else:\n        iter = 1\n", 'kernel-iter'),
      ("        if metadata.causal:\n            iter = 2\n        else:\n            iter = 1\n", "        if metadata.causal:\n            iter = 1\n        else:\n            iter = 1\n", 'wrapper-iter'),
      ("        M = torch.empty((batch, nheads_q, metadata.max_seqlens_q), device=q.device, dtype=torch.float32)\n", "        _q38_compile_bucket = 4096\n        M = torch.empty((batch, nheads_q, _q38_compile_bucket), device=q.device, dtype=torch.float32)\n", 'M-bucket'),
      ("            MAX_SEQLENS_Q=metadata.max_seqlens_q,\n            MAX_SEQLENS_K=metadata.max_seqlens_k,\n", "            MAX_SEQLENS_Q=_q38_compile_bucket,\n            MAX_SEQLENS_K=_q38_compile_bucket,\n", 'MAX-bucket'),
      ("            BATCH_SIZE= q.shape[0]\n", "            BATCH_SIZE=1  # VARLEN: q.shape[0] is total tokens, not batch; kernel never reads BATCH_SIZE\n", 'VARLEN-BATCH-SIZE-key'),
    ]
    fixed=raw
    for old,new,label in reps:
        if fixed.count(old)!=1: raise RuntimeError(f'shortq-bucket anchor {label} count={fixed.count(old)}')
        fixed=fixed.replace(old,new)
    tmp=Path(f'/tmp/q38_shortq_bucket4096_{os.getpid()}.py'); tmp.write_text(fixed)
    spec=importlib.util.spec_from_file_location(f'q38_shortq_bucket4096_{os.getpid()}',tmp)
    mod=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(mod)
    winner=[]
    for c in getattr(mod.attn_fwd,'configs',[]):
        kw=dict(getattr(c,'kwargs',{}) or {})
        if kw.get('BLOCK_M')==64 and kw.get('BLOCK_N')==64 and kw.get('waves_per_eu')==0 and kw.get('PRE_LOAD_V') is True and int(getattr(c,'num_stages',-1))==1 and int(getattr(c,'num_warps',-1))==4:
            winner.append(c)
    if len(winner)!=1: raise RuntimeError(f'shortq-bucket winner count={len(winner)}')
    mod.attn_fwd.configs=winner
    if hasattr(mod.attn_fwd,'cache'): mod.attn_fwd.cache.clear()
    if hasattr(mod.attn_fwd,'best_config'): mod.attn_fwd.best_config=None
    tri_fn=mod.flash_attn_varlen_func
    parent=_packmod.try_pack_paged_kv_to_varlen_attention
    hits=0
    fallback_layers={63}
    def route(*,q,forward_batch,metadata,layer,window_size:tuple,sinks,key_cache,value_cache,page_size:int,k_descale,v_descale,**kwargs):
        nonlocal_dummy=None
        global hits
        qlen=int(q.shape[0]); lid=int(getattr(layer,'layer_id',-1))
        if lid in fallback_layers or not (2048<=qlen<=4095) or int(getattr(forward_batch,'batch_size',-1))!=1:
            return parent(q=q,forward_batch=forward_batch,metadata=metadata,layer=layer,window_size=window_size,sinks=sinks,key_cache=key_cache,value_cache=value_cache,page_size=page_size,k_descale=k_descale,v_descale=v_descale,**kwargs)
        seq=getattr(forward_batch,'seq_lens_cpu',None)
        if seq is None or len(seq)<1:
            return parent(q=q,forward_batch=forward_batch,metadata=metadata,layer=layer,window_size=window_size,sinks=sinks,key_cache=key_cache,value_cache=value_cache,page_size=page_size,k_descale=k_descale,v_descale=v_descale,**kwargs)
        kvlen=int(seq[0])
        eligible=(kvlen==qlen and int(getattr(metadata,'max_seq_len_q',-1))==qlen and int(getattr(metadata,'max_seq_len_k',-1))==kvlen and q.numel()==qlen*6*256 and q.is_contiguous() and q.dtype==torch.bfloat16 and key_cache.dtype==torch.bfloat16 and value_cache.dtype==torch.bfloat16 and int(getattr(layer,'tp_q_head_num',-1))==6 and int(getattr(layer,'tp_k_head_num',-1))==1 and int(getattr(layer,'tp_v_head_num',-1))==1 and int(getattr(layer,'head_dim',-1))==256 and int(getattr(layer,'v_head_dim',-1))==256 and int(page_size)==64 and tuple(window_size)==(-1,-1) and sinks is None and abs(float(getattr(layer,'scaling',-1.0))-0.0625)<=1e-12)
        if not eligible or not _packmod.can_pack_paged_kv_to_varlen(forward_batch=forward_batch,metadata=metadata,layer=layer,window_size=window_size,sinks=sinks,key_cache=key_cache,value_cache=value_cache,page_size=page_size):
            return parent(q=q,forward_batch=forward_batch,metadata=metadata,layer=layer,window_size=window_size,sinks=sinks,key_cache=key_cache,value_cache=value_cache,page_size=page_size,k_descale=k_descale,v_descale=v_descale,**kwargs)
        pk,pv=_packmod.pack_paged_kv_to_varlen(key_cache.view(-1,layer.tp_k_head_num,page_size,layer.head_dim),value_cache.view(-1,layer.tp_v_head_num,layer.v_head_dim,page_size),metadata.page_table,forward_batch.seq_lens_cpu[:1],page_size)
        if int(pk.shape[0])!=kvlen or int(pv.shape[0])!=kvlen: raise RuntimeError(f'shortq-bucket gather mismatch q={qlen} kv={kvlen}')
        hits+=1
        if hits<=8 or hits%16==0: print(f'[K100 TP4 shortq-bucket4096-v2] ACTIVE hit={hits} q=kv={qlen}',flush=True)
        out=tri_fn(q.view(-1,layer.tp_q_head_num,layer.head_dim),pk,pv,metadata.cu_seqlens_q,metadata.cu_seqlens_k,qlen,kvlen,dropout_p=0.0,softmax_scale=layer.scaling,causal=True,return_attn_probs=False)
        return out[0] if isinstance(out,(tuple,list)) else out
    _packmod.try_pack_paged_kv_to_varlen_attention=route
    _fab.try_pack_paged_kv_to_varlen_attention=route
    print('[K100 TP4 shortq-bucket4096-v2] installed exact cold batch1 2048..4095; compile MAX_Q/K=4096; grid/cu_seqlens exact',flush=True)
