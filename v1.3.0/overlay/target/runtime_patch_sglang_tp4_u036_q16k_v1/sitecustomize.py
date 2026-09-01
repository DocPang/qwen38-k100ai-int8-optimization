"""TP4 q16K long-prefill attention addon for the corrected U036 stack.

This addon MUST be loaded after the existing TP4 BA24/U036 target stack. It
intercepts only batch1 q=16384, QH6/KVH1, D256, BF16, page64, full-causal
packed-varlen prefill. q=8192 remains owned by the existing U036 parent.

For KV>=64K an exact real-arithmetic 4-way KV split is used. The first three
segments are fully visible; only the last segment is bottom-right causal. The
four segment outputs are merged with the log-sum-exp identity in FP32. For
shorter audited KV lengths the same frozen BM64/BN64/w4/preloadV kernel runs
unsplit. All other shapes fall back to the already-correct parent.
"""
from __future__ import annotations
import importlib.util, os
from pathlib import Path
import torch, triton
from sglang.srt.layers.attention import pack_paged_kv_to_varlen as _packmod
from sglang.srt.layers.attention import flashattention_backend as _fabackend

_ROOT="/data/qwen38-27b-k100ai-int8-opt"
_SRC=f"{_ROOT}/external_images/sglang0.5.12-20260620-src/flash_attn/flash_attn_triton_mqa_gqa.py"
_ALLOWED={16384,32768,49152,65536,81920,98304,114688,131072,147456,163840,180224,196608,212992,229376,245760}
_raw=os.environ.get("SGLANG_Q38_TP4_Q16_KV_LENGTHS","").strip()
if not _raw:
    raise RuntimeError("TP4-q16 fail-closed: SGLANG_Q38_TP4_Q16_KV_LENGTHS must be explicit")
try: _KV={int(x.strip()) for x in _raw.split(',') if x.strip()}
except ValueError as exc: raise RuntimeError(f"TP4-q16 bad KV list: {_raw!r}") from exc
if not _KV or not _KV.issubset(_ALLOWED):
    raise RuntimeError(f"TP4-q16 fail-closed requested={sorted(_KV)} allowed={sorted(_ALLOWED)}")
_SPLIT=int(os.environ.get("SGLANG_Q38_TP4_Q16_SPLIT_KV","4"))
if _SPLIT not in (1,2,4): raise RuntimeError("TP4-q16 SPLIT_KV must be 1, 2 or 4")
_QSPLIT2=os.environ.get("SGLANG_Q38_TP4_Q16_QSPLIT2","0").strip().lower() in ("1","true","yes","on")
_QSPLIT_LAYER_RAW=os.environ.get("SGLANG_Q38_TP4_Q16_QSPLIT_LAYER_IDS","").strip()
_FULL_ATTN_LAYER_IDS=set(range(3,64,4))
if _QSPLIT_LAYER_RAW:
    try:
        _QSPLIT_LAYER_IDS={int(x.strip()) for x in _QSPLIT_LAYER_RAW.split(',') if x.strip()}
    except ValueError as exc:
        raise RuntimeError(f"TP4-q16 invalid QSPLIT layer list: {_QSPLIT_LAYER_RAW!r}") from exc
    if not _QSPLIT_LAYER_IDS or not _QSPLIT_LAYER_IDS.issubset(_FULL_ATTN_LAYER_IDS):
        raise RuntimeError(
            f"TP4-q16 fail-closed qsplit layers={sorted(_QSPLIT_LAYER_IDS)} "
            f"allowed={sorted(_FULL_ATTN_LAYER_IDS)}"
        )
else:
    _QSPLIT_LAYER_IDS=set(_FULL_ATTN_LAYER_IDS) if _QSPLIT2 else set()
_QSPLIT_KV_MIN=int(os.environ.get("SGLANG_Q38_TP4_Q16_QSPLIT_KV_MIN","0") or 0)
if _QSPLIT_KV_MIN and _QSPLIT_KV_MIN not in _ALLOWED:
    raise RuntimeError(f"TP4-q16 qsplit KV minimum must be 0 or one audited KV length, got {_QSPLIT_KV_MIN}")
_QSPLIT_KV_EXACT_RAW=os.environ.get("SGLANG_Q38_TP4_Q16_QSPLIT_KV_EXACT","").strip()
try:
    _QSPLIT_KV_EXACT={int(x.strip()) for x in _QSPLIT_KV_EXACT_RAW.split(',') if x.strip()}
except ValueError as exc:
    raise RuntimeError(f"TP4-q16 invalid exact qsplit KV list: {_QSPLIT_KV_EXACT_RAW!r}") from exc
if _QSPLIT_KV_EXACT and not _QSPLIT_KV_EXACT.issubset(_ALLOWED):
    raise RuntimeError(f"TP4-q16 fail-closed exact qsplit KVs={sorted(_QSPLIT_KV_EXACT)} allowed={sorted(_ALLOWED)}")
if _QSPLIT_KV_MIN and _QSPLIT_KV_EXACT:
    raise RuntimeError("TP4-q16 qsplit KV policy is ambiguous: use minimum or exact list, not both")
_TAIL_SPLIT_8K=os.environ.get("SGLANG_Q38_TP4_TAIL_SPLIT_8K","0").strip().lower() in ("1","true","yes","on")
_TAIL_SPLIT_MIN_PREFIX=int(os.environ.get("SGLANG_Q38_TP4_TAIL_SPLIT_MIN_PREFIX","131072") or 131072)
_LONG_CHUNK_8K_PREFIX=int(os.environ.get("SGLANG_Q38_TP4_LONG_CHUNK_8K_PREFIX","0") or 0)
if _LONG_CHUNK_8K_PREFIX and _LONG_CHUNK_8K_PREFIX % 8192:
    raise RuntimeError(f"TP4-q16 long q8K chunk switch prefix must be an 8192 multiple, got {_LONG_CHUNK_8K_PREFIX}")
_QTAIL_SPLIT_KV=int(os.environ.get("SGLANG_Q38_TP4_QTAIL_SPLIT_KV","0") or 0)
if _QTAIL_SPLIT_KV not in (0,4,8):
    raise RuntimeError(f"TP4-q16 qtail split must be 0, 4 or 8, got {_QTAIL_SPLIT_KV}")
_QTAIL_KV_RAW=os.environ.get("SGLANG_Q38_TP4_QTAIL_KV_LENGTHS","").strip()
try:
    _QTAIL_KV={int(x.strip()) for x in _QTAIL_KV_RAW.split(',') if x.strip()}
except ValueError as exc:
    raise RuntimeError(f"TP4-q16 invalid qtail KV list: {_QTAIL_KV_RAW!r}") from exc
if _QTAIL_SPLIT_KV and _QTAIL_KV != {257900}:
    raise RuntimeError(f"TP4-q16 qtail fail-closed: only exact KV=257900 is audited, got {sorted(_QTAIL_KV)}")

_parent=_packmod.try_pack_paged_kv_to_varlen_attention
os.environ["FLASH_ATTENTION_PRINT_PARAM"]="0"
raw=Path(_SRC).read_text()
# SourceFind's causal two-pass mapping assumes q_len is an exact BLOCK_M
# multiple.  For the real 257.9K tail q=3948, floor(Q/BM)-1-pid both
# duplicates one block and drops the final partial block, causing duplicate
# writers / nondeterministic hidden states.  Use ceil(Q/BM)-1-pid instead.
# This is algebraically identical for the audited q=8192/16384 paths.
old_start="""            start_m = MAX_SEQLENS_Q // BLOCK_M -1 - tl.program_id(0)\n"""
new_start="""            start_m = (MAX_SEQLENS_Q + BLOCK_M - 1) // BLOCK_M - 1 - tl.program_id(0)\n"""
old_guard="""            if start_m * BLOCK_M > seqlen_q:\n                return\n"""
new_guard="""            if start_m * BLOCK_M > seqlen_q:\n                start_m = 0  # unreachable with corrected admitted grids; avoids Triton loop-return issue\n"""
if raw.count(old_start)!=1 or raw.count(old_guard)!=1:
    raise RuntimeError("TP4-q16 vendor Triton causal mapping changed; fail closed")
raw=raw.replace(old_start,new_start).replace(old_guard,new_guard)
tmp=f"/tmp/q38_tp4_q16_flash_attn_{os.getpid()}.py"; Path(tmp).write_text(raw)
spec=importlib.util.spec_from_file_location(f"q38_tp4_q16_triton_{os.getpid()}",tmp); tri=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(tri)
wins=[]
for c in getattr(tri.attn_fwd,'configs',[]):
    kw=dict(getattr(c,'kwargs',{}) or {})
    if kw.get('BLOCK_M')==64 and kw.get('BLOCK_N')==64 and kw.get('waves_per_eu')==0 and kw.get('PRE_LOAD_V') is True and int(getattr(c,'num_stages',-1))==1 and int(getattr(c,'num_warps',-1))==4:
        wins.append(c)
if len(wins)!=1: raise RuntimeError(f"TP4-q16 expected one BM64/BN64/w4/preloadV config, got {len(wins)}")
tri.attn_fwd.configs=wins
if hasattr(tri.attn_fwd,'cache'): tri.attn_fwd.cache.clear()
if hasattr(tri.attn_fwd,'best_config'): tri.attn_fwd.best_config=None
tri_fn=tri.flash_attn_varlen_func
_hits={k:0 for k in sorted(_KV)}


def _split_attention(q,pk,pv,layer,ns:int):
    Q=int(q.shape[0]); K=int(pk.shape[0])
    if K<65536 or K//ns<Q: return None
    qv=q.view(-1,layer.tp_q_head_num,layer.head_dim); qh=int(layer.tp_q_head_num); kh=int(layer.tp_k_head_num); d=int(layer.head_dim)
    bounds=[K*i//ns for i in range(ns+1)]; outs=[]; lses=[]
    cuq=torch.tensor([0,Q],device=q.device,dtype=torch.int32)
    for i in range(ns):
        ks=pk[bounds[i]:bounds[i+1]]; vs=pv[bounds[i]:bounds[i+1]]; kl=int(ks.shape[0]); cuk=torch.tensor([0,kl],device=q.device,dtype=torch.int32)
        o=torch.empty_like(qv,dtype=vs.dtype); m=torch.empty((1,qh,Q),device=q.device,dtype=torch.float32)
        qs=(0,qv.stride(1),qv.stride(0),qv.stride(2)); kst=(0,ks.stride(1),ks.stride(0),ks.stride(2)); vst=(0,vs.stride(1),vs.stride(0),vs.stride(2)); ost=(0,o.stride(1),o.stride(0),o.stride(2)); zeros=(0,0,0,0); az=(0,0)
        causal=i==ns-1; it=2 if causal else 1; grid=lambda META:(triton.cdiv(Q,META['BLOCK_M']*it),qh,1)
        tri.attn_fwd[grid](qv,ks,vs,None,layer.scaling,m,o,*qs,*kst,*vst,*ost,*zeros,*az,cuq,cuk,dropout_p=0.0,philox_seed=0x1BF52,philox_offset_base=0x1D4B42,encoded_softmax=None,alibi_slopes=None,HQ=qh,HK=kh,ACTUAL_BLOCK_DMODEL=d,MAX_SEQLENS_Q=Q,MAX_SEQLENS_K=kl,VARLEN=True,IS_CAUSAL=causal,BLOCK_DMODEL=256,BIAS_TYPE=0,ENABLE_DROPOUT=False,RETURN_ENCODED_SOFTMAX=False,USE_ALIBI=False,BATCH_SIZE=Q)
        outs.append(o); lses.append(m[0].transpose(0,1).contiguous())
    mf=torch.stack(lses,0); mm=mf.max(0).values; w=torch.exp2(mf-mm.unsqueeze(0)); of=torch.stack([x.float() for x in outs],0)
    return ((of*w.unsqueeze(-1)).sum(0)/w.sum(0).unsqueeze(-1)).to(torch.bfloat16)


def _direct_attention(q,pk,pv,layer):
    Q=int(q.shape[0]); K=int(pk.shape[0])
    qv=q.view(-1,layer.tp_q_head_num,layer.head_dim)
    cuq=torch.tensor([0,Q],device=q.device,dtype=torch.int32)
    cuk=torch.tensor([0,K],device=q.device,dtype=torch.int32)
    out=tri_fn(qv,pk,pv,cuq,cuk,Q,K,dropout_p=0.0,softmax_scale=layer.scaling,causal=True,return_attn_probs=False)
    return out[0] if isinstance(out,(tuple,list)) else out


def _qsplit2_attention(q,pk,pv,layer):
    """Scheduler-equivalent 2x8K full-attention inside one 16K target forward.

    The first half sees prefix+8K KV and the second half sees prefix+16K KV.
    Each 8K subcall reproduces the current q8 U036 policy: split4 only at
    KV>=64K, otherwise the same frozen direct BM64/BN64/w4/preloadV kernel.
    """
    Q=int(q.shape[0]); K=int(pk.shape[0])
    if Q!=16384 or K<Q: return None
    q0=q[:8192]; q1=q[8192:]
    k0=K-8192
    if k0<8192: return None
    def q8_call(qh,kh,vh):
        kk=int(kh.shape[0])
        if kk>=65536:
            out=_split_attention(qh,kh,vh,layer,4)
            if out is None: raise RuntimeError(f"TP4-q16 qsplit2 q8 split4 rejected kv={kk}")
            return out
        return _direct_attention(qh,kh,vh,layer)
    y0=q8_call(q0,pk[:k0],pv[:k0])
    y1=q8_call(q1,pk,pv)
    return torch.cat((y0,y1),dim=0)


def _common_geometry(*,q,forward_batch,metadata,layer,window_size,sinks,key_cache,value_cache,page_size):
    if int(getattr(forward_batch,'batch_size',-1))!=1: return False,None
    sl=getattr(forward_batch,'seq_lens_cpu',None)
    if sl is None or len(sl)<1: return False,None
    kv=int(sl[0])
    if int(page_size)!=64 or int(getattr(layer,'tp_q_head_num',-1))!=6 or int(getattr(layer,'tp_k_head_num',-1))!=1 or int(getattr(layer,'tp_v_head_num',-1))!=1: return False,None
    if int(getattr(layer,'head_dim',-1))!=256 or int(getattr(layer,'v_head_dim',-1))!=256: return False,None
    if q.dtype!=torch.bfloat16 or key_cache.dtype!=torch.bfloat16 or value_cache.dtype!=torch.bfloat16: return False,None
    if tuple(window_size)!=(-1,-1) or sinks is not None or abs(float(getattr(layer,'scaling',-1.0))-0.0625)>1e-12: return False,None
    if not _packmod.can_pack_paged_kv_to_varlen(forward_batch=forward_batch,metadata=metadata,layer=layer,window_size=window_size,sinks=sinks,key_cache=key_cache,value_cache=value_cache,page_size=page_size): return False,None
    return True,kv


def _eligible(*,q,forward_batch,metadata,layer,window_size,sinks,key_cache,value_cache,page_size):
    ok,kv=_common_geometry(q=q,forward_batch=forward_batch,metadata=metadata,layer=layer,window_size=window_size,sinks=sinks,key_cache=key_cache,value_cache=value_cache,page_size=page_size)
    if not ok or int(q.shape[0])!=16384: return False,None
    if q.numel()!=16384*6*256 or not q.is_contiguous() or int(getattr(metadata,'max_seq_len_q',-1))!=16384: return False,None
    if kv not in _KV or int(getattr(metadata,'max_seq_len_k',-1))!=kv: return False,None
    return True,kv


_qtail_probe_hits=0
_qtail_active_hits=0

def _try_qtail(*,q,forward_batch,metadata,layer,window_size,sinks,key_cache,value_cache,page_size):
    global _qtail_probe_hits,_qtail_active_hits
    qn=int(q.shape[0]); maxq=int(getattr(metadata,'max_seq_len_q',-1))
    if not _QTAIL_SPLIT_KV or qn not in (3948,3968): return None,False
    ok,kv=_common_geometry(q=q,forward_batch=forward_batch,metadata=metadata,layer=layer,window_size=window_size,sinks=sinks,key_cache=key_cache,value_cache=value_cache,page_size=page_size)
    if not ok: return None,False
    maxk=int(getattr(metadata,'max_seq_len_k',-1))
    _qtail_probe_hits+=1
    if _qtail_probe_hits<=8:
        print(f"[K100 SGLang TP4 qtail-probe] candidate q={qn} maxq={maxq} seqkv={kv} maxk={maxk} contiguous={q.is_contiguous()}",flush=True)
    if qn!=maxq or kv not in _QTAIL_KV or maxk!=kv or not q.is_contiguous() or q.numel()!=qn*6*256:
        return None,False
    pk,pv=_packmod.pack_paged_kv_to_varlen(key_cache.view(-1,layer.tp_k_head_num,page_size,layer.head_dim),value_cache.view(-1,layer.tp_v_head_num,layer.v_head_dim,page_size),metadata.page_table,forward_batch.seq_lens_cpu[:forward_batch.batch_size],page_size)
    if int(pk.shape[0])!=kv or int(pv.shape[0])!=kv:
        raise RuntimeError(f"TP4-qtail gather mismatch q={qn} kv={kv} K={tuple(pk.shape)} V={tuple(pv.shape)}")
    out=_split_attention(q,pk,pv,layer,_QTAIL_SPLIT_KV)
    if out is None: raise RuntimeError(f"TP4-qtail split{_QTAIL_SPLIT_KV} rejected admitted q={qn} kv={kv}")
    _qtail_active_hits+=1
    if _qtail_active_hits<=4 or _qtail_active_hits%16==0:
        print(f"[K100 SGLang TP4 qtail] ACTIVE layer={int(getattr(layer,'layer_id',-1))} q={qn} kv={kv} split={_QTAIL_SPLIT_KV} hit={_qtail_active_hits}",flush=True)
    return out,True


def _q16_try(*,q,forward_batch,metadata,layer,window_size,sinks,key_cache,value_cache,page_size,k_descale,v_descale,**kwargs):
    tail_out,tail_hit=_try_qtail(q=q,forward_batch=forward_batch,metadata=metadata,layer=layer,window_size=window_size,sinks=sinks,key_cache=key_cache,value_cache=value_cache,page_size=page_size)
    if tail_hit:
        return tail_out
    ok,kv=_eligible(q=q,forward_batch=forward_batch,metadata=metadata,layer=layer,window_size=window_size,sinks=sinks,key_cache=key_cache,value_cache=value_cache,page_size=page_size)
    if not ok:
        return _parent(q=q,forward_batch=forward_batch,metadata=metadata,layer=layer,window_size=window_size,sinks=sinks,key_cache=key_cache,value_cache=value_cache,page_size=page_size,k_descale=k_descale,v_descale=v_descale,**kwargs)
    pk,pv=_packmod.pack_paged_kv_to_varlen(key_cache.view(-1,layer.tp_k_head_num,page_size,layer.head_dim),value_cache.view(-1,layer.tp_v_head_num,layer.v_head_dim,page_size),metadata.page_table,forward_batch.seq_lens_cpu[:forward_batch.batch_size],page_size)
    if int(pk.shape[0])!=kv or int(pv.shape[0])!=kv: raise RuntimeError(f"TP4-q16 gather mismatch kv={kv} K={tuple(pk.shape)} V={tuple(pv.shape)}")
    _hits[kv]+=1; h=_hits[kv]
    layer_id=int(getattr(layer,'layer_id',-1))
    qsplit_kv_ok=(
        kv in _QSPLIT_KV_EXACT
        if _QSPLIT_KV_EXACT
        else (_QSPLIT_KV_MIN == 0 or kv >= _QSPLIT_KV_MIN)
    )
    use_qsplit=(
        _QSPLIT2
        and layer_id in _QSPLIT_LAYER_IDS
        and qsplit_kv_ok
    )
    mode="qsplit2" if use_qsplit else f"kvsplit{_SPLIT if kv>=65536 else 1}"
    if h<=4 or h%16==0: print(f"[K100 SGLang TP4 q16K] ACTIVE layer={layer_id} kv={kv} hit={h} q=16384 qh=6 kvh=1 mode={mode}",flush=True)
    if use_qsplit:
        out=_qsplit2_attention(q,pk,pv,layer)
        if out is None: raise RuntimeError(f"TP4-q16 qsplit2 rejected admitted kv={kv}")
        return out
    if _SPLIT in (2,4) and kv>=65536:
        out=_split_attention(q,pk,pv,layer,_SPLIT)
        if out is None: raise RuntimeError(f"TP4-q16 split{_SPLIT} rejected admitted kv={kv}")
        return out
    return _direct_attention(q,pk,pv,layer)

_packmod.try_pack_paged_kv_to_varlen_attention=_q16_try
_fabackend.try_pack_paged_kv_to_varlen_attention=_q16_try

if _TAIL_SPLIT_8K or _LONG_CHUNK_8K_PREFIX:
    from sglang.srt.managers.schedule_policy import PrefillAdder as _PrefillAdder
    _orig_add_chunked_req=_PrefillAdder.add_chunked_req
    _chunk8_seen=False
    def _long_chunk8_add_chunked_req(self,req):
        global _chunk8_seen
        rem=self.rem_chunk_tokens
        tail=int(getattr(req,'extend_input_len',0) or 0)
        prefix_len=len(getattr(req,'prefix_indices',()))
        long_switch=(
            _LONG_CHUNK_8K_PREFIX > 0
            and prefix_len >= _LONG_CHUNK_8K_PREFIX
            and tail > 8192
        )
        final_tail_split=(
            _TAIL_SPLIT_8K
            and prefix_len >= _TAIL_SPLIT_MIN_PREFIX
            and 8192 < tail < 16384
        )
        if rem is not None and int(rem)==16384 and (long_switch or final_tail_split):
            saved=int(rem)
            self.rem_chunk_tokens=8192
            ret=_orig_add_chunked_req(self,req)
            after=int(self.rem_chunk_tokens or 0)
            consumed=8192-after
            if consumed!=8192:
                raise RuntimeError(f"TP4 long-chunk8 budget invariant failed: consumed={consumed} after={after}")
            self.rem_chunk_tokens=saved-consumed
            if not _chunk8_seen:
                _chunk8_seen=True
                print(
                    f"[K100 SGLang TP4 long-chunk8] ACTIVE prefix={prefix_len} tail={tail} "
                    f"switch_prefix={_LONG_CHUNK_8K_PREFIX} -> 8192+{tail-8192}",
                    flush=True,
                )
            return ret
        return _orig_add_chunked_req(self,req)
    _PrefillAdder.add_chunked_req=_long_chunk8_add_chunked_req

print(f"[K100 SGLang TP4 q16K] installed kv={sorted(_KV)} split={_SPLIT} qsplit2={_QSPLIT2} qsplit_layers={sorted(_QSPLIT_LAYER_IDS)} qsplit_kv_min={_QSPLIT_KV_MIN} qsplit_kv_exact={sorted(_QSPLIT_KV_EXACT)} tail_split_8k={_TAIL_SPLIT_8K} long_chunk_8k_prefix={_LONG_CHUNK_8K_PREFIX} qtail_split={_QTAIL_SPLIT_KV} qtail_kv={sorted(_QTAIL_KV)}; q8K parent preserved",flush=True)
