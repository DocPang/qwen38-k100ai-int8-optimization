"""N5: SGLang-native frozen compact true-M1 lm_head for Qwen3.8-27B/K100.

Mechanism ported, implementation rewritten at SGLang-native boundaries:
  dynamic full-hidden INT8 quant (lmslim) -> frozen uniform2048 feature gather
  -> full-vocab INT8 selector -> Top512 -> original BF16 row rerank
  -> dense sparse BF16 logits (-inf outside Top512).

This patch deliberately does NOT import vLLM runtime operators.  It chains the
accepted SGLang W8A8 compatibility + sparse tune-cache fallback and only
specializes LogitsProcessor's M=1 DECODE lm_head computation.  EXTEND/prefill,
logprob-ish metadata, TP/all-gather, FP32 head, bias, unsupported head shapes,
and non-M1 calls fall back to the untouched SGLang implementation.

The current full-model launcher is an experimental plain-greedy benchmark gate.
A broader production semantic guard (penalties/grammar/logit-bias/etc.) remains
mandatory before general deployment.
"""
from __future__ import annotations

import contextvars
import hashlib
import os
import runpy
import time

import torch
import triton
import triton.language as tl

_BASE = (
    "/data/qwen38-27b-k100ai-int8-opt/"
    "runtime_patch_sglang_n4_sparse_w8a8_cache/sitecustomize.py"
)
runpy.run_path(_BASE, run_name="__q38_sglang_sparse_w8a8_cache__")

if os.getenv("SGLANG_Q38_COMPACT_HEAD_M1", "0") == "1":
    from lmslim.layers.gemm.int8_utils import matmul_int8, per_token_quant_int8
    from sglang.srt.layers.logits_processor import LogitsProcessor
    from sglang.srt.models.qwen3_5 import Qwen3_5ForConditionalGeneration

    VOCAB = 248320
    K_FULL = 5120
    K_SELECTOR = 2048
    TOPK = int(os.getenv("SGLANG_Q38_COMPACT_HEAD_TOPK", "512"))
    if TOPK not in (512, 1024, 2048, 4096):
        raise RuntimeError(f"compact-head unsupported TOPK={TOPK}")
    FROZEN_INDEX_SHA = "1d1487248e97511306e6ba1192304f01ec44deee5930c1b992bf3c5e3d0330a2"
    SELECTOR_CONFIG = {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 512,
        "GROUP_SIZE_M": 8,
        "SPLIT_K": 1,
        "num_stages": 0,
        "num_warps": 8,
    }
    _FAST_CTX: contextvars.ContextVar[bool] = contextvars.ContextVar(
        "q38_sglang_compact_head_fast", default=False
    )
    _seen_fast = False

    def _frozen_indices(device=None):
        idx = (
            torch.linspace(0, K_FULL - 1, K_SELECTOR, dtype=torch.float64)
            .round()
            .to(torch.long)
            .unique(sorted=True)
        )
        if int(idx.numel()) != K_SELECTOR:
            raise RuntimeError(f"compact-head uniform2048 cardinality drift {idx.numel()}")
        digest = hashlib.sha256(idx.numpy().tobytes()).hexdigest()
        if digest != FROZEN_INDEX_SHA:
            raise RuntimeError(f"compact-head uniform2048 SHA drift {digest}")
        return idx.to(device).contiguous() if device is not None else idx.contiguous()

    def _build_selector(head):
        if hasattr(head, "k100_q38_selector_weight"):
            return
        w = getattr(head, "weight", None)
        if (
            not isinstance(w, torch.Tensor)
            or w.dtype is not torch.bfloat16
            or tuple(w.shape) != (VOCAB, K_FULL)
        ):
            raise RuntimeError(
                f"compact-head unexpected lm_head dtype={getattr(w,'dtype',None)} shape={getattr(w,'shape',None)}"
            )
        idx = _frozen_indices(w.device)
        q = torch.empty((VOCAB, K_SELECTOR), dtype=torch.int8, device=w.device)
        s = torch.empty((VOCAB, 1), dtype=torch.float32, device=w.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            for a in range(0, VOCAB, 2048):
                b = min(a + 2048, VOCAB)
                wf = w[a:b].float()
                scale = wf.abs().amax(1).clamp_min_(1e-12).div_(127.0)
                sub = torch.index_select(wf, 1, idx)
                qq = torch.round(sub / scale[:, None]).clamp_(-127, 127).to(torch.int8)
                q[a:b].copy_(qq)
                s[a:b, 0].copy_(scale)
        head.register_buffer("k100_q38_selector_idx", idx, persistent=False)
        head.register_buffer("k100_q38_selector_weight", q, persistent=False)
        head.register_buffer("k100_q38_selector_scale", s, persistent=False)
        torch.cuda.synchronize()
        print(
            "[K100 SGLang compact head] selector ready "
            f"shape={tuple(q.shape)} topk={TOPK} index_sha={FROZEN_INDEX_SHA} "
            f"bytes={q.numel()*q.element_size()+s.numel()*s.element_size()} "
            f"build_s={time.perf_counter()-t0:.3f}; BF16 head preserved",
            flush=True,
        )

    @triton.jit
    def _rowreduce_kernel(
        x_ptr,
        w_ptr,
        ids_ptr,
        out_ptr,
        SXM: tl.constexpr,
        SIM: tl.constexpr,
        SOM: tl.constexpr,
        M_: tl.constexpr,
        C_: tl.constexpr,
        K_: tl.constexpr,
        BK_: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row = pid // C_
        col = pid - row * C_
        valid = row < M_
        vid = tl.load(ids_ptr + row * SIM + col, mask=valid, other=0).to(tl.int64)
        offs = tl.arange(0, BK_)
        acc = 0.0
        for k0 in range(0, K_, BK_):
            kk = k0 + offs
            km = kk < K_
            x = tl.load(x_ptr + row * SXM + kk, mask=valid & km, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + vid * K_ + kk, mask=valid & km, other=0.0).to(tl.float32)
            acc += tl.sum(x * w, axis=0)
        tl.store(out_ptr + row * SOM + col, acc.to(tl.bfloat16), mask=valid)

    def _compact_dense(hidden_states, head):
        x = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
        if tuple(x.shape) != (1, K_FULL):
            raise RuntimeError(f"compact-head expected [1,{K_FULL}], got {tuple(x.shape)}")
        xq, xs = per_token_quant_int8(x)
        xsub = torch.index_select(xq, 1, head.k100_q38_selector_idx).contiguous()
        approx = matmul_int8(
            xsub,
            xs,
            head.k100_q38_selector_weight.t(),
            head.k100_q38_selector_scale,
            torch.bfloat16,
            SELECTOR_CONFIG,
        )
        ids = torch.topk(
            approx, TOPK, dim=-1, largest=True, sorted=False
        ).indices.contiguous()
        scores = torch.empty((1, TOPK), dtype=torch.bfloat16, device=x.device)
        _rowreduce_kernel[(TOPK,)](
            x,
            head.weight,
            ids,
            scores,
            x.stride(0),
            ids.stride(0),
            scores.stride(0),
            M_=1,
            C_=TOPK,
            K_=K_FULL,
            BK_=1024,
            num_warps=4,
            num_stages=1,
            waves_per_eu=1,
        )
        dense = torch.full(
            (1, VOCAB), -float("inf"), dtype=torch.bfloat16, device=x.device
        )
        dense.scatter_(1, ids, scores)
        return dense

    _orig_load_weights = Qwen3_5ForConditionalGeneration.load_weights

    def _load_weights_and_build(self, weights):
        result = _orig_load_weights(self, weights)
        head = getattr(self, "lm_head", None)
        if head is not None:
            _build_selector(head)
        return result

    Qwen3_5ForConditionalGeneration.load_weights = _load_weights_and_build

    _orig_get_logits = LogitsProcessor._get_logits
    _orig_compute_lm_head = LogitsProcessor._compute_lm_head

    def _metadata_safe(self, hidden_states, lm_head, logits_metadata, embedding_bias):
        try:
            if not logits_metadata.forward_mode.is_decode():
                return False
            if tuple(hidden_states.reshape(-1, hidden_states.shape[-1]).shape) != (1, K_FULL):
                return False
            if embedding_bias is not None:
                return False
            if self.use_fp32_lm_head or self.do_tensor_parallel_all_gather:
                return False
            if self.final_logit_softcapping is not None:
                return False
            if logits_metadata.extend_return_logprob:
                return False
            tops = logits_metadata.top_logprobs_nums
            if tops is not None and any(int(x) > 0 for x in tops):
                return False
            tids = logits_metadata.token_ids_logprobs
            if tids is not None and any(bool(x) for x in tids):
                return False
            if not hasattr(lm_head, "k100_q38_selector_weight"):
                return False
            if tuple(lm_head.weight.shape) != (VOCAB, K_FULL):
                return False
            return True
        except Exception:
            return False

    def _get_logits_wrapped(self, hidden_states, lm_head, logits_metadata, embedding_bias=None):
        fast = _metadata_safe(self, hidden_states, lm_head, logits_metadata, embedding_bias)
        tok = _FAST_CTX.set(fast)
        try:
            return _orig_get_logits(self, hidden_states, lm_head, logits_metadata, embedding_bias)
        finally:
            _FAST_CTX.reset(tok)

    def _compute_lm_head_wrapped(self, hidden_states, lm_head, embedding_bias=None):
        global _seen_fast
        if _FAST_CTX.get(False):
            if not _seen_fast:
                _seen_fast = True
                print(
                    f"[K100 SGLang compact head] ACTIVE M=1 DECODE Top{TOPK} dense-sparse; prefill/logprob-ish paths fallback",
                    flush=True,
                )
            return _compact_dense(hidden_states, lm_head)
        return _orig_compute_lm_head(self, hidden_states, lm_head, embedding_bias)

    LogitsProcessor._get_logits = _get_logits_wrapped
    LogitsProcessor._compute_lm_head = _compute_lm_head_wrapped
    print(
        "[K100 SGLang compact head] installed at LogitsProcessor; SGLang-native lmslim selector + BF16 rerank",
        flush=True,
    )
