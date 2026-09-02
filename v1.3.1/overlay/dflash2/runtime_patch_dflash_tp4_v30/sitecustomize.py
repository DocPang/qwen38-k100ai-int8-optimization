"""Qwen3.8 TP4 v30 final candidate.

Parent chain:
  shortq-bucket4096-v2
    -> v30 candidate / paged-btstride-runtime-v2
    -> requestwise-q8 bs2..8
    -> full v29-RC3 chain.

v30 changes only short cold batch1 prefill compilation behavior:
1) <2K paged-varlen attention uses runtime block-table stride and disables
   value/alignment specialization of that stride.
2) 2K..4095 historical short-Q one-pass keeps exact runtime cu_seqlens/grid,
   but compiles MAX_Q/K at a shared 4096 bucket and fixes an unused VARLEN
   BATCH_SIZE constexpr key to 1 (vendor used q.shape[0] == total tokens).
3) SGLang pre-listen warmup compiles representative 1024/2048/2230 shapes,
   then flushes KV/Radix state before HTTP starts listening.

No >=4096 model math or routing is changed here.
"""
from __future__ import annotations

import os
import runpy

_ROOT = "/data/qwen38-dflash2-k100ai"
runpy.run_path(
    f"{_ROOT}/runtime_patch_dflash_tp4_shortq_bucket4096_v2/sitecustomize.py",
    run_name="__tp4_v30_final_parent__",
)

try:
    _cmd = open("/proc/self/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "replace")
except Exception:
    _cmd = ""
_helper = (
    "multiprocessing.resource_tracker" in _cmd
    or "/usr/local/bin/ninja" in _cmd
    or " ninja --version" in _cmd
    or ("cdll.LoadLibrary" in _cmd and "torchinductor" in _cmd)
)

if not _helper:
    from sglang.srt.entrypoints.warmup import warmup as _warmup
    from sglang.srt.managers.io_struct import GenerateReqInput

    @_warmup("q38_v30_shortctx")
    async def _q38_v30_shortctx_warmup(disaggregation_mode, tokenizer_manager):
        # 1024 compiles the <2K runtime-BT paged path. 2048 compiles the
        # shared short-Q 4096 bucket. 2230 pays the remaining common odd-size
        # alignment/helper specializations observed after the first bucket
        # compile. All requests are internal and cache is flushed at the end.
        sizes = (1024, 2048, 2230)
        print(f"[K100 TP4 v30] pre-listen shortctx warmup begin sizes={list(sizes)}", flush=True)
        for size in sizes:
            input_ids = [1000 + ((i * 17 + size) % 127) for i in range(size)]
            req = GenerateReqInput(
                input_ids=input_ids,
                sampling_params={
                    "max_new_tokens": 1,
                    "temperature": 0.0,
                    "ignore_eos": True,
                },
            )
            async for _ in tokenizer_manager.generate_request(req, None):
                pass
            print(f"[K100 TP4 v30] pre-listen shortctx warmup shape complete q={size}", flush=True)
        ret = await tokenizer_manager.flush_cache(timeout_s=30.0)
        if not bool(getattr(ret, "success", False)):
            raise RuntimeError(f"v30 shortctx warmup flush_cache failed: {ret!r}")
        print("[K100 TP4 v30] pre-listen shortctx warmup complete; cache flushed", flush=True)

    print("[K100 TP4 v30] final candidate installed; warmup=q38_v30_shortctx", flush=True)
