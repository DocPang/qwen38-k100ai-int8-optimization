"""Topology-neutral Qwen3.8 DFlash2 v30 release layer for TP1 and TP2.

The profile wrapper must load its already-validated topology parent before this
file.  This layer then composes the v30 runtime semantics that do not depend on
TP4 QH6/KVH1 math:

* safe multi-request req->KV writeback and retained-page ownership;
* strict orphan-page reclaim, disconnect/abort no-cache insertion, and
  verify-only stock paged allocation;
* ABI-correct native int64 prepare assignment for DFlash width8 batches;
* grammar/tool-required, per-request seed, min-p, qwen3_coder streaming string
  arguments, and an 8192-token default OpenAI completion cap;
* profile-specific pre-listen short-context compilation followed by a strict
  cache flush.

TP1 raw-q8 and every TP4-only attention/native GEMV specialization are
deliberately excluded.  Those remain owned by the topology parent.
"""
from __future__ import annotations

import os
import runpy


_DROOT = "/data/qwen38-dflash2-k100ai"
_profile = os.getenv("SGLANG_Q38_V30_PROFILE", "").strip().lower()
if _profile not in {"tp1", "tp2"}:
    raise RuntimeError(
        "TP124 v30 common requires SGLANG_Q38_V30_PROFILE=tp1 or tp2, "
        f"got {_profile!r}"
    )

try:
    _cmd = open("/proc/self/cmdline", "rb").read().replace(b"\0", b" ").decode(
        "utf-8", "replace"
    )
except Exception:
    _cmd = ""
_helper = (
    "multiprocessing.resource_tracker" in _cmd
    or "/usr/local/bin/ninja" in _cmd
    or " ninja --version" in _cmd
    or ("cdll.LoadLibrary" in _cmd and "torchinductor" in _cmd)
)

if _helper:
    print(f"[K100 DFlash v30 {_profile}] common helper bypass pid={os.getpid()}", flush=True)
else:
    # These files retain their original TP4 parent behavior unless this exact
    # opt-in is set.  Sequential execution lets TP1/TP2 reuse the audited deltas
    # without importing the TP4 attention/math parent chain.
    os.environ["SGLANG_Q38_TP124_V30_SKIP_TP4_PARENT"] = "1"
    _layers = (
        ("runtime_patch_dflash_tp4_assign_verify_fix_v2/sitecustomize.py", "assign_verify"),
        ("runtime_patch_dflash_tp4_retained_alloc_fix_v3/sitecustomize.py", "retained_alloc"),
        ("runtime_patch_dflash_tp4_page_reclaim_fix_v4/sitecustomize.py", "page_reclaim"),
        ("runtime_patch_dflash_tp4_abort_nocache_v8/sitecustomize.py", "abort_nocache"),
        ("runtime_patch_dflash_tp4_verify_stockalloc_v9/sitecustomize.py", "verify_stockalloc"),
        ("runtime_patch_dflash_tp4_prepare_native_i64_v29/sitecustomize.py", "prepare_native_i64"),
        ("runtime_patch_dflash_tp4_v29_rc2/sitecustomize.py", "release_features"),
        ("runtime_patch_dflash_tp4_v29_rc3/sitecustomize.py", "client_guards"),
    )
    for _relative, _label in _layers:
        _path = f"{_DROOT}/{_relative}"
        if not os.path.isfile(_path):
            raise RuntimeError(f"TP124 v30 missing common layer {_label}: {_path}")
        runpy.run_path(_path, run_name=f"__q38_v30_{_profile}_{_label}__")

    from sglang.srt.entrypoints.warmup import warmup as _warmup
    from sglang.srt.managers.io_struct import GenerateReqInput

    _defaults = {
        "tp1": "1024,2048,4274,7433",
        "tp2": "1024,2048,3001,7419,14953",
    }
    _raw_sizes = os.getenv("SGLANG_Q38_V30_WARMUP_SIZES", _defaults[_profile])
    try:
        _sizes = tuple(int(value.strip()) for value in _raw_sizes.split(",") if value.strip())
    except ValueError as exc:
        raise RuntimeError(f"invalid SGLANG_Q38_V30_WARMUP_SIZES={_raw_sizes!r}") from exc
    if not _sizes or any(size < 64 or size > 20000 for size in _sizes):
        raise RuntimeError(f"invalid {_profile} v30 warmup sizes: {_sizes!r}")

    _warmup_name = f"q38_v30_{_profile}_shortctx"

    @_warmup(_warmup_name)
    async def _q38_v30_profile_warmup(disaggregation_mode, tokenizer_manager):
        print(
            f"[K100 DFlash v30 {_profile}] pre-listen warmup begin sizes={list(_sizes)}",
            flush=True,
        )
        for size in _sizes:
            input_ids = [1000 + ((index * 17 + size) % 127) for index in range(size)]
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
            print(
                f"[K100 DFlash v30 {_profile}] pre-listen warmup shape complete q={size}",
                flush=True,
            )
        ret = await tokenizer_manager.flush_cache(timeout_s=30.0)
        if not bool(getattr(ret, "success", False)):
            raise RuntimeError(f"v30 {_profile} warmup flush_cache failed: {ret!r}")
        print(
            f"[K100 DFlash v30 {_profile}] pre-listen warmup complete; cache flushed",
            flush=True,
        )

    print(
        f"[K100 DFlash v30 {_profile}] topology-neutral release layer installed; "
        f"warmup={_warmup_name}; TP4-only math excluded",
        flush=True,
    )
