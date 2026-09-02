"""TP4 DFlash abort lifecycle fix v8.

True source-level fix for real Hermes disconnect/abort churn.

Old SGLang behavior marks a running request with FINISH_ABORT, lets one final
forward complete, then routes it through the normal finished-request cleanup.
That cleanup calls release_kv_cache(..., is_insert=True), which can insert an
aborted speculative/paged/Mamba request's transitional req->KV mapping into the
Radix cache.  Under DFlash this produced physical-slot aliasing across Radix
branches and eventually duplicate slots / pool-checker crashes.

Fix: FINISH_ABORT requests are never inserted into Radix.  They preserve their
already-protected prefix and release all request-owned KV/Mamba tail state via
the runtime's existing is_insert=False cleanup.  Normal EOS/length/stop
finishes are untouched.

No checker suppression, orphan reconciliation, cache flush, attention/sampling
change, or hot-path decode change is included.  Strict pool checks remain the
authority.
"""
from __future__ import annotations

import os
import runpy
import sys

_ROOT = "/data/qwen38-dflash2-k100ai"
if os.getenv("SGLANG_Q38_TP124_V30_SKIP_TP4_PARENT", "0") != "1":
    runpy.run_path(
        f"{_ROOT}/runtime_patch_dflash_tp4_retained_alloc_fix_v3/sitecustomize.py",
        run_name="__tp4_abort_nocache_v8_parent__",
    )

from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.mem_cache import common as _common

_parent_release = _common.release_kv_cache
_events = 0


def _release_kv_cache_abort_safe(req, tree_cache, is_insert: bool = True):
    global _events
    abort = isinstance(getattr(req, "finished_reason", None), FINISH_ABORT)
    effective_insert = bool(is_insert) and not abort
    if abort and is_insert:
        _events += 1
        print(
            "[K100 DFlash TP4 abort-nocache-v8] ACTIVE "
            f"event={_events} rid={getattr(req, 'rid', '?')} "
            f"protected={int(getattr(req, 'cache_protected_len', -1))} "
            f"committed={int(getattr(req, 'kv_committed_len', -1))} "
            f"allocated={int(getattr(req, 'kv_allocated_len', -1))}",
            flush=True,
        )
    return _parent_release(req, tree_cache, is_insert=effective_insert)


# Patch the authority module first.
_common.release_kv_cache = _release_kv_cache_abort_safe

# Old SGLang imports release_kv_cache into several manager modules with
# `from ...common import release_kv_cache`, so update those bound aliases too.
# Importing these modules is safe at sitecustomize time after the parent stack
# has already imported ScheduleBatch / DFlash and the scheduler dependencies.
for _name in (
    "sglang.srt.managers.scheduler_output_processor_mixin",
    "sglang.srt.managers.scheduler",
):
    try:
        if _name in sys.modules:
            _mod = sys.modules[_name]
        else:
            _mod = __import__(_name, fromlist=["*"])
        if hasattr(_mod, "release_kv_cache"):
            setattr(_mod, "release_kv_cache", _release_kv_cache_abort_safe)
    except Exception as _exc:
        raise RuntimeError(f"abort-nocache-v8 failed to patch {_name}: {_exc}") from _exc

print(
    "[K100 DFlash TP4 abort-nocache-v8] installed: FINISH_ABORT skips Radix insert; strict checker unchanged",
    flush=True,
)
