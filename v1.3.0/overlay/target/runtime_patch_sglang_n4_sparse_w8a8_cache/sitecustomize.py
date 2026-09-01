"""SGLang 0.5.12 sparse W8A8 tune-cache fallback for K100/gfx928.

The SourceFind W8A8 dispatcher treats the existence of an M=1 entry for a
weight shape as proof that every rounded runtime M key exists, then indexes the
cache with ``dict[key]``.  A deliberately curated M=1-only cache therefore
crashes during SGLang's 128-token general warmup with KeyError instead of
falling back to the stock Triton config.

This patch preserves the accepted Qwen3.8 compressed-tensors ignore semantics
and changes only cache-miss behavior: a missing exact M/N/K tune-cache key
returns ``None``, which is already the documented/implemented input for the
stock lmslim ``matmul_int8`` fallback configuration. Existing cache hits remain
byte-for-byte the same dict objects/configs.
"""
from __future__ import annotations

import runpy

_BASELINE = (
    "/data/qwen38-27b-k100ai-int8-opt/"
    "runtime_patch_sglang_w8a8_compat/sitecustomize.py"
)
runpy.run_path(_BASELINE, run_name="__q38_sglang_w8a8_compat__")

from sglang.srt.layers.quantization.compressed_tensors.schemes import (  # noqa: E402
    compressed_tensors_w8a8_int8 as _w8,
)


class _SparseTuneDict(dict):
    _logged_missing: set[str] = set()

    def __missing__(self, key):
        # ``best_config=None`` is the existing lmslim fallback contract.
        if key not in self._logged_missing:
            print(
                f"[K100 SGLang sparse W8A8 cache] miss -> stock fallback: {key}",
                flush=True,
            )
            self._logged_missing.add(key)
        return None


_old = _w8.W8A8_TRITONJSON.triton_json_dict
if not isinstance(_old, _SparseTuneDict):
    _w8.W8A8_TRITONJSON.triton_json_dict = _SparseTuneDict(_old)

print(
    "[K100 SGLang sparse W8A8 cache] installed: exact cache hit unchanged; missing M/N/K -> best_config=None",
    flush=True,
)
