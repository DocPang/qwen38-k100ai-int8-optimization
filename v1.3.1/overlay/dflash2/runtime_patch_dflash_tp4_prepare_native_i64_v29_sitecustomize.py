"""TP4 DFlash prepare fixed-width assignment with an ABI-correct HIP kernel.

The SourceFind native assignment symbol declares its start/end offsets as
``int const*``, while real DFlash prepare passes int64 tensors.  v27 snapshot
replay shows the mismatch skips request row 2 and writes its values into row 3.

For the only affected hot shape (prepare, batch>1, draft width=8), this patch
uses a one-block/one-wave HIP kernel that reads the existing int64 tensors
directly.  It avoids both the invalid ABI call and v28's two temporary int32
allocations.  Verify writeback and every other path remain inherited from v9.
"""

from __future__ import annotations

import importlib.util
import os
import runpy

import torch


_ROOT = "/data/qwen38-dflash2-k100ai"
runpy.run_path(
    f"{_ROOT}/runtime_patch_dflash_tp4_verify_stockalloc_v9/sitecustomize.py",
    run_name="__tp4_prepare_native_i64_v29_parent__",
)

from sglang.srt.speculative import dflash_info as _di


_SO = os.getenv(
    "SGLANG_Q38_ASSIGN_FIXED8_I64_SO",
    f"{_ROOT}/native_ext/k100_assign_fixed8_i64_v1.so",
)
_spec = importlib.util.spec_from_file_location("k100_assign_fixed8_i64_v1", _SO)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load fixed8 int64 assignment extension: {_SO}")
_native = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_native)

_parent_assign = _di.assign_req_to_token_pool_func
_parent_verify = _di.DFlashVerifyInput.verify
_verify_depth = 0
_hits = 0
_audit_limit = 8


def _verify_scoped(self, *args, **kwargs):
    global _verify_depth
    _verify_depth += 1
    try:
        return _parent_verify(self, *args, **kwargs)
    finally:
        _verify_depth -= 1


def _assign_prepare_native_i64(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
    batch_size: int,
):
    global _hits
    bs = int(batch_size)
    if (
        _verify_depth == 0
        and bs > 1
        and int(out_cache_loc.numel()) == bs * 8
        and req_pool_indices.dtype is torch.int64
        and req_to_token.dtype is torch.int32
        and start_offset.dtype is torch.int64
        and out_cache_loc.dtype is torch.int64
        and req_pool_indices.is_contiguous()
        and req_to_token.is_contiguous()
        and start_offset.is_contiguous()
        and out_cache_loc.is_contiguous()
    ):
        _native.assign_fixed8_i64(
            req_pool_indices,
            req_to_token,
            start_offset,
            out_cache_loc,
            int(req_to_token.shape[1]),
            bs,
        )
        _hits += 1

        if _hits <= _audit_limit:
            torch.cuda.synchronize()
            rows = out_cache_loc.view(bs, 8)
            bad = []
            for index in range(bs):
                req_idx = int(req_pool_indices[index].item())
                start = int(start_offset[index].item())
                expected = rows[index].detach().to("cpu", dtype=torch.int64).tolist()
                actual = (
                    req_to_token[req_idx, start : start + 8]
                    .detach()
                    .to("cpu", dtype=torch.int64)
                    .tolist()
                )
                if expected != actual:
                    bad.append(
                        {
                            "i": index,
                            "req_idx": req_idx,
                            "start": start,
                            "expected": expected,
                            "actual": actual,
                        }
                    )
            if bad:
                raise RuntimeError(
                    "TP4 prepare-native-i64-v29 audit mismatch: " + repr(bad)
                )
            print(
                f"[K100 TP4 prepare-native-i64-v29] AUDIT_OK "
                f"hit={_hits} bs={bs}",
                flush=True,
            )
        elif _hits % 128 == 0:
            print(
                f"[K100 TP4 prepare-native-i64-v29] ACTIVE "
                f"hit={_hits} bs={bs}",
                flush=True,
            )
        return None

    return _parent_assign(
        req_pool_indices,
        req_to_token,
        start_offset,
        end_offset,
        out_cache_loc,
        batch_size,
    )


_di.assign_req_to_token_pool_func = _assign_prepare_native_i64
_di.DFlashVerifyInput.verify = _verify_scoped

print(
    f"[K100 TP4 prepare-native-i64-v29] installed from {_SO}: batch>1 "
    "DFlash prepare width8 uses one-wave native int64-offset assignment",
    flush=True,
)
