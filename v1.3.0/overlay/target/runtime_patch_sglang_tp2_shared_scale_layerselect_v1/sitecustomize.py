"""TP2 production-v1 shared-scale transformer-layer selection bisection.

This diagnostic overlay keeps the validated production-v1 request/phase policy
(prefill + request-aware first decode) and both required RowParallel W8A8
families (Klocal=3072 and 8704), but allows shared dynamic token scale only on
an explicit subset of transformer layer ids. Other selected RowParallel calls
use the exact local-scale apply implementation saved by production-v1.

Environment:
  SGLANG_Q38_SHARED_LAYER_IDS=0-31,48,50-63

Fail closed:
  * layer selection is mandatory and must resolve to ids in [0, 63]
  * every target RowParallelLinear must have a parseable model.layers.<id> prefix
  * all non-target shapes keep production-v1 behavior unchanged

This is causal diagnostic code, not a delivery/champion patch.
"""
from __future__ import annotations

import os
import re
import runpy

_ROOT = "/data/qwen38-27b-k100ai-int8-opt"
_PARENT = f"{_ROOT}/runtime_patch_sglang_tp2_shared_scale_production_v1/sitecustomize.py"
_parent_ns = runpy.run_path(_PARENT, run_name="__q38_tp2_layerselect_parent__")

import torch
import sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 as _w8
from sglang.srt.layers.linear import RowParallelLinear

_TARGET_K = frozenset({3072, 8704})
_raw = os.environ.get("SGLANG_Q38_SHARED_LAYER_IDS", "").strip()
if not _raw:
    raise RuntimeError("TP2 layer-select fail-closed: SGLANG_Q38_SHARED_LAYER_IDS is required")


def _parse_layer_ids(spec: str) -> frozenset[int]:
    out: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            a, b = item.split("-", 1)
            lo, hi = int(a), int(b)
            if lo > hi:
                raise RuntimeError(f"TP2 layer-select bad range {item!r}")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(item))
    if not out or min(out) < 0 or max(out) > 63:
        raise RuntimeError(f"TP2 layer-select invalid ids={sorted(out)}")
    return frozenset(out)


_SHARED_LAYER_IDS = _parse_layer_ids(_raw)
_parent_apply = _w8.CompressedTensorsW8A8Int8.apply_weights
_local_apply = _parent_ns.get("_orig_apply_weights")
if _local_apply is None:
    raise RuntimeError("TP2 layer-select parent did not expose _orig_apply_weights")

# Model construction happens after sitecustomize. Preserve the full prefix on
# each RowParallelLinear so apply_weights can make a transformer-layer decision.
_orig_row_init = RowParallelLinear.__init__
_layer_re = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def _patched_row_init(self, *args, **kwargs):
    prefix = kwargs.get("prefix", None)
    if prefix is None and len(args) > 8:
        prefix = args[8]
    if prefix is None:
        prefix = ""
    _orig_row_init(self, *args, **kwargs)
    self._q38_prefix = str(prefix)
    m = _layer_re.search(str(prefix))
    self._q38_layer_id = int(m.group(1)) if m else None


RowParallelLinear.__init__ = _patched_row_init
_seen: set[tuple] = set()
_parent_globals = getattr(_parent_apply, "__globals__", {})


def _patched_apply_weights(self, layer, x, bias, input_quant_args=None, silu_quant_args=None):
    is_target = (
        isinstance(x, torch.Tensor)
        and x.ndim >= 2
        and layer.__class__.__name__ == "RowParallelLinear"
        and int(getattr(layer, "tp_size", 1)) > 1
        and int(x.shape[-1]) in _TARGET_K
    )
    if not is_target:
        return _parent_apply(
            self,
            layer,
            x,
            bias,
            input_quant_args=input_quant_args,
            silu_quant_args=silu_quant_args,
        )

    lid = getattr(layer, "_q38_layer_id", None)
    if lid is None:
        raise RuntimeError(
            "TP2 layer-select target RowParallelLinear has no layer id: "
            f"K={int(x.shape[-1])} prefix={getattr(layer, '_q38_prefix', None)!r}"
        )

    label = str(_parent_globals.get("_ctx_label", "none"))
    needs_shared = bool(
        _parent_globals.get("_ctx_full_shared", False)
        or _parent_globals.get("_ctx_decode_mask", None) is not None
    )
    use_shared = int(lid) in _SHARED_LAYER_IDS
    if needs_shared:
        key = (label, int(lid), int(x.shape[-1]), use_shared)
        if key not in _seen:
            _seen.add(key)
            print(
                "[K100 TP2 shared layer-select] "
                f"mode={label} layer={lid} Klocal={int(x.shape[-1])} "
                f"policy={'shared' if use_shared else 'local'}",
                flush=True,
            )

    if not use_shared:
        return _local_apply(
            self,
            layer,
            x,
            bias,
            input_quant_args=input_quant_args,
            silu_quant_args=silu_quant_args,
        )
    return _parent_apply(
        self,
        layer,
        x,
        bias,
        input_quant_args=input_quant_args,
        silu_quant_args=silu_quant_args,
    )


_w8.CompressedTensorsW8A8Int8.apply_weights = _patched_apply_weights
print(
    "[K100 TP2 shared layer-select v1] installed after production-v1; "
    f"shared_layers={sorted(_SHARED_LAYER_IDS)}; other target RowParallel layers use local scale",
    flush=True,
)
