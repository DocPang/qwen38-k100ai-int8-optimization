"""Minimal SGLang compatibility fix for Qwen3.8 SmoothQuant W8A8.

SourceFind SGLang 0.5.12 uses substring matching for compressed-tensors ignore
rules. Qwen3.8's checkpoint ignores the parent linear_attn module plus selected
BF16 children (in_proj_a/in_proj_b) while quantizing qkv/z/out children. Substring
matching incorrectly disables quantization for the quantized children.

SGLang also renames `model.language_model.*` to `model.*` and `model.visual.*`
to `visual.*`. Normalize those prefixes, then restore upstream-style exact/regex
matching. No model math or kernel path is changed here.
"""
from __future__ import annotations
from typing import Iterable
from sglang.srt.layers.quantization.compressed_tensors import utils as _u


def _normalize(name: str) -> str:
    if name.startswith("model.language_model."):
        return "model." + name[len("model.language_model."):]
    if name.startswith("model.visual."):
        return "visual." + name[len("model.visual."):]
    return name


def _exact_match(layer_name: str, targets: Iterable[str]) -> bool:
    layer_name = _normalize(layer_name)
    for target in targets:
        if target.startswith("re:"):
            if _u._is_equal_or_regex_match(layer_name, target):
                return True
        elif _u._is_equal_or_regex_match(layer_name, _normalize(target)):
            return True
    return False


_u.check_equal_or_regex_match = _exact_match
print("[K100 SGLang W8A8] normalized exact ignore semantics active", flush=True)
