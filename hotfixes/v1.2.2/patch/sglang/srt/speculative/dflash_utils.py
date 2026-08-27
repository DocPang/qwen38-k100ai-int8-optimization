from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.utils import is_cuda, is_musa

DEFAULT_DFLASH_MASK_TOKEN = "<|MASK|>"

_DFLASH_SAMPLING_VERIFY_AVAILABLE = False
_DFLASH_CHAIN_VERIFY_BUFFERS: dict[tuple[Optional[int], int], dict[str, Any]] = {}
_DFLASH_VERIFY_SKIP_CUSTOM_MASK_BACKENDS = frozenset(
    {
        "FlashInferAttnBackend",
        "FlashInferMLAAttnBackend",
        "FlashAttentionBackend",
        "TRTLLMHAAttnBackend",
        "TRTLLMMLABackend",
    }
)


if is_cuda() or is_musa():
    try:
        from sgl_kernel import (
            top_k_renorm_prob,
            top_p_renorm_prob,
            tree_speculative_sampling_target_only,
        )

        _DFLASH_SAMPLING_VERIFY_AVAILABLE = True
    except Exception:
        top_k_renorm_prob = None
        top_p_renorm_prob = None
        tree_speculative_sampling_target_only = None
else:
    top_k_renorm_prob = None
    top_p_renorm_prob = None
    tree_speculative_sampling_target_only = None


def is_dflash_sampling_verify_available() -> bool:
    return _DFLASH_SAMPLING_VERIFY_AVAILABLE


def scale_kv_cell_size_per_token_for_dflash(
    *,
    target_cell_size_per_token: int,
    target_num_layers: int,
    draft_num_layers: int,
    draft_cell_size_per_token: Optional[int] = None,
) -> int:
    """Compute bytes/token budget for combined target+draft KV pools (DFLASH).

    DFLASH runs a separate draft runner with its own KV pool. The target runner's
    token capacity must fit both pools in aggregate.

    Returns:
        Approximate per-token bytes for (target KV + draft KV), expressed as a
        scaled version of `target_cell_size_per_token`, unless an explicit
        `draft_cell_size_per_token` is provided (in which case we sum them).
    """
    if target_cell_size_per_token <= 0:
        raise ValueError(
            "target_cell_size_per_token must be positive, "
            f"got {target_cell_size_per_token}."
        )

    if draft_cell_size_per_token is not None:
        draft_cell_size_per_token = int(draft_cell_size_per_token)
        if draft_cell_size_per_token <= 0:
            raise ValueError(
                "draft_cell_size_per_token must be positive when provided, "
                f"got {draft_cell_size_per_token}."
            )
        return int(target_cell_size_per_token) + int(draft_cell_size_per_token)

    if target_num_layers <= 0 or draft_num_layers <= 0:
        return int(target_cell_size_per_token)

    total_layers = int(target_num_layers) + int(draft_num_layers)
    return (
        int(target_cell_size_per_token) * int(total_layers) + int(target_num_layers) - 1
    ) // int(target_num_layers)


def resolve_dflash_verify_mask_policy(attn_backend: Any) -> tuple[str, bool]:
    backend = attn_backend
    for _ in range(4):
        full_backend = getattr(backend, "full_attn_backend", None)
        if full_backend is None:
            break
        backend = full_backend
    backend_name = type(backend).__name__
    return backend_name, (backend_name not in _DFLASH_VERIFY_SKIP_CUSTOM_MASK_BACKENDS)


def _get_or_create_chain_verify_buffers(
    *,
    bs: int,
    draft_token_num: int,
    device: torch.device,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    key = (device.index, int(draft_token_num))
    cached = _DFLASH_CHAIN_VERIFY_BUFFERS.get(key)
    cap_bs = 0 if cached is None else int(cached["cap_bs"])
    if cap_bs < bs:
        new_cap = max(int(bs), cap_bs * 2 if cap_bs > 0 else int(bs))
        retrieve_index = torch.arange(
            new_cap * draft_token_num, dtype=torch.int64, device=device
        ).view(new_cap, draft_token_num)
        row_next = torch.arange(
            1, draft_token_num + 1, dtype=torch.int64, device=device
        )
        row_next[-1] = -1
        retrieve_next_token = row_next.unsqueeze(0).expand(new_cap, -1).clone()
        retrieve_next_sibling = torch.full(
            (new_cap, draft_token_num), -1, dtype=torch.int64, device=device
        )
        predicts = torch.empty(
            (new_cap * draft_token_num,), dtype=torch.int32, device=device
        )
        accept_index = torch.empty(
            (new_cap, draft_token_num), dtype=torch.int32, device=device
        )
        accept_token_num = torch.empty((new_cap,), dtype=torch.int32, device=device)
        cached = {
            "cap_bs": int(new_cap),
            "retrieve_index": retrieve_index,
            "retrieve_next_token": retrieve_next_token,
            "retrieve_next_sibling": retrieve_next_sibling,
            "predicts": predicts,
            "accept_index": accept_index,
            "accept_token_num": accept_token_num,
        }
        _DFLASH_CHAIN_VERIFY_BUFFERS[key] = cached

    assert cached is not None
    retrieve_index = cached["retrieve_index"][:bs]
    retrieve_next_token = cached["retrieve_next_token"][:bs]
    retrieve_next_sibling = cached["retrieve_next_sibling"][:bs]
    predicts = cached["predicts"][: bs * draft_token_num]
    accept_index = cached["accept_index"][:bs]
    accept_token_num = cached["accept_token_num"][:bs]
    return (
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        predicts,
        accept_index,
        accept_token_num,
    )


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> List[int]:
    """Select target layer indices used to build DFlash context features.

    Args:
        num_target_layers: Number of transformer layers in the runtime target model.
        num_draft_layers: Number of layers in the DFlash draft model.

    Returns:
        A list of 0-based target layer indices of length `num_draft_layers`.

    Notes:
        - DFlash uses hidden states after each selected target layer (HF-style).
        - SGLang captures "before layer i", so the model hook will typically add +1
          when mapping to capture points.
    """
    if num_target_layers <= 0:
        raise ValueError(
            f"num_target_layers must be positive, got {num_target_layers}."
        )
    if num_draft_layers <= 0:
        raise ValueError(f"num_draft_layers must be positive, got {num_draft_layers}.")

    if num_draft_layers == 1:
        return [num_target_layers // 2]

    start = 1
    end = num_target_layers - 3
    if end < start:
        raise ValueError(
            "DFlash layer selection requires num_target_layers >= 4. "
            f"Got num_target_layers={num_target_layers}."
        )

    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


def get_dflash_layer_types(config: Any) -> Optional[Sequence[str]]:
    text_config = _get_text_config(config)
    layer_types = _cfg_get(text_config, "layer_types", _cfg_get(config, "layer_types"))
    if layer_types is None:
        return None
    if isinstance(layer_types, str) or not isinstance(layer_types, Sequence):
        raise ValueError(
            "DFLASH config.layer_types must be a sequence of attention type strings."
        )
    return layer_types


def get_dflash_attention_sliding_window_size(config: Any) -> Optional[int]:
    layer_types = get_dflash_layer_types(config)
    if layer_types is None or "sliding_attention" not in layer_types:
        return None

    text_config = _get_text_config(config)
    sliding_window = _cfg_get(
        text_config, "sliding_window", _cfg_get(config, "sliding_window")
    )
    if sliding_window is None:
        raise ValueError(
            "DFLASH sliding_attention layers require config.sliding_window."
        )

    # Hugging Face counts the current token in sliding_window; SGLang stores
    # window_left, so a HF window of 2048 becomes 2047 here.
    return int(sliding_window) - 1


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _get_text_config(config: Any) -> Any:
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get("text_config", config)
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return text_config
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        try:
            resolved = get_text_config()
            if resolved is not None:
                return resolved
        except TypeError:
            pass
    return config


def _get_dflash_config(config: Any) -> dict:
    if isinstance(config, dict):
        cfg = config.get("dflash_config", None)
    else:
        cfg = getattr(config, "dflash_config", None)
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg

    try:
        return dict(cfg)
    except Exception:
        return {}


def _parse_optional_int(
    value: Any,
    *,
    field_name: str,
    min_value: Optional[int] = None,
) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except Exception as e:
        raise ValueError(f"Invalid {field_name}={value!r}.") from e
    if min_value is not None and parsed < int(min_value):
        comparator = "positive" if int(min_value) == 1 else f">= {int(min_value)}"
        raise ValueError(f"{field_name} must be {comparator}, got {parsed}.")
    return parsed


@dataclass(frozen=True)
class DFlashDraftConfig:
    num_hidden_layers: Optional[int]
    num_target_layers: Optional[int]
    block_size: Optional[int]
    conv_kernel_size: int
    conv_group_size: int
    selector_rank: int
    selector_top_k: int
    output_multiplier: float
    final_logit_softcapping: Optional[float]
    target_layer_ids: Optional[List[int]]
    mask_token: str
    mask_token_id: Optional[int]

    def require_num_layers(self) -> int:
        if self.num_hidden_layers is None:
            raise ValueError(
                "DFLASH requires draft num_hidden_layers in config. "
                "Got config without num_hidden_layers."
            )
        return int(self.num_hidden_layers)

    def resolve_block_size(self, *, default: Optional[int] = None) -> Optional[int]:
        return self.block_size if self.block_size is not None else default

    def resolve_target_layer_ids(
        self,
        *,
        target_num_layers: int,
        draft_num_layers: Optional[int] = None,
    ) -> List[int]:
        target_num_layers = int(target_num_layers)
        if target_num_layers <= 0:
            raise ValueError(
                f"target_num_layers must be positive, got {target_num_layers}."
            )

        if self.target_layer_ids is None:
            if draft_num_layers is None:
                draft_num_layers = self.require_num_layers()
            return build_target_layer_ids(target_num_layers, int(draft_num_layers))

        resolved = list(self.target_layer_ids)
        if len(resolved) <= 0:
            raise ValueError(
                "DFLASH dflash_config.target_layer_ids must be non-empty. "
                f"Got len(target_layer_ids)={len(resolved)}."
            )
        for idx, val in enumerate(resolved):
            if val < 0 or val >= target_num_layers:
                raise ValueError(
                    "DFLASH target_layer_ids contains an out-of-range layer id. "
                    f"target_layer_ids[{idx}]={val}, target_num_layers={target_num_layers}."
                )
        return resolved


def parse_dflash_draft_config(*, draft_hf_config: Any) -> DFlashDraftConfig:
    """Parse and validate DFLASH draft config fields from HF config/dict."""
    dflash_cfg = _get_dflash_config(draft_hf_config)
    draft_text_config = _get_text_config(draft_hf_config)

    num_hidden_layers = _parse_optional_int(
        _cfg_get(draft_text_config, "num_hidden_layers", None),
        field_name="DFLASH draft num_hidden_layers",
        min_value=1,
    )
    raw_num_target_layers = dflash_cfg.get(
        "num_target_layers",
        _cfg_get(draft_hf_config, "num_target_layers", None),
    )
    num_target_layers = _parse_optional_int(
        raw_num_target_layers,
        field_name="DFLASH draft num_target_layers",
        min_value=1,
    )

    # Keep support for current checkpoints where block_size is top-level.
    raw_block_size = dflash_cfg.get(
        "block_size",
        _cfg_get(draft_hf_config, "block_size", None),
    )
    block_size = _parse_optional_int(
        raw_block_size,
        field_name="DFLASH block_size",
        min_value=1,
    )

    # DFlash2: grouped dynamic depthwise convolution and candidate selector.
    # Keep all fields optional/zero so first-generation DFlash checkpoints keep
    # the exact legacy path.
    conv_kernel_size = _parse_optional_int(
        dflash_cfg.get("conv_kernel_size", 0),
        field_name="DFLASH conv_kernel_size",
        min_value=0,
    )
    conv_group_size = _parse_optional_int(
        dflash_cfg.get("conv_group_size", 0),
        field_name="DFLASH conv_group_size",
        min_value=0,
    )
    if bool(conv_kernel_size) != bool(conv_group_size):
        raise ValueError(
            "DFLASH grouped convolution needs conv_kernel_size and conv_group_size "
            f"together. Got conv_kernel_size={conv_kernel_size}, "
            f"conv_group_size={conv_group_size}."
        )

    selector_rank = _parse_optional_int(
        dflash_cfg.get("selector_rank", 0),
        field_name="DFLASH selector rank",
        min_value=0,
    )
    selector_top_k = _parse_optional_int(
        dflash_cfg.get("selector_top_k", 0),
        field_name="DFLASH selector top_k",
        min_value=0,
    )
    if bool(selector_rank) != bool(selector_top_k):
        raise ValueError(
            "DFLASH selector needs rank and top_k together. "
            f"Got rank={selector_rank}, top_k={selector_top_k}."
        )

    output_multiplier = float(dflash_cfg.get("output_multiplier", 1.0))
    if output_multiplier <= 0:
        raise ValueError("DFLASH output_multiplier must be positive.")
    softcap = float(dflash_cfg.get("final_logit_softcapping") or 0.0)
    final_logit_softcapping = softcap if softcap > 0 else None

    layer_ids = dflash_cfg.get(
        "target_layer_ids",
        _cfg_get(draft_hf_config, "target_layer_ids", None),
    )
    parsed_target_layer_ids: Optional[List[int]]
    if layer_ids is None:
        parsed_target_layer_ids = None
    else:
        if not isinstance(layer_ids, (list, tuple)):
            raise ValueError(
                "DFLASH dflash_config.target_layer_ids must be a list of ints, "
                f"got type={type(layer_ids).__name__}."
            )
        parsed_target_layer_ids = [int(x) for x in layer_ids]
        if len(parsed_target_layer_ids) <= 0:
            raise ValueError(
                "DFLASH dflash_config.target_layer_ids must be non-empty. "
                f"Got len(target_layer_ids)={len(parsed_target_layer_ids)}."
            )

    mask_token = dflash_cfg.get("mask_token", None)
    if mask_token is None:
        mask_token = DEFAULT_DFLASH_MASK_TOKEN
    if not isinstance(mask_token, str) or not mask_token:
        raise ValueError(
            "DFLASH dflash_config.mask_token must be a non-empty string, "
            f"got {mask_token!r}."
        )

    mask_token_id = dflash_cfg.get("mask_token_id", None)
    if mask_token_id is not None:
        if not isinstance(mask_token_id, Integral) or isinstance(mask_token_id, bool):
            raise ValueError(
                "DFLASH dflash_config.mask_token_id must be an integer, "
                f"got {mask_token_id!r} (type={type(mask_token_id).__name__})."
            )
        mask_token_id = int(mask_token_id)
        if mask_token_id < 0:
            raise ValueError(
                "DFLASH dflash_config.mask_token_id must be non-negative, "
                f"got {mask_token_id}."
            )

    return DFlashDraftConfig(
        num_hidden_layers=num_hidden_layers,
        num_target_layers=num_target_layers,
        block_size=block_size,
        conv_kernel_size=conv_kernel_size,
        conv_group_size=conv_group_size,
        selector_rank=selector_rank,
        selector_top_k=selector_top_k,
        output_multiplier=output_multiplier,
        final_logit_softcapping=final_logit_softcapping,
        target_layer_ids=parsed_target_layer_ids,
        mask_token=mask_token,
        mask_token_id=mask_token_id,
    )


def can_dflash_slice_qkv_weight(qkv_proj: Any) -> Tuple[bool, str]:
    """Validate whether DFlash can slice KV weights from a fused QKV linear layer."""
    quant_method = getattr(qkv_proj, "quant_method", None)
    if not isinstance(quant_method, UnquantizedLinearMethod):
        return (
            False,
            "quantized qkv_proj is not supported for this path "
            f"(quant_method={type(quant_method).__name__})",
        )
    if not hasattr(qkv_proj, "weight"):
        return False, "qkv weight tensor is missing"
    return True, ""


def can_dflash_use_fused_qkv_proj(qkv_proj: Any) -> Tuple[bool, str]:
    """Validate whether a QKV layer is eligible for DFlash fused KV materialization."""
    eligible, reason = can_dflash_slice_qkv_weight(qkv_proj)
    if not eligible:
        return False, reason
    if getattr(qkv_proj, "bias", None) is not None:
        return False, "qkv bias is not supported for fused KV path"
    return True, ""


def compute_dflash_correct_drafts_and_bonus(
    *,
    candidates: torch.Tensor,
    target_predict: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute DFlash accept lengths and bonus tokens (greedy verify rule).

    Args:
        candidates: Token ids proposed by the DFlash draft, including the current token.
            Shape: [bs, block_size]. candidates[:, 0] is the current token.
        target_predict: Token ids predicted by the target model for each position in the block.
            Shape: [bs, block_size]. target_predict[:, t] corresponds to argmax at position t.

    Returns:
        correct_len: int32 tensor [bs], number of accepted *draft* tokens (excluding current token and bonus token).
        bonus: int64 tensor [bs], the target-predicted token at index correct_len (the "bonus" token to append).

    Notes:
        Matches the reference implementation rule:
          accept while candidates[:, 1:] == target_predict[:, :-1] consecutively.
    """
    if candidates.ndim != 2:
        raise ValueError(f"candidates must be 2D, got shape={tuple(candidates.shape)}")
    if target_predict.shape != candidates.shape:
        raise ValueError(
            "target_predict must have the same shape as candidates. "
            f"candidates.shape={tuple(candidates.shape)}, target_predict.shape={tuple(target_predict.shape)}"
        )

    bs, block_size = candidates.shape
    if bs <= 0:
        raise ValueError(f"batch size must be positive, got {bs}.")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    matches = candidates[:, 1:] == target_predict[:, :-1]
    correct_len = matches.to(torch.int32).cumprod(dim=1).sum(dim=1)
    bonus = target_predict[torch.arange(bs, device=target_predict.device), correct_len]
    return correct_len, bonus.to(torch.int64)


def _build_dflash_reference_target_probs(
    *,
    next_token_logits: torch.Tensor,
    sampling_info: Any,
    draft_token_num: int,
) -> torch.Tensor:
    """Build the target distribution used by normal SGLang sampling.

    This HIP-safe reference path intentionally uses only PyTorch ops.  The common
    case (small top-k such as 20/50) avoids a full-vocabulary sort; a full sort is
    used only when top-k spans the whole vocabulary and top-p requires ranking.
    """
    if next_token_logits.ndim != 2:
        raise ValueError(
            "next_token_logits must be 2D, "
            f"got shape={tuple(next_token_logits.shape)}."
        )
    if draft_token_num <= 0:
        raise ValueError(f"draft_token_num must be positive, got {draft_token_num}.")

    rows, vocab_size = next_token_logits.shape
    if rows % draft_token_num != 0:
        raise ValueError(
            "next_token_logits row count must be divisible by draft_token_num, "
            f"got rows={rows}, draft_token_num={draft_token_num}."
        )

    temperatures = torch.repeat_interleave(
        sampling_info.temperatures, draft_token_num, dim=0
    ).reshape(-1, 1)
    temperatures = temperatures.to(
        device=next_token_logits.device, dtype=torch.float32
    ).clamp_min_(1e-5)
    scaled_logits = next_token_logits.float() / temperatures
    probs = F.softmax(scaled_logits, dim=-1)

    top_ks = torch.repeat_interleave(
        sampling_info.top_ks, draft_token_num, dim=0
    ).reshape(-1)
    top_ks = top_ks.to(device=probs.device, dtype=torch.int64).clamp_(1, vocab_size)
    top_ps = torch.repeat_interleave(
        sampling_info.top_ps, draft_token_num, dim=0
    ).reshape(-1)
    top_ps = top_ps.to(device=probs.device, dtype=probs.dtype)
    min_ps_src = getattr(sampling_info, "min_ps", None)
    if min_ps_src is None:
        min_ps = torch.zeros_like(top_ps)
    else:
        min_ps = torch.repeat_interleave(min_ps_src, draft_token_num, dim=0).reshape(-1)
        min_ps = min_ps.to(device=probs.device, dtype=probs.dtype)

    need_top_p = bool(getattr(sampling_info, "need_top_p_sampling", False))
    need_min_p = bool(getattr(sampling_info, "need_min_p_sampling", False))
    max_top_k = int(top_ks.max().item())

    if max_top_k < vocab_size:
        # top-k values are still probabilities from the full softmax, so top-p
        # thresholds match the normal sampler before the final renormalization.
        vals, idx = torch.topk(probs, k=max_top_k, dim=-1)
        ranks = torch.arange(max_top_k, device=probs.device, dtype=torch.int64)[None, :]
        keep = ranks < top_ks[:, None]
        if need_top_p:
            cdf = torch.cumsum(vals, dim=-1)
            keep &= (cdf - vals) <= top_ps[:, None]
        if need_min_p:
            keep &= vals >= vals[:, :1] * min_ps[:, None]
        vals = torch.where(keep, vals, torch.zeros_like(vals))
        vals.div_(vals.sum(dim=-1, keepdim=True).clamp_min_(1e-20))
        return torch.zeros_like(probs).scatter_(1, idx, vals)

    if need_top_p:
        vals, idx = torch.sort(probs, dim=-1, descending=True)
        ranks = torch.arange(vocab_size, device=probs.device, dtype=torch.int64)[None, :]
        keep = ranks < top_ks[:, None]
        cdf = torch.cumsum(vals, dim=-1)
        keep &= (cdf - vals) <= top_ps[:, None]
        if need_min_p:
            keep &= vals >= vals[:, :1] * min_ps[:, None]
        vals = torch.where(keep, vals, torch.zeros_like(vals))
        vals.div_(vals.sum(dim=-1, keepdim=True).clamp_min_(1e-20))
        return torch.zeros_like(probs).scatter_(1, idx, vals)

    if need_min_p:
        threshold = probs.amax(dim=-1, keepdim=True) * min_ps[:, None]
        probs = torch.where(probs >= threshold, probs, torch.zeros_like(probs))
        probs.div_(probs.sum(dim=-1, keepdim=True).clamp_min_(1e-20))
    return probs


def _sample_probs_with_uniform(
    probs: torch.Tensor, uniforms: torch.Tensor
) -> torch.Tensor:
    """Sample one token per row from normalized probabilities using fixed uniforms."""
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D, got shape={tuple(probs.shape)}")
    uniforms = uniforms.to(device=probs.device, dtype=probs.dtype).reshape(-1)
    if uniforms.numel() != probs.shape[0]:
        raise ValueError(
            f"uniform count mismatch: uniforms={uniforms.numel()}, rows={probs.shape[0]}."
        )
    cdf = torch.cumsum(probs, dim=-1)
    token = torch.sum(cdf < uniforms[:, None], dim=-1).to(torch.int64)
    return token.clamp_(max=probs.shape[-1] - 1)


def compute_dflash_selector_sampling_correct_drafts_and_bonus(
    *,
    candidates: torch.Tensor,
    next_token_logits: torch.Tensor,
    sampling_info: Any,
    selector_candidate_ids: torch.Tensor,
    selector_q_rows: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference rejection sampling for the DFlash2 selector path.

    ``selector_q_rows`` is the sparse proposal distribution q over
    ``selector_candidate_ids`` for each of the block_size-1 proposed tokens.
    Target probabilities p follow the normal SGLang temperature/top-k/top-p/min-p
    semantics.  The implementation keeps q sparse and only materializes relu(p-q)
    for rows that actually reject a proposal.
    """
    if candidates.ndim != 2:
        raise ValueError(f"candidates must be 2D, got shape={tuple(candidates.shape)}")
    bs, draft_token_num = candidates.shape
    gamma = draft_token_num - 1
    if gamma <= 0:
        raise ValueError(
            f"selector sampling requires draft_token_num >= 2, got {draft_token_num}."
        )
    if selector_candidate_ids.shape[:2] != (bs, gamma):
        raise ValueError(
            "selector_candidate_ids shape mismatch: "
            f"expected prefix={(bs, gamma)}, got={tuple(selector_candidate_ids.shape)}."
        )
    if selector_q_rows.shape != selector_candidate_ids.shape:
        raise ValueError(
            "selector_q_rows shape mismatch: "
            f"q={tuple(selector_q_rows.shape)}, ids={tuple(selector_candidate_ids.shape)}."
        )
    if uniform_samples.shape != (bs, gamma):
        raise ValueError(
            f"uniform_samples shape mismatch: expected={(bs, gamma)}, "
            f"got={tuple(uniform_samples.shape)}."
        )
    if uniform_samples_for_final_sampling.shape != (bs,):
        raise ValueError(
            "uniform_samples_for_final_sampling shape mismatch: "
            f"expected={(bs,)}, got={tuple(uniform_samples_for_final_sampling.shape)}."
        )

    target_probs = _build_dflash_reference_target_probs(
        next_token_logits=next_token_logits,
        sampling_info=sampling_info,
        draft_token_num=draft_token_num,
    ).view(bs, draft_token_num, -1)
    selector_candidate_ids = selector_candidate_ids.to(torch.int64)
    selector_q_rows = selector_q_rows.float()
    uniforms = uniform_samples.to(device=candidates.device, dtype=torch.float32)
    final_uniforms = uniform_samples_for_final_sampling.to(
        device=candidates.device, dtype=torch.float32
    )

    correct_len = torch.full(
        (bs,), gamma, dtype=torch.int32, device=candidates.device
    )
    bonus = torch.empty((bs,), dtype=torch.int64, device=candidates.device)
    active = torch.ones((bs,), dtype=torch.bool, device=candidates.device)

    for step in range(gamma):
        proposed = candidates[:, step + 1].to(torch.int64)
        q_ids = selector_candidate_ids[:, step, :]
        q_vals = selector_q_rows[:, step, :]
        q_selected = torch.sum(
            torch.where(q_ids == proposed[:, None], q_vals, torch.zeros_like(q_vals)),
            dim=-1,
        )
        p_selected = target_probs[:, step, :].gather(1, proposed[:, None]).squeeze(1)
        accept_prob = torch.where(
            q_selected > 0,
            (p_selected / q_selected.clamp_min(1e-20)).clamp(max=1.0),
            torch.zeros_like(p_selected),
        )
        accepted = uniforms[:, step] < accept_prob
        rejected = active & ~accepted
        rejected_rows = torch.nonzero(rejected, as_tuple=False).reshape(-1)
        if rejected_rows.numel() > 0:
            residual = target_probs[rejected_rows, step, :].clone()
            residual.scatter_add_(
                1,
                q_ids[rejected_rows],
                -q_vals[rejected_rows].to(residual.dtype),
            )
            residual.clamp_min_(0.0)
            residual_sum = residual.sum(dim=-1, keepdim=True)
            fallback = residual_sum.squeeze(1) <= 1e-20
            if bool(fallback.any().item()):
                residual[fallback] = target_probs[rejected_rows[fallback], step, :]
                residual_sum = residual.sum(dim=-1, keepdim=True)
            residual.div_(residual_sum.clamp_min_(1e-20))
            bonus[rejected_rows] = _sample_probs_with_uniform(
                residual, final_uniforms[rejected_rows]
            )
            correct_len[rejected_rows] = int(step)
            active[rejected_rows] = False

    remaining_rows = torch.nonzero(active, as_tuple=False).reshape(-1)
    if remaining_rows.numel() > 0:
        bonus[remaining_rows] = _sample_probs_with_uniform(
            target_probs[remaining_rows, gamma, :], final_uniforms[remaining_rows]
        )
    return correct_len, bonus


def compute_dflash_sampling_correct_drafts_and_bonus(
    *,
    candidates: torch.Tensor,
    next_token_logits: torch.Tensor,
    sampling_info: Any,
    threshold_single: Optional[float] = None,
    threshold_acc: Optional[float] = None,
    uniform_samples: Optional[torch.Tensor] = None,
    uniform_samples_for_final_sampling: Optional[torch.Tensor] = None,
    use_sparse_topk: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute DFlash accept lengths and bonus tokens for non-greedy sampling.

    This is a chain-specialized variant of speculative target-only verification:
      - DFlash proposals are linear (topk == 1), so each verify level has at most one candidate.
      - When a candidate is rejected at a level, the final token is sampled from
        `relu(q - p)` where `p` has only the rejected candidate mass.
    """
    if not _DFLASH_SAMPLING_VERIFY_AVAILABLE:
        raise RuntimeError(
            "DFLASH non-greedy verification is unavailable on this build/device."
        )
    if candidates.ndim != 2:
        raise ValueError(f"candidates must be 2D, got shape={tuple(candidates.shape)}")
    if next_token_logits.ndim != 2:
        raise ValueError(
            "next_token_logits must be 2D, "
            f"got shape={tuple(next_token_logits.shape)}."
        )

    bs, draft_token_num = candidates.shape
    if bs <= 0:
        raise ValueError(f"batch size must be positive, got {bs}.")
    if draft_token_num <= 0:
        raise ValueError(f"draft_token_num must be positive, got {draft_token_num}.")
    if next_token_logits.shape[0] != bs * draft_token_num:
        raise ValueError(
            "next_token_logits row count mismatch. "
            f"Expected {bs * draft_token_num}, got {next_token_logits.shape[0]}."
        )
    if candidates.device != next_token_logits.device:
        raise ValueError(
            "candidates and next_token_logits must be on the same device, "
            f"got {candidates.device} and {next_token_logits.device}."
        )

    if threshold_single is None:
        from sglang.srt.server_args import get_global_server_args

        threshold_single = get_global_server_args().speculative_accept_threshold_single
    if threshold_acc is None:
        from sglang.srt.server_args import get_global_server_args

        threshold_acc = get_global_server_args().speculative_accept_threshold_acc
    threshold_single = float(threshold_single)
    threshold_acc = max(float(threshold_acc), 1e-9)

    device = next_token_logits.device

    if uniform_samples is None:
        uniform_samples = torch.rand(
            (bs, draft_token_num), dtype=torch.float32, device=device
        )
    else:
        if uniform_samples.shape != (bs, draft_token_num):
            raise ValueError(
                "uniform_samples shape mismatch. "
                f"Expected {(bs, draft_token_num)}, got {tuple(uniform_samples.shape)}."
            )
        uniform_samples = uniform_samples.to(device=device, dtype=torch.float32)

    if uniform_samples_for_final_sampling is None:
        uniform_samples_for_final_sampling = torch.rand(
            (bs,), dtype=torch.float32, device=device
        )
    else:
        if uniform_samples_for_final_sampling.shape != (bs,):
            raise ValueError(
                "uniform_samples_for_final_sampling shape mismatch. "
                f"Expected {(bs,)}, got {tuple(uniform_samples_for_final_sampling.shape)}."
            )
        uniform_samples_for_final_sampling = uniform_samples_for_final_sampling.to(
            device=device,
            dtype=torch.float32,
        )

    need_top_k = bool(getattr(sampling_info, "need_top_k_sampling", True))
    need_top_p = bool(getattr(sampling_info, "need_top_p_sampling", False))
    # Build target distribution once over all verify rows.
    expanded_temperature = torch.repeat_interleave(
        sampling_info.temperatures, draft_token_num, dim=0
    )
    scaled_logits = next_token_logits / expanded_temperature
    sparse_topk_applied = False

    if use_sparse_topk and need_top_k:
        repeated_top_ks = torch.repeat_interleave(
            sampling_info.top_ks, draft_token_num, dim=0
        ).to(dtype=torch.int64)
        vocab_size = int(scaled_logits.shape[-1])
        repeated_top_ks.clamp_(min=1, max=vocab_size)
        max_top_k = int(repeated_top_ks.max().item())

        # Sparse exact path for top-k/top-p (top-k-first semantics), then scatter to dense.
        if 0 < max_top_k < vocab_size:
            topk_logits, topk_indices = torch.topk(scaled_logits, k=max_top_k, dim=-1)
            if not torch.all(repeated_top_ks == max_top_k):
                ranks = torch.arange(max_top_k, device=device, dtype=torch.int64)[
                    None, :
                ]
                valid = ranks < repeated_top_ks.unsqueeze(1)
                topk_logits = topk_logits.masked_fill(~valid, float("-inf"))

            topk_probs = F.softmax(topk_logits, dim=-1)
            if need_top_p:
                repeated_top_ps = torch.repeat_interleave(
                    sampling_info.top_ps, draft_token_num, dim=0
                )
                topk_probs = top_p_renorm_prob(topk_probs, repeated_top_ps)

            target_probs = torch.zeros_like(scaled_logits, dtype=topk_probs.dtype)
            target_probs.scatter_(1, topk_indices, topk_probs)
            sparse_topk_applied = True

    if not sparse_topk_applied:
        target_probs = F.softmax(scaled_logits, dim=-1)
        if need_top_k:
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(sampling_info.top_ks, draft_token_num, dim=0),
            )
        if need_top_p:
            target_probs = top_p_renorm_prob(
                target_probs,
                torch.repeat_interleave(sampling_info.top_ps, draft_token_num, dim=0),
            )
    target_probs = target_probs.view(bs, draft_token_num, -1).contiguous()
    draft_probs = torch.zeros_like(target_probs)

    (
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        predicts,
        accept_index,
        accept_token_num,
    ) = _get_or_create_chain_verify_buffers(
        bs=bs,
        draft_token_num=draft_token_num,
        device=device,
    )
    candidates_i64 = (
        candidates if candidates.dtype == torch.int64 else candidates.to(torch.int64)
    )
    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates_i64,
        # kwarg LHS retained as `retrive_*` to match sgl_kernel op schema.
        retrive_index=retrieve_index,
        retrive_next_token=retrieve_next_token,
        retrive_next_sibling=retrieve_next_sibling,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=True,
    )

    correct_len = accept_token_num
    row_ids = torch.arange(bs, dtype=torch.long, device=device)
    accept_pos = accept_index[row_ids, correct_len.to(torch.long)].to(torch.long)
    bonus = predicts[accept_pos].to(torch.int64)
    return correct_len, bonus
