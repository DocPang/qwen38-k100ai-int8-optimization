"""TP4 corrected U036 + exact Gemma RMSNorm->INT8 QKVZ producer fusion v1.

Goal: remove the standalone Gemma RMSNorm + QKVZ dynamic-token INT8 quant pair
from true-M1 GDN decode while keeping TP4 communication, BA math, attention and
all prefill paths unchanged.

For an eligible linear-attention layer with an existing residual:
  stock: hidden+residual -> GemmaRMSNorm BF16 -> QKVZ dynamic INT8 quant -> GEMV
                                      \-------> BA BF16 projection
  v1:    lm_faster_rmsquant(hidden,residual,update_input=True)
           -> hidden is the exact normalized BF16 carrier
           -> residual is the exact stock residual_out
           -> q/scale are bitwise equal to stock per_token_quant_int8(normed)
         QKVZ consumes q/scale directly; BA still consumes normalized BF16.

The residual=None first layer is intentionally left on the parent path so no
clone/copy is introduced merely to preserve the pre-norm residual contract.
Only exact TP4 M1 QKVZ K5120->N4096 + BA K5120->N24 shapes are admitted.
"""
from __future__ import annotations

import os
import runpy

_ROOT = "/data/qwen38-27b-k100ai-int8-opt"
_PARENT = f"{_ROOT}/runtime_patch_sglang_tp4_u036_longkv_v1/sitecustomize.py"
runpy.run_path(_PARENT, run_name="__q38_tp4_u036_parent_for_rmsqkvz_v1__")

if os.getenv("SGLANG_Q38_TP4_RMS_QKVZ_M1", "0") == "1":
    import torch
    from lmslim.quantize.quant_ops import lm_faster_rmsquant
    from sglang.srt.layers.communicator import LayerCommunicator
    from sglang.srt.layers.layernorm import GemmaRMSNorm
    from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
    from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet

    K = 5120
    NQ = 4096
    NB = 24
    _producer_hits = 0
    _consumer_hits = 0
    _guard_miss_logged = False

    # N7 (already in the inherited stack) marks the linear-attention decoder
    # communicators and installs a TP1-only producer.  Wrap that producer and
    # preempt only exact TP4/M1/residual-present rows; every other case falls
    # through to the inherited implementation unchanged.
    _prev_prepare_attn = LayerCommunicator.prepare_attn

    def _prepare_attn_tp4_rmsqkvz(
        self,
        hidden_states,
        residual,
        forward_batch,
        quant_format="",
        post_residual_addition=None,
    ):
        global _producer_hits
        norm = getattr(self, "input_layernorm", None)
        if (
            getattr(self, "k100_q38_rms_gdn_m1", False)
            and int(getattr(self._context, "tp_size", -1)) == 4
            and int(getattr(self._context, "attn_tp_size", -1)) == 4
            and post_residual_addition is None
            and isinstance(norm, GemmaRMSNorm)
            and isinstance(hidden_states, torch.Tensor)
            and hidden_states.ndim == 2
            and tuple(hidden_states.shape) == (1, K)
            and hidden_states.dtype is torch.bfloat16
            and isinstance(residual, torch.Tensor)
            and residual.ndim == 2
            and tuple(residual.shape) == (1, K)
            and residual.dtype is torch.bfloat16
        ):
            q, s = lm_faster_rmsquant(
                input=hidden_states,
                rms_weight=norm.gemma_weight,
                epsilon=norm.variance_epsilon,
                quant_dtype=torch.int8,
                residual=residual,
                update_input=True,
            )
            if (
                q.dtype is not torch.int8
                or tuple(q.shape) != (1, K)
                or s.dtype is not torch.float32
                or tuple(s.shape) != (1, 1)
            ):
                raise RuntimeError(
                    f"TP4 RMS-QKVZ producer drift q={tuple(q.shape)}/{q.dtype} "
                    f"s={tuple(s.shape)}/{s.dtype}"
                )
            _producer_hits += 1
            if _producer_hits <= 4 or _producer_hits in (16, 32, 64, 128):
                print(
                    "[K100 SGLang TP4 RMS->QKVZ INT8] PRODUCER ACTIVE "
                    f"hit={_producer_hits} graph={bool(get_is_capture_mode())}",
                    flush=True,
                )
            # hidden_states was updated in-place to the exact normalized BF16
            # output; residual was updated in-place to stock residual_out.
            return (hidden_states, q, s), residual
        return _prev_prepare_attn(
            self,
            hidden_states,
            residual,
            forward_batch,
            quant_format=quant_format,
            post_residual_addition=post_residual_addition,
        )

    LayerCommunicator.prepare_attn = _prepare_attn_tp4_rmsqkvz

    _prev_input_proj = Qwen3_5GatedDeltaNet._forward_input_proj

    def _input_proj_tp4_prequant(self, hidden_states):
        global _consumer_hits, _guard_miss_logged
        if isinstance(hidden_states, tuple) and len(hidden_states) == 3:
            carrier, q, s = hidden_states
            qkvz = self.in_proj_qkvz
            ba = self.in_proj_ba
            q_method = getattr(qkvz, "quant_method", None)
            qw = getattr(qkvz, "weight", None)
            qs = getattr(qkvz, "weight_scale", None)
            bw = getattr(ba, "weight", None)
            exact = (
                isinstance(carrier, torch.Tensor)
                and carrier.dtype is torch.bfloat16
                and tuple(carrier.shape) == (1, K)
                and isinstance(q, torch.Tensor)
                and q.dtype is torch.int8
                and tuple(q.shape) == (1, K)
                and q.is_contiguous()
                and isinstance(s, torch.Tensor)
                and s.dtype is torch.float32
                and tuple(s.shape) == (1, 1)
                and q_method is not None
                and isinstance(qw, torch.Tensor)
                and qw.dtype is torch.int8
                and tuple(qw.shape) == (K, NQ)
                and isinstance(qs, torch.Tensor)
                and qs.dtype is torch.float32
                and qs.numel() == NQ
                and getattr(qkvz, "bias", None) is None
                and not bool(getattr(qkvz, "gather_output", False))
                and isinstance(bw, torch.Tensor)
                and bw.dtype in (torch.bfloat16, torch.float16)
                and tuple(bw.shape) == (NB, K)
                and getattr(ba, "bias", None) is None
                and not bool(getattr(ba, "gather_output", False))
            )
            if exact:
                # Call the quant method directly so prequantized input is used
                # without globally enabling SGLANG_USE_FUSED_RMS_QUANT.  Its
                # CompressedTensors apply forwards input_quant_args to the W8A8
                # scheme, which then reaches the inherited TP4 LDS-x scaled_mm.
                projected_qkvz = q_method.apply(
                    qkvz, carrier, None, input_quant_args=[q, s]
                )
                # Keep BA exactly on the inherited BF16 path. It consumes the
                # same normalized BF16 carrier stock GemmaRMSNorm would return.
                projected_ba, _ = ba(carrier)
                _consumer_hits += 1
                if _consumer_hits <= 4 or _consumer_hits in (16, 32, 64, 128):
                    print(
                        "[K100 SGLang TP4 RMS->QKVZ INT8] CONSUMER ACTIVE "
                        f"hit={_consumer_hits} qkvz=({K},{NQ}) ba=({K},{NB})",
                        flush=True,
                    )
                return projected_qkvz, projected_ba
            if int(getattr(self, "attn_tp_size", -1)) == 4:
                raise RuntimeError(
                    "TP4 RMS->QKVZ consumer guard drift after producer tuple: "
                    f"carrier={getattr(carrier, 'shape', None)}/{getattr(carrier, 'dtype', None)} "
                    f"q={getattr(q, 'shape', None)}/{getattr(q, 'dtype', None)} "
                    f"s={getattr(s, 'shape', None)}/{getattr(s, 'dtype', None)} "
                    f"qkvz_w={getattr(qw, 'shape', None)}/{getattr(qw, 'dtype', None)} "
                    f"ba_w={getattr(bw, 'shape', None)}/{getattr(bw, 'dtype', None)}"
                )
            if not _guard_miss_logged:
                _guard_miss_logged = True
                print(
                    "[K100 SGLang TP4 RMS->QKVZ INT8] non-TP4 tuple guard miss -> parent",
                    flush=True,
                )
        return _prev_input_proj(self, hidden_states)

    Qwen3_5GatedDeltaNet._forward_input_proj = _input_proj_tp4_prequant
    print(
        "[K100 SGLang TP4 RMS->QKVZ INT8 v1] installed: residual-present M1 "
        "linear-attn only; BF16 BA preserved; first/residual-none layer parent",
        flush=True,
    )
