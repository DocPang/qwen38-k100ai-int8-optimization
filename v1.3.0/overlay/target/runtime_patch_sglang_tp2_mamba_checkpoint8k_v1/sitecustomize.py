"""TP2 cache/resume candidate: align Mamba prefill checkpoints to 8K chunks.

This is the TP2 analogue of the proven TP1 checkpoint-v2 / current TP4
checkpoint16k mechanism.  It changes checkpoint *creation* in ScheduleBatch,
not only ancestor selection at restore time.

Parent: current TP2 longtail-v1 target/DFlash stack.
Launch contract: --mamba-track-interval 8192.

Do NOT change ServerArgs.mamba_cache_chunk_size here.  The goal is a narrow
prefill checkpoint policy A/B while preserving the rest of SourceFind 0.5.12.
"""
from __future__ import annotations

import runpy

_ROOT = "/data/qwen38-27b-k100ai-int8-opt"
_PARENT = f"{_ROOT}/runtime_patch_sglang_tp2_longtail_varlen_v1/sitecustomize.py"
_CHECKPOINT = 8192

runpy.run_path(_PARENT, run_name="__q38_tp2_checkpoint8k_parent__")

from sglang.srt.managers.schedule_batch import ScheduleBatch


def _tp2_checkpoint8k_prepare(
    self,
    req,
    mamba_track_mask_cpu,
    mamba_track_indices_cpu,
    mamba_track_seqlens_cpu,
):
    prefix_len = len(req.prefix_indices)
    extend_len = int(req.extend_input_len)
    end_len = prefix_len + extend_len
    desired = (end_len // _CHECKPOINT) * _CHECKPOINT
    can_track_new = desired > prefix_len

    mamba_track_mask_cpu.append(can_track_new)
    mamba_track_indices_cpu.append(
        req.mamba_ping_pong_track_buffer[req.mamba_next_track_idx].item()
    )

    track_seqlen = -1
    if can_track_new:
        # Exact endpoint can use last_recurrent_state.  For an interior checkpoint
        # request h at desired by passing desired+1, matching SourceFind's
        # _force_track_h convention used by the validated TP1/TP4 policies.
        track_seqlen = desired if desired == end_len else desired + 1

        branch = req.mamba_branching_seqlen
        if (
            branch is not None
            and branch > prefix_len
            and branch < track_seqlen
            and branch % _CHECKPOINT == 0
        ):
            desired = int(branch)
            track_seqlen = desired + 1

        req.mamba_next_track_idx = (
            self.req_to_token_pool.get_mamba_ping_pong_other_idx(
                req.mamba_next_track_idx
            )
        )
        req.mamba_last_track_seqlen = desired
    elif prefix_len > 0 and prefix_len % _CHECKPOINT == 0:
        req.mamba_last_track_seqlen = prefix_len
    else:
        req.mamba_last_track_seqlen = None

    mamba_track_seqlens_cpu.append(track_seqlen)


ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend = _tp2_checkpoint8k_prepare

print(
    "[K100 TP2 Mamba checkpoint8k-v1] ACTIVE: prefill checkpoints=8192; "
    "launcher must pin --mamba-track-interval 8192",
    flush=True,
)
