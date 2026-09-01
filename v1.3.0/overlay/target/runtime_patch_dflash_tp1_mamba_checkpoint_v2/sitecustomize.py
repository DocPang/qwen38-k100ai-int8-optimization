"""TP1 narrow Mamba prefix-checkpoint repair v2.

Keep the production TP1 Final-v2 runtime unchanged (notably
mamba_cache_chunk_size=64), but constrain *prefill prefix-cache checkpoints* to
absolute 8192-token boundaries so a cached continuation follows the same outer
chunk partition as cold chunked-prefill.

Unlike v1, this does NOT globally change mamba_cache_chunk_size and therefore
must not perturb DFlash verify/decode kernels.

Launch contract: --mamba-track-interval 8192.  This only limits decode-time
prefix checkpoints; it does not change mamba_cache_chunk_size.
"""
from __future__ import annotations

import runpy

_ROOT = "/data/qwen38-27b-k100ai-int8-opt"
_PARENT = f"{_ROOT}/runtime_patch_dflash_tp1_final_v2/sitecustomize.py"
_CHECKPOINT = 8192
_FLA = 64

runpy.run_path(_PARENT, run_name="__q38_tp1_checkpoint_v2_parent__")

from sglang.srt.managers.schedule_batch import ScheduleBatch


def _tp1_checkpoint8192_prepare(
    self,
    req,
    mamba_track_mask_cpu,
    mamba_track_indices_cpu,
    mamba_track_seqlens_cpu,
):
    """Track only absolute 8192-token prefill checkpoints.

    The ping-pong buffer is left untouched on a short tail; in that case
    mamba_last_track_seqlen is pinned to the already-protected 8192 boundary so
    cache_finished_req reuses the previous valid recurrent state instead of
    relabeling a finer-grained state.
    """
    prefix_len = len(req.prefix_indices)
    extend_len = int(req.extend_input_len)
    end_len = prefix_len + extend_len

    # The prefix returned by this repaired cache should itself be checkpoint
    # aligned.  On a cold first chunk prefix_len can be zero.
    desired = (end_len // _CHECKPOINT) * _CHECKPOINT
    can_track_new = desired > prefix_len

    mamba_track_mask_cpu.append(can_track_new)
    mamba_track_indices_cpu.append(
        req.mamba_ping_pong_track_buffer[req.mamba_next_track_idx].item()
    )

    track_seqlen = -1
    if can_track_new:
        # desired is always 8192-aligned and therefore FLA(64)-aligned.
        # If desired is the final position, last_recurrent_state is correct.
        # Otherwise request h at the desired boundary by passing desired+1,
        # matching upstream's _force_track_h convention.
        track_seqlen = desired if desired == end_len else desired + 1

        # A valid Radix branching checkpoint can take precedence only when it
        # is itself on the safe 8192 grid and lies in this extend batch.
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
    else:
        # No new full checkpoint in this tail.  Preserve the previous valid
        # recurrent state in the ping-pong buffer and tell cache_finished_req
        # to truncate KV/token insertion to that same boundary.
        if prefix_len > 0 and prefix_len % _CHECKPOINT == 0:
            req.mamba_last_track_seqlen = prefix_len
        else:
            req.mamba_last_track_seqlen = None

    mamba_track_seqlens_cpu.append(track_seqlen)


ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend = (
    _tp1_checkpoint8192_prepare
)

print(
    "[K100 TP1 Mamba checkpoint repair v2] ACTIVE: prefill checkpoints=8192; "
    "mamba_cache_chunk_size remains upstream; launcher must use track_interval=8192",
    flush=True,
)
