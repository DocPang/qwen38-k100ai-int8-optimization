"""TP4 DFlash/Mamba verify-boundary correctness hotfix v1.

Fixes the DFlash Mamba checkpoint materialization bug described upstream in
sgl-project/sglang#37817.  The old DFlash worker compares
``seq_lens_pre_verify`` with ``batch.seq_lens`` before the latter is advanced by
accepted verify tokens, so a verify round that crosses a Mamba tracking boundary
is not detected and the boundary state is never copied to the tracking buffer.

This layer changes only DFlashWorker._update_target_mamba_state_after_verify.
All attention, sampling, allocator, Radix, verifier and scheduler behavior is
inherited unchanged from the already-loaded parent stack.
"""
from __future__ import annotations

import os

import torch

from sglang.srt.speculative.dflash_worker import DFlashWorker


_PATCH_TAG = "q38_tp4_mamba_verify_boundary_fix_v1"
_orig = DFlashWorker._update_target_mamba_state_after_verify


def _compute_steps_to_track(
    seq_lens_pre_verify: torch.Tensor,
    commit_lens: torch.Tensor,
    interval: int,
) -> torch.Tensor:
    """Return accepted verify step to save at a crossed tracking boundary.

    -1 means no checkpoint is crossed or the crossing step was rejected.
    Step indices are zero-based inside the verify block.
    """
    if interval <= 0:
        raise RuntimeError(f"invalid mamba_track_interval={interval}")
    commit = commit_lens.to(dtype=seq_lens_pre_verify.dtype)
    seq_lens_post_verify = seq_lens_pre_verify + commit
    to_track_mask = (
        seq_lens_pre_verify // interval
        != seq_lens_post_verify // interval
    )
    tracking_point = (
        seq_lens_post_verify // interval * interval
    )
    to_track_ith = torch.clamp(
        tracking_point - seq_lens_pre_verify - 1,
        min=0,
    )
    can_track_mask = to_track_mask & (
        to_track_ith < commit
    )
    return torch.where(
        can_track_mask,
        to_track_ith.to(torch.int64),
        torch.full_like(to_track_ith, -1, dtype=torch.int64),
    )


def _fixed_update_target_mamba_state_after_verify(
    self,
    *,
    batch,
    seq_lens_pre_verify: torch.Tensor,
    commit_lens: torch.Tensor,
) -> None:
    attn_backend = self.target_worker.model_runner.attn_backend
    if not hasattr(attn_backend, "update_mamba_state_after_mtp_verify"):
        return

    last_correct_step_indices = commit_lens.to(torch.int64) - 1
    mamba_steps_to_track = None

    if batch.mamba_track_indices is not None:
        interval = int(self.server_args.mamba_track_interval)
        mamba_steps_to_track = _compute_steps_to_track(
            seq_lens_pre_verify,
            commit_lens,
            interval,
        )

    attn_backend.update_mamba_state_after_mtp_verify(
        last_correct_step_indices=last_correct_step_indices,
        mamba_track_indices=batch.mamba_track_indices,
        mamba_steps_to_track=mamba_steps_to_track,
        model=self.target_worker.model_runner.model,
    )


if getattr(DFlashWorker._update_target_mamba_state_after_verify, "_q38_patch_tag", None) != _PATCH_TAG:
    _fixed_update_target_mamba_state_after_verify._q38_patch_tag = _PATCH_TAG
    _fixed_update_target_mamba_state_after_verify._q38_parent = _orig
    DFlashWorker._update_target_mamba_state_after_verify = (
        _fixed_update_target_mamba_state_after_verify
    )

print(
    "[K100 TP4 DFlash Mamba verify-boundary-fix-v1] installed: "
    "post_verify=pre_verify+commit_lens; all non-Mamba/DFlash paths unchanged",
    flush=True,
)
