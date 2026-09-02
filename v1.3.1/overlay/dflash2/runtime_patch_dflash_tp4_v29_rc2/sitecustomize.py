"""TP4 v29-rc2 release-functionality layer.

RC2 keeps the v29 FEATURES functionality but removes seed bookkeeping and
verify-wrapper work from ordinary unseeded/non-grammar batches. This is a
performance-only fast-path correction; special-request semantics are unchanged.

Parent: runtime_patch_dflash_tp4_prepare_native_i64_v29.

This layer closes three user-visible gaps without changing the proven v29
allocator/assignment path for ordinary requests:

1. DFlash grammar/tool_choice=required support for the spec-v1 linear chain.
   The target verify logits are constrained with the same grammar-object API used
   by normal SGLang sampling; committed tokens still advance the existing grammar
   state in DFlashVerifyInput.verify.
2. Per-request sampling_seed support without globally enabling deterministic
   inference.  Explicitly seeded batches receive SamplingBatchInfo.sampling_seed;
   DFlash rejection/final uniforms are derived from SGLang's murmur hash using
   request seed + logical sequence position + verify slot.  Ordinary unseeded
   batches retain the parent RNG path.
3. min_p support in DFlash non-greedy verification.  Only batches that actually
   request min_p (or deterministic seed) use the dense exact target-probability
   path.  Existing top-k/top-p-only requests keep the parent fast path.

Fail-closed rules:
- return_logprob remains rejected by the frozen DFlash scheduler validation;
- grammar + custom logit processor is rejected until ordering is separately
  audited (normal tool/JSON-schema grammar has no custom processor);
- all ordinary greedy/non-greedy requests fall through to the frozen parent.
"""

from __future__ import annotations

import hashlib
import os
import runpy
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


_ROOT = "/data/qwen38-dflash2-k100ai"
if os.getenv("SGLANG_Q38_TP124_V30_SKIP_TP4_PARENT", "0") != "1":
    runpy.run_path(
        f"{_ROOT}/runtime_patch_dflash_tp4_prepare_native_i64_v29_sitecustomize.py",
        run_name="__tp4_v29_features_v1_parent__",
    )

import sglang.srt.layers.sampler as _sampler
import sglang.srt.managers.scheduler as _scheduler
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo as _SamplingBatchInfo
from sglang.srt.speculative import dflash_info as _di
from sglang.srt.speculative import dflash_utils as _du
from sglang.srt.speculative import dflash_worker as _dw


# ---------------------------------------------------------------------------
# 1) Request admission: allow grammar requests into the DFlash worker.
# ---------------------------------------------------------------------------
_parent_validate_dflash_request = _scheduler.validate_dflash_request


def _validate_dflash_request_features(req):
    err = _parent_validate_dflash_request(req)
    if err is None:
        return None

    grammar_requested = (
        req.sampling_params.json_schema is not None
        or req.sampling_params.regex is not None
        or req.sampling_params.ebnf is not None
        or req.sampling_params.structural_tag is not None
    )
    if grammar_requested and "grammar-constrained" in str(err):
        # Custom processors can in principle overwrite -inf grammar logits after
        # masking.  Keep that un-audited combination fail-closed for this release.
        if getattr(req, "custom_logit_processor", None) is not None:
            return (
                "DFLASH grammar + custom logit processor is not release-audited; "
                "request rejected fail-closed."
            )
        return None
    return err


_scheduler.validate_dflash_request = _validate_dflash_request_features


# ---------------------------------------------------------------------------
# 2) Seed plumbing: populate SamplingBatchInfo only when a request explicitly
#    asks for a seed.  Unseeded-only batches preserve the parent fast path.
# ---------------------------------------------------------------------------
_parent_from_schedule_batch = _SamplingBatchInfo.from_schedule_batch.__func__
_parent_sampling_filter_batch = _SamplingBatchInfo.filter_batch
_parent_sampling_merge_batch = _SamplingBatchInfo.merge_batch


def _implicit_seed_for_req(req) -> int:
    cached = getattr(req, "_q38_implicit_sampling_seed", None)
    if cached is not None:
        return int(cached)
    # rid is already request-unique; hashing it gives all TP ranks the same stable
    # seed without coordinating a host RNG.  It changes across ordinary requests.
    rid = str(getattr(req, "rid", ""))
    digest = hashlib.blake2b(rid.encode("utf-8", "replace"), digest_size=8).digest()
    value = int.from_bytes(digest, "little") & ((1 << 63) - 1)
    setattr(req, "_q38_implicit_sampling_seed", value)
    return value


def _from_schedule_batch_with_explicit_seed(cls, batch, vocab_size: int):
    ret = _parent_from_schedule_batch(cls, batch, vocab_size)
    reqs = batch.reqs
    explicit = [r.sampling_params.sampling_seed for r in reqs]
    explicit_mask = [seed is not None for seed in explicit]

    # RC2 fast path: an ordinary batch must be indistinguishable from the frozen
    # v29 parent.  FEATURES-V1 allocated two device tensors and performed a GPU
    # .any().item() sync even when no request asked for a seed; that cost scales
    # with scheduler churn at c2/c3.  Do absolutely no seed bookkeeping here.
    if not any(explicit_mask):
        return ret

    values = [
        int(seed) if seed is not None else _implicit_seed_for_req(req)
        for req, seed in zip(reqs, explicit, strict=True)
    ]
    ret._q38_seed_values = torch.tensor(
        values,
        dtype=torch.int64,
        device=batch.device,
    )
    ret._q38_seed_explicit = torch.tensor(
        explicit_mask,
        dtype=torch.bool,
        device=batch.device,
    )
    # Explicit per-request seed is a batch-wide sampler field.  Unseeded rows in
    # the same batch receive stable filler seeds, preserving TP agreement while
    # retaining no deterministic guarantee for those rows.
    if ret.sampling_seed is None:
        ret.sampling_seed = ret._q38_seed_values
    return ret


def _filter_batch_seed_features(self, keep_indices, keep_indices_device):
    _parent_sampling_filter_batch(self, keep_indices, keep_indices_device)
    values = getattr(self, "_q38_seed_values", None)
    explicit = getattr(self, "_q38_seed_explicit", None)
    if values is None or explicit is None:
        return
    self._q38_seed_values = values[keep_indices_device]
    self._q38_seed_explicit = explicit[keep_indices_device]
    if bool(self._q38_seed_explicit.any().item()):
        self.sampling_seed = self._q38_seed_values
    elif not bool(getattr(self, "_q38_global_deterministic", False)):
        self.sampling_seed = None


def _mixed_unseeded_fill(info, *, device):
    # Only used when an ordinary batch is merged with a batch containing an
    # explicit seed.  Values merely need to agree across TP ranks; unseeded rows
    # intentionally carry no reproducibility contract.
    n = int(info.temperatures.shape[0])
    values = torch.arange(n, dtype=torch.int64, device=device) + 0x51A70000
    explicit = torch.zeros(n, dtype=torch.bool, device=device)
    return values, explicit


def _merge_batch_seed_features(self, other):
    self_values = getattr(self, "_q38_seed_values", None)
    self_explicit = getattr(self, "_q38_seed_explicit", None)
    other_values = getattr(other, "_q38_seed_values", None)
    other_explicit = getattr(other, "_q38_seed_explicit", None)

    # Pure ordinary merge: exactly the frozen parent path, with no allocations or
    # synchronisation added by this layer.
    if self_values is None and other_values is None:
        return _parent_sampling_merge_batch(self, other)

    if self_values is None:
        self_values, self_explicit = _mixed_unseeded_fill(
            self, device=other_values.device
        )
    if other_values is None:
        other_values, other_explicit = _mixed_unseeded_fill(
            other, device=self_values.device
        )

    _parent_sampling_merge_batch(self, other)
    self._q38_seed_values = torch.cat([self_values, other_values])
    self._q38_seed_explicit = torch.cat([self_explicit, other_explicit])
    if bool(self._q38_seed_explicit.any().item()):
        self.sampling_seed = self._q38_seed_values
    else:
        self.sampling_seed = None


_SamplingBatchInfo.from_schedule_batch = classmethod(
    _from_schedule_batch_with_explicit_seed
)
_SamplingBatchInfo.filter_batch = _filter_batch_seed_features
_SamplingBatchInfo.merge_batch = _merge_batch_seed_features


# The normal sampler's flashinfer complex-sampling branch rejects sampling_seed,
# and its PyTorch min_p+seed helper intentionally asserts.  Route only seeded
# complex batches through an exact deterministic torch implementation.  This is
# also used for the first token produced by target prefill, so the whole response
# honors the request seed rather than only later speculative rounds.
_parent_sample_from_probs = _sampler.Sampler._sample_from_probs


def _seeded_complex_sample(
    probs: torch.Tensor,
    sampling_info,
    positions: torch.Tensor,
) -> torch.Tensor:
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)

    ranks = torch.arange(probs.shape[-1], device=probs.device).view(1, -1)
    probs_sort[ranks >= sampling_info.top_ks.view(-1, 1)] = 0.0
    probs_sort[
        (probs_sum - probs_sort) > sampling_info.top_ps.view(-1, 1)
    ] = 0.0

    if sampling_info.need_min_p_sampling:
        min_p_thresholds = probs_sort[:, 0] * sampling_info.min_ps
        probs_sort[
            probs_sort < min_p_thresholds.view(-1, 1)
        ] = 0.0

    # Gumbel-max works with unnormalised positive weights: a per-row normalising
    # constant cancels from argmax.  Zero weights become -inf.
    log_weights = probs_sort.to(torch.float64)
    log_weights.log_()
    sampled_index = _sampler.multinomial_with_seed(
        log_weights,
        sampling_info.sampling_seed,
        positions,
    )
    probs_idx = probs_idx.to(torch.int32)
    return torch.gather(probs_idx, dim=1, index=sampled_index).view(-1)


def _sample_from_probs_features(
    self,
    probs: torch.Tensor,
    sampling_info,
    positions: torch.Tensor,
    simple_sampling_case: bool,
):
    if sampling_info.sampling_seed is None or simple_sampling_case:
        return _parent_sample_from_probs(
            self,
            probs,
            sampling_info,
            positions,
            simple_sampling_case,
        )
    return _seeded_complex_sample(probs, sampling_info, positions)


_sampler.Sampler._sample_from_probs = _sample_from_probs_features


# ---------------------------------------------------------------------------
# 3) Deterministic/min-p DFlash non-greedy verification.
# ---------------------------------------------------------------------------
_parent_compute_sampling = _di.compute_dflash_sampling_correct_drafts_and_bonus


def _hash_uniforms(
    *,
    seed: torch.Tensor,
    positions: torch.Tensor,
    draft_token_num: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # One extra hash column is reserved for the rejection/final draw.
    cols = torch.arange(draft_token_num + 1, device=seed.device, dtype=torch.int64)
    hashed = _sampler.murmur_hash32(
        seed.to(torch.uint64),
        positions.to(device=seed.device, dtype=torch.int64),
        cols,
    )
    # Convert uint32 -> open interval (0,1), avoiding exact 0/1 corner cases.
    uniforms = ((hashed.to(torch.float64) + 0.5) / float(1 << 32)).to(torch.float32)
    return uniforms[:, :draft_token_num].contiguous(), uniforms[:, draft_token_num].contiguous()


def _compute_sampling_features(
    *,
    candidates: torch.Tensor,
    next_token_logits: torch.Tensor,
    sampling_info,
    threshold_single: Optional[float] = None,
    threshold_acc: Optional[float] = None,
    uniform_samples: Optional[torch.Tensor] = None,
    uniform_samples_for_final_sampling: Optional[torch.Tensor] = None,
    use_sparse_topk: bool = True,
):
    seeded = sampling_info.sampling_seed is not None
    need_min_p = bool(getattr(sampling_info, "need_min_p_sampling", False))
    if not seeded and not need_min_p:
        return _parent_compute_sampling(
            candidates=candidates,
            next_token_logits=next_token_logits,
            sampling_info=sampling_info,
            threshold_single=threshold_single,
            threshold_acc=threshold_acc,
            uniform_samples=uniform_samples,
            uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
            use_sparse_topk=use_sparse_topk,
        )

    if not getattr(_du, "_DFLASH_SAMPLING_VERIFY_AVAILABLE", False):
        raise RuntimeError(
            "DFLASH non-greedy verification is unavailable on this build/device."
        )
    if candidates.ndim != 2 or next_token_logits.ndim != 2:
        raise ValueError("DFLASH seeded/min-p verify expects 2D candidates/logits")

    bs, draft_token_num = candidates.shape
    if next_token_logits.shape[0] != bs * draft_token_num:
        raise ValueError(
            "DFLASH seeded/min-p verify logits row mismatch: "
            f"expected={bs * draft_token_num} got={next_token_logits.shape[0]}"
        )

    if threshold_single is None or threshold_acc is None:
        from sglang.srt.server_args import get_global_server_args

        args = get_global_server_args()
        if threshold_single is None:
            threshold_single = args.speculative_accept_threshold_single
        if threshold_acc is None:
            threshold_acc = args.speculative_accept_threshold_acc
    threshold_single = float(threshold_single)
    threshold_acc = max(float(threshold_acc), 1e-9)

    device = next_token_logits.device
    if seeded and uniform_samples is None and uniform_samples_for_final_sampling is None:
        positions = getattr(sampling_info, "_q38_dflash_positions", None)
        if positions is None:
            raise RuntimeError(
                "seeded DFlash verify missing logical positions; feature wrapper invariant broken"
            )
        uniform_samples, uniform_samples_for_final_sampling = _hash_uniforms(
            seed=sampling_info.sampling_seed,
            positions=positions,
            draft_token_num=draft_token_num,
        )
    else:
        if uniform_samples is None:
            uniform_samples = torch.rand(
                (bs, draft_token_num), dtype=torch.float32, device=device
            )
        else:
            uniform_samples = uniform_samples.to(device=device, dtype=torch.float32)
        if uniform_samples_for_final_sampling is None:
            uniform_samples_for_final_sampling = torch.rand(
                (bs,), dtype=torch.float32, device=device
            )
        else:
            uniform_samples_for_final_sampling = uniform_samples_for_final_sampling.to(
                device=device, dtype=torch.float32
            )

    expanded_temperature = torch.repeat_interleave(
        sampling_info.temperatures, draft_token_num, dim=0
    )
    scaled_logits = next_token_logits / expanded_temperature
    target_probs = F.softmax(scaled_logits, dim=-1)

    if bool(getattr(sampling_info, "need_top_k_sampling", True)):
        target_probs = _du.top_k_renorm_prob(
            target_probs,
            torch.repeat_interleave(sampling_info.top_ks, draft_token_num, dim=0),
        )
    if bool(getattr(sampling_info, "need_top_p_sampling", False)):
        target_probs = _du.top_p_renorm_prob(
            target_probs,
            torch.repeat_interleave(sampling_info.top_ps, draft_token_num, dim=0),
        )
    if need_min_p:
        repeated_min_ps = torch.repeat_interleave(
            sampling_info.min_ps, draft_token_num, dim=0
        )
        max_probs = target_probs.max(dim=-1, keepdim=True).values
        keep = target_probs >= (max_probs * repeated_min_ps[:, None])
        target_probs = target_probs * keep
        denom = target_probs.sum(dim=-1, keepdim=True).clamp_min_(1e-20)
        target_probs = target_probs / denom

    target_probs = target_probs.view(bs, draft_token_num, -1).contiguous()
    draft_probs = torch.zeros_like(target_probs)

    (
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        predicts,
        accept_index,
        accept_token_num,
    ) = _du._get_or_create_chain_verify_buffers(
        bs=bs,
        draft_token_num=draft_token_num,
        device=device,
    )
    candidates_i64 = (
        candidates if candidates.dtype == torch.int64 else candidates.to(torch.int64)
    )
    _du.tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates_i64,
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


_di.compute_dflash_sampling_correct_drafts_and_bonus = _compute_sampling_features
_du.compute_dflash_sampling_correct_drafts_and_bonus = _compute_sampling_features


# ---------------------------------------------------------------------------
# 4) Grammar mask for the spec-v1 linear DFlash chain.
# ---------------------------------------------------------------------------
_parent_prepare_spec = _dw.DFlashWorker._prepare_for_speculative_decoding
_parent_verify = _di.DFlashVerifyInput.verify  # v29 verify-scoped wrapper stays parent
_selector_rng_contract = (
    3 if hasattr(_dw.DFlashWorker, "_tp_synced_uniform") else 0
)
_grammar_hits = 0
_seed_hits = 0
_min_p_hits = 0


def _prepare_spec_features(self, batch, draft_input):
    grammar_requested = bool(getattr(batch, "has_grammar", False))
    sampling_info = getattr(batch, "sampling_info", None)
    seeded_sampling = bool(
        sampling_info is not None
        and sampling_info.sampling_seed is not None
        and not sampling_info.is_all_greedy
    )

    if not grammar_requested and not seeded_sampling:
        return _parent_prepare_spec(self, batch, draft_input)

    # The frozen worker only rejects grammar at its entry. Its proposal/verify
    # construction is otherwise grammar-agnostic, so bypass only that invariant.
    if grammar_requested:
        batch.has_grammar = False

    # v1.2.2 selector sampling draws three TP-synchronised RNG tensors during
    # prepare: proposal, acceptance, final. For explicit sampling_seed requests,
    # replace those torch.rand draws with SGLang's stateless murmur hash keyed by
    # (request seed, logical seq position, stage/slot). This keeps cold/cached and
    # TP-rank execution deterministic without changing unseeded sampling.
    had_instance_uniform = "_tp_synced_uniform" in self.__dict__
    old_instance_uniform = self.__dict__.get("_tp_synced_uniform")
    rng_call = 0

    if seeded_sampling:
        seed = sampling_info.sampling_seed.detach().to(
            device=batch.seq_lens.device, dtype=torch.int64
        )
        logical_positions = batch.seq_lens.detach().to(
            device=batch.seq_lens.device, dtype=torch.int64
        )
        if seed.numel() != logical_positions.numel():
            raise RuntimeError(
                "seeded DFlash selector seed/position size mismatch: "
                f"seed={seed.numel()} pos={logical_positions.numel()}"
            )

        def _deterministic_tp_uniform(shape, *, device):
            nonlocal rng_call
            stage = rng_call
            rng_call += 1
            if stage >= 3:
                raise RuntimeError(
                    "seeded DFlash selector requested more than three RNG tensors; "
                    "runtime contract changed"
                )

            if len(shape) == 1:
                rows = int(shape[0])
                width = 1
            elif len(shape) == 2:
                rows = int(shape[0])
                width = int(shape[1])
            else:
                raise RuntimeError(
                    f"unexpected seeded DFlash RNG shape={tuple(shape)}"
                )
            if rows != int(seed.numel()):
                raise RuntimeError(
                    "seeded DFlash RNG batch mismatch: "
                    f"shape={tuple(shape)} seed_rows={seed.numel()}"
                )

            stage_base = (0, 64, 128)[stage]
            cols = torch.arange(
                stage_base,
                stage_base + width,
                dtype=torch.int64,
                device=device,
            )
            hashed = _sampler.murmur_hash32(
                seed.to(device=device, dtype=torch.uint64),
                logical_positions.to(device=device, dtype=torch.int64),
                cols,
            )
            out = ((hashed.to(torch.float64) + 0.5) / float(1 << 32)).to(
                torch.float32
            )
            return out[:, 0] if len(shape) == 1 else out

        # v1.2.2's non-greedy selector draws proposal/accept/final uniforms
        # through this hook.  The frozen TP2 champion overlay predates that
        # selector path: it proposes a deterministic greedy lattice and draws
        # no selector uniforms, while target verification below still receives
        # request-seeded hash uniforms.  Supporting both contracts preserves
        # the old TP2 numerical base without weakening the newer fail-closed
        # three-draw contract.
        if _selector_rng_contract == 3:
            self._tp_synced_uniform = _deterministic_tp_uniform

    try:
        out = _parent_prepare_spec(self, batch, draft_input)
        if seeded_sampling and rng_call != _selector_rng_contract:
            raise RuntimeError(
                "seeded DFlash selector RNG contract mismatch: "
                f"expected={_selector_rng_contract} observed={rng_call}"
            )
        return out
    finally:
        if seeded_sampling and _selector_rng_contract == 3:
            if had_instance_uniform:
                self._tp_synced_uniform = old_instance_uniform
            else:
                self.__dict__.pop("_tp_synced_uniform", None)
        if grammar_requested:
            batch.has_grammar = True


def _token_allowed(mask_row: torch.Tensor, token_id: int, vocab_size: int) -> bool:
    if token_id < 0 or token_id >= vocab_size:
        return False
    word = int(mask_row[token_id // 32].item())
    return (word & (1 << (token_id % 32))) != 0


def _build_linear_grammar_mask(batch, draft_tokens_2d: torch.Tensor, vocab_size: int):
    bs, chain_len = draft_tokens_2d.shape
    draft_cpu = draft_tokens_2d.detach().to("cpu", dtype=torch.int64)
    vocab_mask = None
    grammar_handle = None

    for i, req in enumerate(batch.reqs):
        grammar = getattr(req, "grammar", None)
        if grammar is None:
            continue
        if vocab_mask is None:
            vocab_mask = grammar.allocate_vocab_mask(
                vocab_size=vocab_size,
                batch_size=bs * chain_len,
                device="cpu",
            )
            grammar_handle = grammar

        base = i * chain_len
        accepted = 0
        try:
            if not grammar.is_terminated():
                grammar.fill_vocab_mask(vocab_mask, base)
            else:
                continue

            for j in range(1, chain_len):
                token_id = int(draft_cpu[i, j].item())
                if not _token_allowed(vocab_mask[base + j - 1], token_id, vocab_size):
                    break
                grammar.accept_token(token_id)
                accepted += 1
                if grammar.is_terminated():
                    break
                grammar.fill_vocab_mask(vocab_mask, base + j)
        finally:
            if accepted:
                grammar.rollback(accepted)

    return grammar_handle, vocab_mask


def _verify_features(self, *, batch, logits_output, page_size: int):
    global _grammar_hits, _seed_hits, _min_p_hits
    sampling_info = batch.sampling_info
    seeded = sampling_info is not None and sampling_info.sampling_seed is not None
    need_min_p = sampling_info is not None and bool(
        getattr(sampling_info, "need_min_p_sampling", False)
    )
    has_grammar = bool(getattr(batch, "has_grammar", False))

    # RC2 hot path: ordinary greedy/non-greedy requests bypass this layer before
    # any tensor operation, counter bookkeeping, or logging condition.
    if not seeded and not need_min_p and not has_grammar:
        return _parent_verify(
            self,
            batch=batch,
            logits_output=logits_output,
            page_size=page_size,
        )

    if seeded:
        sampling_info._q38_dflash_positions = batch.seq_lens.detach().to(
            device=logits_output.next_token_logits.device,
            dtype=torch.int64,
        )
        _seed_hits += 1

    if need_min_p:
        _min_p_hits += 1

    if has_grammar:
        if sampling_info is None:
            raise RuntimeError("DFLASH grammar verify requires sampling_info")
        grammar, vocab_mask = _build_linear_grammar_mask(
            batch,
            self.draft_token.view(batch.batch_size(), self.draft_token_num),
            int(sampling_info.vocab_size),
        )
        if vocab_mask is not None:
            vocab_mask = vocab_mask.to(
                logits_output.next_token_logits.device,
                non_blocking=True,
            )
            grammar.apply_vocab_mask(
                logits=logits_output.next_token_logits,
                vocab_mask=vocab_mask,
            )
            if hasattr(sampling_info, "grammar_mask"):
                sampling_info.grammar_mask = None
            _grammar_hits += 1

    out = _parent_verify(
        self,
        batch=batch,
        logits_output=logits_output,
        page_size=page_size,
    )

    total = _grammar_hits + _seed_hits + _min_p_hits
    if total <= 12 or (total and total % 128 == 0):
        print(
            "[K100 TP4 v29-rc2] ACTIVE "
            f"grammar={_grammar_hits} seed={_seed_hits} min_p={_min_p_hits}",
            flush=True,
        )
    return out


_dw.DFlashWorker._prepare_for_speculative_decoding = _prepare_spec_features
_di.DFlashVerifyInput.verify = _verify_features

print(
    "[K100 TP4 v29-rc2] installed: grammar/tool-required + explicit "
    "sampling_seed + DFlash min_p; ordinary v29 path unchanged",
    flush=True,
)
