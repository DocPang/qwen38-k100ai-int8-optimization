"""TP4 Agent/128K production composition: BA24 target + DFlash2 q8 verifier.

The target side inherits the proven corrected TP4 BA24 stack from the main
Qwen3.8 W8A8 project (correct gfx928 prefill consumer, TP4 row/rank-local GEMV,
RMS->QKVZ producer and BA24 decode path). DFlash2 verification normally emits
q=8; on gfx928 the vendor q>=5 paged-varlen kernel is a no-write path, while
q<=4 is proven correct. Therefore only exact batch1 causal q=8 verification is
split into two native q=4 calls. All other shapes retain the BA24 parent's
correct routing.

TorchInductor/resource-tracker/ninja helper interpreters must not import the
full model patch stack. This mirrors the already validated TP8 helper bypass
and removes multi-minute recursive sitecustomize work during cold startup.
"""
from __future__ import annotations

import os

_ROOT = "/data/qwen38-dflash2-k100ai"
_TARGET_ROOT = "/data/qwen38-27b-k100ai-int8-opt"

try:
    _cmd = (
        open("/proc/self/cmdline", "rb")
        .read()
        .replace(b"\0", b" ")
        .decode("utf-8", "replace")
    )
except Exception:
    _cmd = ""

_helper = (
    ("cdll.LoadLibrary" in _cmd and "torchinductor_tp4" in _cmd)
    or "multiprocessing.resource_tracker" in _cmd
    or "/usr/local/bin/ninja" in _cmd
    or " ninja --version" in _cmd
)


def _install() -> None:
    import importlib.metadata as _metadata
    import runpy

    runpy.run_path(
        f"{_TARGET_ROOT}/runtime_patch_sglang_tp4_u036_rmsqkvz_ba24_v1/sitecustomize.py",
        run_name="__q38_tp4_ba24_parent_for_dflash2_agent128k__",
    )

    import torch
    import flash_attn_2_cuda as _fa2
    from sglang.srt.layers.attention import flashattention_backend as _fab
    from sglang.srt.layers.attention import flashattention_interface as _fai

    parent = _fab.vllm_flash_attn_varlen_func
    native = _fai.vllm_flash_attn_varlen_func_interface
    native_q8_paged = os.getenv("SGLANG_DFLASH2_Q8_NATIVE_PAGED", "0").strip().lower() in (
        "1", "true", "yes", "on"
    )
    raw_paged_layout: int | None = None
    if native_q8_paged:
        flash_attn_version = _metadata.version("flash-attn")
        legacy_abi = "2.8.3+das.opt1.dtk2604.torch290.2606021702.ge93bd4"
        layout_abi = "2.8.3+das.opt1.dtk2604.torch290.2607280958.gebb4be"
        if flash_attn_version == legacy_abi:
            raw_paged_layout = None
        elif flash_attn_version == layout_abi:
            # SourceFind 260728 adds a final paged_attention layout selector.
            # The shipped vllm_flash_attn_varlen_func passes layout=0 for the
            # audited paged KV cache geometry [page, H, S, D].  Isolated TP4
            # q8 gates at 16K/64K/128K/257.9K prove layout=0 is bitwise equal
            # to the exact 2xq4 verifier; layout=1 is numerically wrong here.
            raw_paged_layout = 0
        else:
            raise RuntimeError(
                "TP4 raw-q8 paged verifier has no audited flash-attn ABI for "
                f"version={flash_attn_version}"
            )
        print(
            f"[K100 DFlash2 TP4 raw-q8 ABI] flash-attn={flash_attn_version} "
            f"layout_arg={'legacy' if raw_paged_layout is None else raw_paged_layout}",
            flush=True,
        )
    seen = False
    cuq_cache: dict[tuple[int | None, int], torch.Tensor] = {}

    def cuq(device: torch.device, qlen: int) -> torch.Tensor:
        key = (device.index, qlen)
        t = cuq_cache.get(key)
        if t is None:
            t = torch.tensor([0, qlen], device=device, dtype=torch.int32)
            cuq_cache[key] = t
        return t

    def native_call(
        *,
        q,
        k,
        v,
        qlen,
        seqused_k,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size,
        block_table,
        fa_version,
        q_descale,
        k_descale,
        v_descale,
    ):
        return native(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cuq(q.device, qlen),
            max_seqlen_q=qlen,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            block_table=block_table,
            fa_version=fa_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )

    def dflash_q8_split_native_else_parent(
        q,
        k,
        v,
        cu_seqlens_q,
        max_seqlen_q,
        seqused_k,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size,
        block_table,
        fa_version,
        q_descale,
        k_descale,
        v_descale,
    ):
        nonlocal seen
        qlen = int(max_seqlen_q)
        batch = int(cu_seqlens_q.numel() - 1)
        eligible = (
            qlen == 8
            and int(q.shape[0]) == 8
            and batch == 1
            and int(seqused_k.numel()) == 1
            and bool(causal)
            and window_size == (-1, -1)
            and block_table is not None
        )
        if not eligible:
            return parent(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                seqused_k=seqused_k,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                block_table=block_table,
                fa_version=fa_version,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
            )

        if native_q8_paged:
            exact_native_q8 = (
                q.dtype == torch.bfloat16
                and k.dtype == torch.bfloat16
                and v.dtype == torch.bfloat16
                and tuple(q.shape) == (8, 6, 256)
                and int(k.ndim) == 4
                and int(v.ndim) == 4
                and int(k.shape[1]) == 1
                and int(v.shape[1]) == 1
                and int(k.shape[2]) == 64
                and int(v.shape[3]) == 64
                and int(k.shape[3]) == 256
                and int(v.shape[2]) == 256
                and int(fa_version) == 2
            )
            if not exact_native_q8:
                raise RuntimeError(
                    "DFlash2 native q8 paged verifier requested outside audited TP4 geometry"
                )
            if not seen:
                seen = True
                print(
                    "[K100 DFlash2 TP4 Agent128K] ACTIVE q=8 verifier -> single raw paged_attention",
                    flush=True,
                )
            out = torch.empty_like(q)
            paged_args = (
                out,
                q.reshape(1, 8, 6, 256),
                k,
                v,
                softmax_scale,
                block_table,
                seqused_k,
                None,
                "auto",
                q_descale,
                k_descale,
                v_descale,
                max_seqlen_k,
                None,
            )
            if raw_paged_layout is None:
                _fa2.paged_attention(*paged_args)
            else:
                _fa2.paged_attention(*paged_args, raw_paged_layout)
            return out

        if not seen:
            seen = True
            print(
                "[K100 DFlash2 TP4 Agent128K] ACTIVE q=8 verifier -> 2x native q=4",
                flush=True,
            )

        # Original q=8 corresponds to absolute positions [K-8, ..., K-1].
        # The first q4 must see K-4 keys; the second q4 sees all K keys.
        seq_k_first = seqused_k - 4
        out0 = native_call(
            q=q[:4],
            k=k,
            v=v,
            qlen=4,
            seqused_k=seq_k_first,
            max_seqlen_k=max(1, int(max_seqlen_k) - 4),
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            block_table=block_table,
            fa_version=fa_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        out1 = native_call(
            q=q[4:],
            k=k,
            v=v,
            qlen=4,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            block_table=block_table,
            fa_version=fa_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        if isinstance(out0, (tuple, list)) or isinstance(out1, (tuple, list)):
            raise RuntimeError("DFlash2 TP4 q8split expected tensor outputs")
        return torch.cat((out0, out1), dim=0)

    _fai.vllm_flash_attn_varlen_func = dflash_q8_split_native_else_parent
    _fab.vllm_flash_attn_varlen_func = dflash_q8_split_native_else_parent
    print(
        "[K100 DFlash2 TP4 Agent128K] installed: BA24 target + exact q8 verifier; "
        f"native_q8_paged={native_q8_paged}; radix/KV policy is launcher-owned",
        flush=True,
    )


if _helper:
    print(f"[K100 DFlash2 TP4 helper bypass] pid={os.getpid()}", flush=True)
else:
    _install()
