# v1.3.0 / v30 production profile kit

This directory contains the exact source overlay, launch contracts, idle-cache guard, and production manifests used for the 2026-09-01 TP1/TP2 v30 release. TP4 remains the already-final v30 service and is not changed by this kit.

## Release scope

- TP1: `Final-v2 -> Mamba checkpoint8192 -> Hybrid14 -> v30 common`, one K100AI, `max-running=1`, `mem_fraction_static=0.84`.
- TP2: legacy SourceFind image -> longtail-v1 / frozen `{3,23}` -> Mamba checkpoint8192 -> v30 common, two K100AI, `max-running=4`, `mem_fraction_static=0.88`.
- TP4: no deployment change; its existing v30 FINAL manifest is included in the release evidence.
- Cache-resume: exact cold determinism and cached output-ID equality at 16K, 64K, and 139,265.
- API/functionality: greedy, sampling, min-p, seed, JSON schema, regex, required/named tool, mixed c3, streaming and non-stream disconnect recovery.
- Quality: no-thinking arithmetic remains the frozen 18/20 baseline with only case 8/17; thinking arithmetic is 20/20.

## Base requirements

This is a production overlay for the existing K100AI v1.2.2 workspace layout, not a second 6.5 GiB image download.

- TP1 base image must resolve to `sha256:ad30e85d745574295921f677054bebebee57f8beb444680205ac5fd5d5e05e0c`.
- TP2 intentionally uses the legacy SourceFind image at `sha256:5d6305a6fb1695ebcb3675a7f9b87aca59478aaae21c9eeda3ebb59ddb5f9ad8`; do not switch TP2 to the unified image because that build regressed the frozen arithmetic gate.
- Default roots are `/data/qwen38-dflash2-k100ai` and `/data/qwen38-27b-k100ai-int8-opt`.
- The TP2 launcher also expects the existing SourceFind SGLang overlay at `work/sourcefind_sglang_overlay_tp4/python`. The launcher fails closed if it is absent.

## Install without clobbering local work

```bash
cd v1.3.0
sha256sum -c SHA256SUMS
bash install_overlay.sh
```

`install_overlay.sh` refuses to overwrite any divergent directory or file. Existing byte-identical files are retained. Keep the previous service/container as the rollback path until the new stable-name container passes its release gates.

The launchers are host-specific reference contracts: verify GPU/render mappings and ports before use. The production mapping was GPU2/8042 for TP1 and GPU0-1/8062 for TP2.

## Idle-cache guard

The included guard is external to SGLang. It only calls the official `/flush_cache` endpoint after `running=0`, `waiting=0`, and `used_tokens=0`; request activity or `/v1/loads` timeout resets the timer and never flushes.

The validated production policy is:

- emergency: HBM >= 99% and fully idle for 15 seconds;
- normal: HBM >= 97% and fully idle for 600 seconds.

Review the unit paths, then install the templates under `systemd/` if desired. The model containers do not depend on the guard service.

## Evidence

Machine-readable manifests and gates are in [`../results/v1.3.0`](../results/v1.3.0). The original 24-file evidence archive is [`../results/qwen38_k100ai_v130_release_evidence_20260901.tar.gz`](../results/qwen38_k100ai_v130_release_evidence_20260901.tar.gz), SHA256 `a4b46213ba8f856a1676ffac5179bf376a2c0ad455dc49916c081a08162af919`.

Performance caveat: the complete same-build formal10 files are published intact. Targeted 128K/257.9K cold repeats demonstrate that the low TP1 formal10 tail was an occasional runtime state, but the release does not construct a cross-run best-of curve.
