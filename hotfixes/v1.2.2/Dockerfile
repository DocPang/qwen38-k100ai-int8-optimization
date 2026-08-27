ARG BASE_IMAGE=qwen38-k100ai-int8:unified-20260827-v1.2.1
FROM ${BASE_IMAGE}

COPY patch/sglang/srt/speculative/dflash_worker.py /usr/local/lib/python3.10/dist-packages/sglang/srt/speculative/dflash_worker.py
COPY patch/sglang/srt/speculative/dflash_info.py /usr/local/lib/python3.10/dist-packages/sglang/srt/speculative/dflash_info.py
COPY patch/sglang/srt/speculative/dflash_utils.py /usr/local/lib/python3.10/dist-packages/sglang/srt/speculative/dflash_utils.py

RUN python3 -m py_compile \
      /usr/local/lib/python3.10/dist-packages/sglang/srt/speculative/dflash_worker.py \
      /usr/local/lib/python3.10/dist-packages/sglang/srt/speculative/dflash_info.py \
      /usr/local/lib/python3.10/dist-packages/sglang/srt/speculative/dflash_utils.py \
    && grep -q "compute_dflash_selector_sampling_correct_drafts_and_bonus" \
      /usr/local/lib/python3.10/dist-packages/sglang/srt/speculative/dflash_utils.py \
    && ! grep -q "K100AI DFlash2 backport currently supports greedy decoding only" \
      /usr/local/lib/python3.10/dist-packages/sglang/srt/speculative/dflash_worker.py

LABEL org.opencontainers.image.version="1.2.2" \
      local.qwen38.hotfix="dflash2-non-greedy-v1" \
      local.qwen38.change_scope="TP4 DFlash2 non-greedy sampling; greedy path preserved" \
      local.qwen38.base_image="qwen38-k100ai-int8:unified-20260827-v1.2.1"
