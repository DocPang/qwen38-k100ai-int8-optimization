ARG BASE_IMAGE=qwen38-k100ai-int8:unified-20260826-fa260728-q8split-rc2
FROM ${BASE_IMAGE}

COPY patch/runtime_patch_dflash_tp4_agent128k_v1/ /data/qwen38-dflash2-k100ai/runtime_patch_dflash_tp4_agent128k_v1/
COPY patch/runtime_patch_dflash_tp4_fa260728_rawq8_layout0_v1/ /data/qwen38-dflash2-k100ai/runtime_patch_dflash_tp4_fa260728_rawq8_layout0_v1/

RUN python3 -m py_compile \
      /data/qwen38-dflash2-k100ai/runtime_patch_dflash_tp4_agent128k_v1/sitecustomize.py \
      /data/qwen38-dflash2-k100ai/runtime_patch_dflash_tp4_fa260728_rawq8_layout0_v1/sitecustomize.py \
    && rm -f /data/qwen38-dflash2-k100ai/runtime_patch_tp4 \
    && ln -s /data/qwen38-dflash2-k100ai/runtime_patch_dflash_tp4_fa260728_rawq8_layout0_v1 \
             /data/qwen38-dflash2-k100ai/runtime_patch_tp4 \
    && test "$(readlink -f /data/qwen38-dflash2-k100ai/runtime_patch_tp4)" = \
            "/data/qwen38-dflash2-k100ai/runtime_patch_dflash_tp4_fa260728_rawq8_layout0_v1"

LABEL org.opencontainers.image.version="1.2.1" \
      local.qwen38.hotfix="tp4-fa260728-rawq8-layout0" \
      local.qwen38.change_scope="TP4 only; restore exact q8 verifier with corrected FA260728 layout ABI" \
      local.qwen38.base_image="qwen38-k100ai-int8:unified-20260826-fa260728-q8split-rc2"
