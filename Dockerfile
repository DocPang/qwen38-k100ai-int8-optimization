ARG BASE_IMAGE=harbor.sourcefind.cn:5443/dcu/admin/base/custom@sha256:366525b25f452f85eb0ea5813604a64f03c648627bc824bb498b56cf5a325dde
FROM ${BASE_IMAGE}

COPY qwen38-k100ai-patchset.tar.gz /tmp/qwen38-k100ai-patchset.tar.gz
RUN mkdir -p /opt/qwen38-k100ai \
 && tar -xzf /tmp/qwen38-k100ai-patchset.tar.gz -C /opt/qwen38-k100ai \
 && /opt/qwen38-k100ai/install_into_image.sh \
 && rm -f /tmp/qwen38-k100ai-patchset.tar.gz

# Default: use the seven prebuilt userspace HIP extensions validated against
# the pinned SourceFind image / DTK stack.
COPY native_ext/prebuilt/ /data/qwen38-27b-k100ai-int8-opt/native_ext/

# Optional source rebuild: build_native.sh writes .so files here. In a normal
# fresh clone this directory only contains .gitkeep, so prebuilt binaries remain.
COPY .build/native/ /data/qwen38-27b-k100ai-int8-opt/native_ext/

RUN test -s /data/qwen38-27b-k100ai-int8-opt/native_ext/k100_int8_gemv_v7_sglang.so \
 && test -s /data/qwen38-27b-k100ai-int8-opt/native_ext/k100_int8_gemv_generic_v2_sglang.so \
 && test -s /data/qwen38-27b-k100ai-int8-opt/native_ext/k100_int8_gemv_k5120_full5_sglang.so \
 && test -s /data/qwen38-27b-k100ai-int8-opt/native_ext/k100_int8_gemv_k5120_ldsx_v1_sglang.so \
 && test -s /data/qwen38-27b-k100ai-int8-opt/native_ext/k100_int8_gemv_deep_v4_sglang.so \
 && test -s /data/qwen38-27b-k100ai-int8-opt/native_ext/k100_int8_gemv_tp2_row_ldsx_v1_sglang.so \
 && test -s /data/qwen38-27b-k100ai-int8-opt/native_ext/k100_int8_gemv_tp4_row_ldsx_v1_sglang.so

ENTRYPOINT ["/opt/qwen38-k100ai/start.sh"]
