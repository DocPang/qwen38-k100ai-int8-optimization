ARG BASE_IMAGE=harbor.sourcefind.cn:5443/dcu/admin/base/custom@sha256:366525b25f452f85eb0ea5813604a64f03c648627bc824bb498b56cf5a325dde
FROM ${BASE_IMAGE}

COPY qwen38-k100ai-patchset.tar.gz /tmp/qwen38-k100ai-patchset.tar.gz
RUN mkdir -p /opt/qwen38-k100ai \
 && tar -xzf /tmp/qwen38-k100ai-patchset.tar.gz -C /opt/qwen38-k100ai \
 && /opt/qwen38-k100ai/install_into_image.sh \
 && rm -f /tmp/qwen38-k100ai-patchset.tar.gz

ENTRYPOINT ["/opt/qwen38-k100ai/start.sh"]
