# v1.3.1 Release Notes — v30 final unified image

发布日期：2026-09-02

v1.3.1 是当前 v30 正式版。相比 v1.3.0，本次不只是更新 overlay：**已经重新制作并完整验证一个最终 Docker 镜像，TP1 / TP2 / TP4 全部从同一镜像启动。**

## 最终镜像

夸克网盘：

**https://pan.quark.cn/s/2e35321f6278?pwd=gPVV**

提取码：`gPVV`

文件：

```text
Qwen3.8-K100AI-v1.3.1-final-image.tar.zst
```

SHA256：

```text
4e43edd8a0cf5ee0e501aefb051170587b1da3003392fdd2d32a1d9712e11d8f
```

导入后：

```text
qwen38-k100ai-int8:v1.3.1
```

Image ID：

```text
sha256:534f9512a3d5217c8f65f11183ed75db19e7aa3df1ecc4da1ba993e98793973d
```

压缩包约 6.3 GiB，Docker archive 约 33.3 GiB。

## 1. TP1 / TP2 / TP4 真正统一为一个成品镜像

使用：

```text
PROFILE=tp1
PROFILE=tp2
PROFILE=tp4
```

切换三种拓扑。

v1.3.0 的 TP2 correctness 合同要求 legacy SourceFind flash-attn 260602，而 TP1 / TP4 使用 260728。为了既保持 correctness 又只发布一个镜像，v1.3.1 最终镜像采用 side-by-side 方案：

- 系统默认 flash-attn：260728，供 TP1 / TP4 使用；
- 镜像额外内置 TP2 legacy flash-attn 260602 sidecar；
- `PROFILE=tp2` 时自动把 legacy sidecar 放到 Python 路径最前；
- 启动时 fail-closed 校验 `flash_attn` 与 `flash_attn_2_cuda` 确实来自 TP2 sidecar。

因此新用户**不再需要单独下载 TP2 legacy Docker 镜像，也不需要安装 v1.3.0 overlay 才能使用 TP2**。

## 2. 显存 high-water 改成源码级根修

v1.3.0 使用 external `idle_cache_flush_guard.py` 在完全 idle 后调用 `/flush_cache`。后续隔离实验确认，长期增长的主体不是 KV / Radix / Mamba 逻辑泄漏，而是 PyTorch / HIP caching allocator 保留的 transient workspace high-water。

SGLang 0.5.12 官方 idle trim 原实现又只挂在 rank0 `IdleSleeper` 上，因此 TP>1 时只有 rank0 GPU 会清 allocator。

v1.3.1 根修为：

- CPU/ZMQ sleep 继续由 rank0 处理；
- GPU allocator trim 移到所有 scheduler rank 都会执行的 idle housekeeping；
- 每个 rank 调自己的 `empty_device_cache(self.device_module)`；
- production 默认 `SGLANG_EMPTY_CACHE_INTERVAL=60`；
- 不调用 full `/flush_cache`，不清 prefix / Radix / KV / Mamba 逻辑缓存。

最终验证：

- TP1：长请求后 HBM 98% -> 94%；
- TP2：rank0 + rank1 均独立 trim；
- TP4：rank0 / 1 / 2 / 3 全部独立 trim；
- cache-resume 保持正常。

正式版不再启用旧 external guard。

## 3. 最终启动参数

| Profile | 默认端口 | context | mem | max-running | chunked prefill | Mamba interval |
|---|---:|---:|---:|---:|---:|---:|
| TP1 | 8042 | 262144 | 0.84 | 1 | 8192 | 8192 |
| TP2 | 8062 | 262144 | 0.88 | 4 | 8192 | 8192 |
| TP4 | 8068 | 262144 | 0.95 | 8 | 16384 | 16384 |

TP4 另固定：

```text
MAX_TOTAL_TOKENS=1048576
PP_MAX_MICRO_BATCH_SIZE=8
CUDA_GRAPH_BS=1 2 3 4 5 6 7 8
```

## 4. GitHub launcher 改成最终镜像直启

`scripts/launch.sh` 现在直接启动：

```text
qwen38-k100ai-int8:v1.3.1
```

普通用户不再需要 `install_overlay.sh`。

示例：

```bash
export TARGET_MODEL=/path/to/target
export DRAFT_MODEL=/path/to/draft

HOST_GPU_IDS="0 1 2 3" \
RENDER_DEVICES="/dev/dri/renderD128 /dev/dri/renderD129 /dev/dri/renderD130 /dev/dri/renderD131" \
./scripts/launch.sh tp4
```

常用参数可以通过环境变量改：

```bash
PORT=9000 \
SERVED_MODEL_NAME=my-qwen38 \
CONTEXT_LENGTH=245760 \
MAX_RUNNING_REQUESTS=4 \
./scripts/launch.sh tp4
```

任意 SGLang 参数放到 `--` 后：

```bash
./scripts/launch.sh tp4 -- \
  --enable-deterministic-inference \
  --log-level info
```

最终镜像入口会用 JSON/base64 安全通道把参数传给对应 profile entrypoint，避免 shell 二次拆词。

`install_overlay.sh` 与源码 patch 仍保留，只用于审计、研究和手工复现，不是最终镜像用户的安装步骤。

## 5. Final-image correctness / Agent gate

发布前不是只测“镜像能启动”，而是在最终 image 上重新跑了关键 authority gate。

### TP1

- GPU2 authority 64K cache-resume：PASS；
- cold×2 deterministic；
- cached=57,344 / replay=8,192；
- 请求结束 HBM 自动回约 94%。

### TP2

- function matrix：9/9 PASS；
- mixed c4：PASS；
- thinking arithmetic：20/20；
- no-thinking arithmetic：保持历史 18/20，仅 case 8 / 17，无新增回归；
- 16K cache-resume：PASS；
- 64K cache-resume：PASS；
- rank0 / rank1 allocator trim：PASS。

### TP4

- function matrix：9/9 PASS；
- mixed c8：PASS；
- release-authority 64K cache-resume：PASS；
- rank0 / 1 / 2 / 3 allocator trim：PASS；
- 最终镜像关键 runtime 与现网 TP4 production 逐 SHA 对账通过。

### 镜像归档

发布前实际完成：

1. `docker save`；
2. zstd 压缩；
3. `zstd -t`；
4. 删除本地 image tag；
5. `zstd -dc | docker load`；
6. load 后 Image ID 与原始 image 完全一致；
7. Mac 下载后再次计算 SHA256 与服务器一致。

## 6. Agent 建议

主 Agent 推荐 TP4。

建议：

```text
soft compaction: ~230K
hard compaction: ~240K
parallel_tool_calls: false
```

规划 / 分析轮：`enable_thinking=true`。

工具执行轮，尤其 write / patch / shell / 大 JSON：`enable_thinking=false`。

TP2 长字符串 tool argument fresh gate：

- 1K chars：PASS；
- 2K chars：PASS；
- 4K chars：会开始重复并撞生成上限。

因此 coding Agent 推荐 `apply_patch` / chunked write，而不是一次 tool call 重写整个大文件。

当前 frozen API authority 仍以 `/v1/chat/completions` 为主；`parallel_tool_calls=true` 与 `/v1/responses` 不作为本版 frozen release authority。

## 7. 升级方式

新部署：直接下载并导入 v1.3.1 最终镜像，不再需要 v1.2.2 镜像。

```bash
zstd -dc Qwen3.8-K100AI-v1.3.1-final-image.tar.zst | docker load
```

旧 v1.3.0 / v1.2.2 用户：Target / Draft 权重可以原样复用，只需要换镜像和启动脚本。

如果旧 external guard 曾启用：

```bash
sudo systemctl disable --now q38-idle-flush-tp1.service 2>/dev/null || true
sudo systemctl disable --now q38-idle-flush-tp2.service 2>/dev/null || true
```

## 8. 回滚

旧 tag / v1.3.0 发布文件仍保留，可按历史版本说明回滚。

v1.3.1 的 TP2 legacy compatibility 已封装在最终镜像内部，因此不要再在最终镜像外手工覆盖 flash-attn 或 SourceFind overlay。

## 发布文件

- [`README.md`](README.md)
- [`VALIDATION.md`](VALIDATION.md)
- [`RELEASE_MANIFEST.json`](RELEASE_MANIFEST.json)
- [`IMAGE_SHA256SUMS`](IMAGE_SHA256SUMS)
- [`scripts/`](scripts/)
- [`rootfix/`](rootfix/)

完整部署方式见 [`README.md`](README.md)。
