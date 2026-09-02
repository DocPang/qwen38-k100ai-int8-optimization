# Qwen3.8 K100AI v1.3.1 / v30

v1.3.1 是当前正式版本。最终发布已经从“v1.2.2 基础镜像 + overlay”升级为 **一个完整 Docker 镜像**，TP1 / TP2 / TP4 都从同一镜像启动。

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

镜像：

```text
qwen38-k100ai-int8:v1.3.1
```

Image ID：

```text
sha256:534f9512a3d5217c8f65f11183ed75db19e7aa3df1ecc4da1ba993e98793973d
```

压缩包约 6.3 GiB，Docker archive 约 33.3 GiB。

导入：

```bash
IMAGE_ARCHIVE=Qwen3.8-K100AI-v1.3.1-final-image.tar.zst

echo "4e43edd8a0cf5ee0e501aefb051170587b1da3003392fdd2d32a1d9712e11d8f  $IMAGE_ARCHIVE" | sha256sum -c -
zstd -t "$IMAGE_ARCHIVE"
zstd -dc "$IMAGE_ARCHIVE" | docker load
```

## 单镜像 Profile

```text
PROFILE=tp1
PROFILE=tp2
PROFILE=tp4
```

| Profile | 默认端口 | context | mem | max-running | Mamba interval |
|---|---:|---:|---:|---:|---:|
| TP1 | 8042 | 262144 | 0.84 | 1 | 8192 |
| TP2 | 8062 | 262144 | 0.88 | 4 | 8192 |
| TP4 | 8068 | 262144 | 0.95 | 8 | 16384 |

TP2 不再要求用户单独准备 legacy SourceFind Docker 镜像。最终镜像内部同时保留：

- TP1 / TP4：flash-attn 260728；
- TP2：legacy flash-attn 260602 sidecar + SourceFind compatibility overlay。

`PROFILE=tp2` 会自动切换到 TP2 sidecar，并在启动时 fail-closed 校验实际加载路径。

## 权重

最终镜像不包含权重，需要挂载：

```text
Target: Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8
Draft : z-lab/Qwen3.8-27B-DFlash2
```

设置：

```bash
export TARGET_MODEL=/path/to/Qwen3.8-27B-SmoothQuant-W8A8-INT8
export DRAFT_MODEL=/path/to/Qwen3.8-27B-DFlash2
```

## 推荐启动

脚本直接调用最终镜像，不再要求安装 overlay：

```bash
chmod +x scripts/*.sh
```

TP1：

```bash
HOST_GPU_ID=0 \
RENDER_DEVICE=/dev/dri/renderD128 \
./scripts/launch.sh tp1
```

TP2：

```bash
HOST_GPU_IDS="0 1" \
RENDER_DEVICES="/dev/dri/renderD128 /dev/dri/renderD129" \
./scripts/launch.sh tp2
```

TP4：

```bash
HOST_GPU_IDS="0 1 2 3" \
RENDER_DEVICES="/dev/dri/renderD128 /dev/dri/renderD129 /dev/dri/renderD130 /dev/dri/renderD131" \
./scripts/launch.sh tp4
```

修改端口 / API 模型名：

```bash
PORT=9000 \
SERVED_MODEL_NAME=my-qwen38 \
./scripts/launch.sh tp4
```

修改常用参数：

```bash
CONTEXT_LENGTH=245760 \
MAX_RUNNING_REQUESTS=4 \
MEM_FRACTION_STATIC=0.93 \
./scripts/launch.sh tp4
```

任意 SGLang 参数透传：

```bash
./scripts/launch.sh tp4 -- \
  --enable-deterministic-inference \
  --log-level info
```

最终镜像 `/opt/qwen38-k100ai/start.sh` 会把 `--` 后参数无损传给对应 SGLang entrypoint。

## 显存 rootfix

v1.3.0 使用 external `/flush_cache` guard。v1.3.1 已改为源码级 allocator rootfix：

- GPU allocator trim 不再绑在 rank0-only `IdleSleeper`；
- 每个 scheduler rank 在 idle housekeeping 中独立执行 `empty_device_cache(self.device_module)`；
- production 默认 `SGLANG_EMPTY_CACHE_INTERVAL=60`；
- 不执行 full `/flush_cache`；
- prefix / Radix / KV / Mamba 逻辑缓存继续保留。

最终验证：

- TP1：长请求后 HBM 可从 98% 回到约 94%；
- TP2：rank0 + rank1 独立 trim；
- TP4：rank0 / 1 / 2 / 3 独立 trim。

旧 guard 不应再作为 v1.3.1 production 策略：

```bash
sudo systemctl disable --now q38-idle-flush-tp1.service 2>/dev/null || true
sudo systemctl disable --now q38-idle-flush-tp2.service 2>/dev/null || true
```

## Agent 推荐

TP4 作为主 Agent backend。

建议：

```text
soft compaction: ~230K
hard compaction: ~240K
parallel_tool_calls: false
```

规划轮：thinking=true。

工具执行轮，尤其 write / patch / shell / 大 JSON：thinking=false。

TP2 tool argument fresh gate：

- 1K chars：PASS；
- 2K chars：PASS；
- 4K chars：会出现重复并撞生成上限。

因此 coding Agent 推荐 patch / chunked write，不要一次 tool call 输出完整大文件。

## 最终镜像验证

| Gate | 结果 |
|---|---|
| TP1 GPU2 authority 64K cache-resume | PASS |
| TP2 function matrix | 9/9 PASS |
| TP2 mixed c4 | PASS |
| TP2 thinking arithmetic | 20/20 |
| TP2 no-thinking arithmetic | 历史 18/20，仅 8/17 |
| TP2 16K / 64K cache-resume | PASS |
| TP4 function matrix | 9/9 PASS |
| TP4 mixed c8 | PASS |
| TP4 64K release-authority cache-resume | PASS |
| all-rank allocator trim | TP1 / TP2 / TP4 PASS |
| Docker save / zstd / docker load round-trip | PASS |
| 最终生产 Chat smoke | TP1 / TP2 / TP4 全部 exact `OK` |

详见 [VALIDATION.md](VALIDATION.md)。

## Healthcheck

```bash
./scripts/healthcheck.sh tp1
./scripts/healthcheck.sh tp2
./scripts/healthcheck.sh tp4
```

## 源码资产

`overlay/`、`rootfix/`、`native_ext/`、`install_overlay.sh` 仍保留，用于：

- 审计本次 production patch；
- 研究 / 二次开发；
- 手工复现旧 overlay 路径。

**最终镜像用户不需要执行 `install_overlay.sh`。**

## 发布校验

- 镜像 SHA：见 [IMAGE_SHA256SUMS](IMAGE_SHA256SUMS)
- 源码文件 SHA：见 [SHA256SUMS](SHA256SUMS)
- 机器可读 manifest：见 [RELEASE_MANIFEST.json](RELEASE_MANIFEST.json)
- 更新说明：见 [RELEASE_NOTES.md](RELEASE_NOTES.md)
