# Qwen3.8-27B W8A8 on K100AI

面向 **Hygon K100AI / gfx928** 的 Qwen3.8-27B W8A8 SGLang + DFlash2 专项优化方案。

当前最新正式版本：**v1.3.1 / v30**。

## 这个项目做了什么

这个项目不是单纯把 Qwen3.8-27B “跑起来”，而是围绕 **K100AI 上的长期实际推理性能、正确性和 Agent 使用体验**做了一整套专项优化：

- 将 **Qwen3.8-27B W8A8 + DFlash2** 适配到 K100AI / gfx928，并修复量化、flash-attn、工具调用等兼容性问题；
- 分别优化 **TP1 / TP2 / TP4**，覆盖 1 / 2 / 4 张 K100AI，并针对不同并行度使用不同优化路径；
- 重点优化 **32K～128K Agent 长上下文**，并持续验证到 **257.9K**；
- 优化 **TTFT、Decode、Prefix/Cache Resume、并发吞吐**，减少长会话和 Agent 连续工作的重复计算；
- 完整验证 OpenAI Compatible API、reasoning、JSON / regex grammar、tool calling、sampling、多模态图片输入和断连回收；
- v1.3.1 进一步完成 **单镜像 TP1 / TP2 / TP4**、all-rank allocator 显存根修和 Agent c4 / c8 功能门禁。

> ⚠️ 本项目是社区研究成果，不是海光、SourceFind、Qwen、SGLang 或 DFlash2 官方发行版。请确保宿主机 K100AI 驱动、`/dev/kfd`、`/opt/hyhal` 和 Docker 本身工作正常。

## 性能展示

下面是仓库公开的 **TP1 / TP2 / TP4 正式十档性能参考**。

统一口径：**canonical corpus / output=256 / DFlash2 / cold / 每档独立 cache flush / contaminated=false**。

> v1.3.1 本次主要更新长期 serving 稳定性、显存 allocator rootfix、Agent 能力和最终单镜像封装，没有把不同测试轮次重新拼成一张“best-of”表。因此这里继续展示已经公开冻结的十档 authority，便于横向比较 TP1 / TP2 / TP4。更晚的 targeted repeat 和 v30 补充结果见 [`PERFORMANCE.md`](PERFORMANCE.md) 与 `results/`。

### 十档 TTFT / Decode

| 上下文 | TP1 TTFT | TP1 Decode | TP2 TTFT | TP2 Decode | TP4 TTFT | TP4 Decode |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 5.93s | 23.92 tok/s | 0.42s | 70.61 tok/s | 0.41s | 100.74 tok/s |
| 2K | 7.21s | 28.65 tok/s | 1.62s | 86.08 tok/s | 1.08s | 119.03 tok/s |
| 4K | 12.41s | 25.65 tok/s | 4.15s | 77.09 tok/s | 2.36s | 95.65 tok/s |
| 8K | 8.42s | 27.58 tok/s | 3.89s | 83.24 tok/s | 2.40s | 110.88 tok/s |
| 12K | 13.74s | 28.29 tok/s | 12.92s | 85.27 tok/s | 4.12s | 91.06 tok/s |
| 16K | 16.55s | 26.59 tok/s | 9.59s | 81.27 tok/s | 4.60s | 113.10 tok/s |
| 32K | 34.01s | 27.30 tok/s | 23.23s | 79.05 tok/s | 10.01s | **128.25 tok/s** |
| 64K | 72.66s | 31.68 tok/s | 40.36s | 73.32 tok/s | 22.18s | **102.21 tok/s** |
| 128K | 174.68s | 33.90 tok/s | 90.66s | 49.98 tok/s | 49.45s | **88.68 tok/s** |
| 257.9K | 466.44s | 24.30 tok/s | 234.28s | 53.10 tok/s | 132.25s | **72.49 tok/s** |

### 十档端到端总耗时

| 上下文 | TP1 Total | TP2 Total | TP4 Total |
|---:|---:|---:|---:|
| 512 | 16.59s | 4.03s | 2.94s |
| 2K | 16.11s | 4.58s | 3.22s |
| 4K | 22.35s | 7.45s | 5.03s |
| 8K | 17.67s | 6.95s | 4.70s |
| 12K | 22.75s | 15.91s | 6.92s |
| 16K | 26.14s | 12.73s | 6.86s |
| 32K | 43.35s | 26.45s | 12.00s |
| 64K | 80.71s | 43.83s | 24.68s |
| 128K | 182.20s | 95.76s | 52.32s |
| 257.9K | 476.93s | 239.09s | 135.77s |

### 长上下文快速对比

| Profile | GPU | 64K Decode | 128K Decode | 257.9K Decode | 建议用途 |
|---|---:|---:|---:|---:|---|
| **TP1** | 1 | 31.68 tok/s | 33.90 tok/s | 24.30 tok/s | GPU 最省、单用户 / worker |
| **TP2** | 2 | 73.32 tok/s | 49.98 tok/s | 53.10 tok/s | 性能 / GPU 成本平衡 |
| **TP4** | 4 | **102.21 tok/s** | **88.68 tok/s** | **72.49 tok/s** | 长上下文、Agent、并发主力 |

![TP1 / TP2 / TP4 十档性能对比](assets/tp1_tp2_tp4_10level.png)

详细测试口径、补充结果和 authority 文件见 [`PERFORMANCE.md`](PERFORMANCE.md)。

---

## v1.3.1 最重要的变化

v1.3.1 已经重新制作成 **一个完整 Docker 镜像**，不再要求新用户先导入 v1.2.2 再安装 overlay。

同一个镜像直接通过：

```text
PROFILE=tp1
PROFILE=tp2
PROFILE=tp4
```

切换 1 卡 / 2 卡 / 4 卡。

主要变化：

- TP1 / TP2 / TP4 全部整合进 **同一个最终镜像**；
- TP2 历史上必须使用的 legacy SourceFind flash-attn ABI 已作为 **260602 sidecar** 内置到同一镜像，`PROFILE=tp2` 会自动切换；
- TP1 / TP4 继续使用验证过的 flash-attn **260728**；
- 修复长期 serving 显存 high-water：allocator trim 从 rank0-only 路径改成 **所有 scheduler rank 独立回收**；
- 不再使用外部 `/flush_cache` systemd guard，因此不会为了回收 allocator 高水位顺带清空 prefix / Radix / KV / Mamba 逻辑缓存；
- TP2 mixed c4、TP4 mixed c8、JSON / regex grammar、sampling、required/named tool 全部重新验证；
- TP1 / TP2 / TP4 均支持 `max_completion_tokens`、`image_url`、OpenAI Compatible Chat Completions；
- 提供新的 v1.3.1 启动脚本，可直接修改常用参数，也可在 `--` 后透传任意 SGLang 参数。

详细变更见 [v1.3.1 Release Notes](RELEASE_NOTES_v1.3.1.md)。

---

# 1. 下载 v1.3.1 最终镜像

夸克网盘：

**[Qwen3.8-K100AI-v1.3.1-final-image.tar.zst](https://pan.quark.cn/s/2e35321f6278?pwd=gPVV)**

提取码：`gPVV`

文件名：

```text
Qwen3.8-K100AI-v1.3.1-final-image.tar.zst
```

SHA256：

```text
4e43edd8a0cf5ee0e501aefb051170587b1da3003392fdd2d32a1d9712e11d8f
```

压缩包约 **6.3 GiB**，解压后的 Docker archive 约 **33.3 GiB**。

最终镜像：

```text
qwen38-k100ai-int8:v1.3.1
```

Image ID：

```text
sha256:534f9512a3d5217c8f65f11183ed75db19e7aa3df1ecc4da1ba993e98793973d
```

校验并导入：

```bash
IMAGE_ARCHIVE=Qwen3.8-K100AI-v1.3.1-final-image.tar.zst

echo "4e43edd8a0cf5ee0e501aefb051170587b1da3003392fdd2d32a1d9712e11d8f  $IMAGE_ARCHIVE" | sha256sum -c -
zstd -t "$IMAGE_ARCHIVE"
zstd -dc "$IMAGE_ARCHIVE" | docker load

docker image inspect qwen38-k100ai-int8:v1.3.1 --format '{{.Id}}'
```

正常应返回：

```text
sha256:534f9512a3d5217c8f65f11183ed75db19e7aa3df1ecc4da1ba993e98793973d
```

发布前已经实际执行过 `docker save -> zstd -> 删除 tag -> zstd -dc | docker load` 回环验证，load 后 Image ID 与原镜像完全一致。

---

# 2. 准备模型权重

最终镜像 **不包含模型权重**。已经部署旧版本的用户可以继续使用原来的 Target / Draft，无需重新下载。

运行时需要：

- Target：`Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8`
- Draft：`z-lab/Qwen3.8-27B-DFlash2`

不需要 BF16 / FP16 Base 权重。

## 方式 A：夸克完整权重包

**[Qwen3.8-K100AI-Weights-20260823](https://pan.quark.cn/s/eb79a87216ba?pwd=Rcxc)**

提取码：`Rcxc`

文件：

```text
qwen38-k100ai-w8a8-dflash2-weights-20260823.tar.zst
```

SHA256：

```text
aa33b9d1ed1e31b1f5c3c6989a302299ecb957ff3f2768f233fdaab17f0073f5
```

示例：

```bash
ARCHIVE=qwen38-k100ai-w8a8-dflash2-weights-20260823.tar.zst

echo "aa33b9d1ed1e31b1f5c3c6989a302299ecb957ff3f2768f233fdaab17f0073f5  $ARCHIVE" | sha256sum -c -
zstd -t "$ARCHIVE"

mkdir -p "$HOME/models/q38-release"
zstd -dc "$ARCHIVE" | tar -xf - -C "$HOME/models/q38-release"

export WEIGHTS_ROOT="$HOME/models/q38-release/Qwen3.8-27B-K100AI-W8A8-DFlash2-Weights-20260823"
export TARGET_MODEL="$WEIGHTS_ROOT/target/Qwen3.8-27B-SmoothQuant-W8A8-INT8"
export DRAFT_MODEL="$WEIGHTS_ROOT/draft/Qwen3.8-27B-DFlash2"
```

## 方式 B：HuggingFace

```bash
python3 -m pip install -U huggingface_hub

# 国内网络可选
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

MODEL_ROOT="$HOME/models"
mkdir -p "$MODEL_ROOT"

hf download Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8 \
  --revision 417ede1e4524c8fdbb586ebdabc9cfc5d0760b3e \
  --local-dir "$MODEL_ROOT/Qwen3.8-27B-SmoothQuant-W8A8-INT8"

hf download z-lab/Qwen3.8-27B-DFlash2 \
  --revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4 \
  --local-dir "$MODEL_ROOT/Qwen3.8-27B-DFlash2"

export TARGET_MODEL="$MODEL_ROOT/Qwen3.8-27B-SmoothQuant-W8A8-INT8"
export DRAFT_MODEL="$MODEL_ROOT/Qwen3.8-27B-DFlash2"
```

两种方式二选一。

---

# 3. 推荐启动方式

v1.3.1 的最终镜像已经包含全部 production patch / rootfix。**普通用户不需要执行 `install_overlay.sh`，也不需要挂载项目研发目录。**

先进入仓库的发布脚本目录：

```bash
cd v1.3.1
chmod +x scripts/*.sh
```

确认：

```bash
export TARGET_MODEL=/你的真实路径/Qwen3.8-27B-SmoothQuant-W8A8-INT8
export DRAFT_MODEL=/你的真实路径/Qwen3.8-27B-DFlash2
```

再根据宿主机的 GPU / `renderD*` 修改设备映射。

```bash
hy-smi
ls -l /dev/dri/renderD*
```

## TP1：1 张卡

默认：context 262144、mem 0.84、max-running 1、端口 8042。

```bash
HOST_GPU_ID=0 \
RENDER_DEVICE=/dev/dri/renderD128 \
./scripts/launch.sh tp1
```

## TP2：2 张卡

默认：context 262144、mem 0.88、max-running 4、端口 8062。

```bash
HOST_GPU_IDS="0 1" \
RENDER_DEVICES="/dev/dri/renderD128 /dev/dri/renderD129" \
./scripts/launch.sh tp2
```

> v1.3.1 最终镜像已经内置 TP2 所需的 legacy flash-attn 260602 sidecar 与 SourceFind compatibility overlay。**不要再单独下载或切换旧 TP2 Docker 镜像。**

## TP4：4 张卡

默认：context 262144、mem 0.95、max-running 8、端口 8068。

```bash
HOST_GPU_IDS="0 1 2 3" \
RENDER_DEVICES="/dev/dri/renderD128 /dev/dri/renderD129 /dev/dri/renderD130 /dev/dri/renderD131" \
./scripts/launch.sh tp4
```

### 修改端口 / API 模型名

```bash
PORT=9000 \
SERVED_MODEL_NAME=my-qwen38 \
./scripts/launch.sh tp4
```

API：

```text
http://服务器IP:9000/v1
```

### 修改常用 SGLang 参数

环境变量示例：

```bash
CONTEXT_LENGTH=245760 \
MAX_RUNNING_REQUESTS=4 \
MEM_FRACTION_STATIC=0.93 \
./scripts/launch.sh tp4
```

### 任意 SGLang 参数透传

参数放在 `--` 后，会由最终镜像入口原样传给 SGLang：

```bash
./scripts/launch.sh tp4 -- \
  --enable-deterministic-inference \
  --log-level info
```

> `v1.3.1/install_overlay.sh` 与源码 patch 继续保留给研究、审计和手工复现使用；**最终镜像用户不需要执行它。**

---

# 4. 不使用脚本时直接 docker run

导入镜像后：

```bash
export IMAGE=qwen38-k100ai-int8:v1.3.1
```

核心只需要：

```text
PROFILE=tp1 | tp2 | tp4
```

例如 TP4：

```bash
docker run -d \
  --name qwen38-tp4 \
  --restart unless-stopped \
  --network host --ipc host \
  --security-opt label=disable \
  --device /dev/kfd:/dev/kfd \
  --device /dev/dri/renderD128:/dev/dri/renderD128 \
  --device /dev/dri/renderD129:/dev/dri/renderD129 \
  --device /dev/dri/renderD130:/dev/dri/renderD130 \
  --device /dev/dri/renderD131:/dev/dri/renderD131 \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$TARGET_MODEL:/models/target:ro" \
  -v "$DRAFT_MODEL:/models/draft:ro" \
  -e PROFILE=tp4 \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -e PORT=8068 \
  -e SERVED_MODEL_NAME=Qwen3.8-27B-W8A8-DFlash2-TP4 \
  "$IMAGE"
```

最终镜像内部会根据 `PROFILE` 自动选择对应 v30 production entrypoint。

---

# 5. 验证

使用仓库提供的 healthcheck：

```bash
./scripts/healthcheck.sh tp1
./scripts/healthcheck.sh tp2
./scripts/healthcheck.sh tp4
```

或者：

```bash
curl http://127.0.0.1:8068/v1/models
curl http://127.0.0.1:8068/v1/loads
```

查看日志：

```bash
docker logs -f qwen38-tp4
```

---

# 6. v1.3.1 最终验证摘要

最终镜像 Image ID：

```text
sha256:534f9512a3d5217c8f65f11183ed75db19e7aa3df1ecc4da1ba993e98793973d
```

发布前实际完成：

| 项目 | 结果 |
|---|---|
| TP1 GPU2 authority 64K cache-resume | PASS；cold×2 deterministic；cached 57,344 |
| TP2 function matrix | 9/9 PASS |
| TP2 mixed concurrency | c4 PASS |
| TP2 thinking arithmetic | 20/20 |
| TP2 no-thinking arithmetic | 历史 18/20，仅 case 8/17；无新增回归 |
| TP2 16K / 64K cache-resume | PASS |
| TP2 allocator trim | rank0 + rank1 PASS |
| TP4 function matrix | 9/9 PASS |
| TP4 mixed concurrency | c8 PASS |
| TP4 64K cache-resume | release-authority 参数 PASS |
| TP4 allocator trim | rank0 / 1 / 2 / 3 PASS |
| Docker save / zstd test / docker load round-trip | PASS |
| 最终生产 TP1 / TP2 / TP4 Chat smoke | 全部 HTTP 200，exact `OK` |
| Container | restart=0 / OOM=false / unless-stopped |

显存 rootfix 的关键点：只回收 PyTorch / HIP caching allocator 的 transient high-water，不执行 full `/flush_cache`，因此 prefix / Radix / KV / Mamba cache 可以继续保留。

完整验证说明见 [`v1.3.1/VALIDATION.md`](v1.3.1/VALIDATION.md)。

---

# 7. Agent 推荐配置

主 Agent 推荐 **TP4**。

建议：

```text
soft context compaction: ~230K
hard context compaction: ~240K
parallel_tool_calls: false
```

规划 / 分析轮：

```json
{
  "chat_template_kwargs": {
    "enable_thinking": true
  }
}
```

工具执行轮，尤其 write / patch / shell / 大 JSON：

```json
{
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

实际测试中 TP1 / TP2 在长字符串 `tool_choice=auto + stream` 时，关闭 thinking 明显更可靠；TP2 1K / 2K 字符 tool argument PASS，4K 会出现重复，因此 coding Agent 推荐使用 `apply_patch` / chunked write，不要一次 tool call 重写整个大文件。

`parallel_tool_calls=true` 与 `/v1/responses` 目前仍不作为 frozen release authority；正式兼容入口是 `/v1/chat/completions`。

---

# 8. 性能参考

代表性 Decode 结果：

| Profile | GPU | 64K Decode | 128K Decode | 257.9K Decode | 建议用途 |
|---|---:|---:|---:|---:|---|
| **TP1** | 1 | ~31.7 tok/s | ~33.9 tok/s | ~24.3 tok/s | GPU 最省、单用户 |
| **TP2** | 2 | ~73.3 tok/s | ~50.0 tok/s | ~53.1 tok/s | 性能 / GPU 成本平衡、worker |
| **TP4** | 4 | ~102.2 tok/s | ~88.7 tok/s | ~72.5 tok/s | 长上下文、Agent、并发 |

TP4 并发总吞吐会随 prompt / cache / 输出长度变化。v1.3.1 正式配置 `max-running=8`，功能层已通过 mixed c8；对于多个 128K 级重 Agent 会话仍应控制并发或允许 scheduler 排队。

更多测速数据见 [PERFORMANCE.md](PERFORMANCE.md)。

![TP1 / TP2 / TP4 十档性能对比](assets/tp1_tp2_tp4_10level.png)

---

# 9. 验证环境

| 项目 | 验证环境 |
|---|---|
| GPU | Hygon K100AI / gfx928 |
| OS | Kylin Linux Advanced Server V10 (Halberd) |
| Host kernel / amdgpu | `4.19.90-89.27.v2401.ky10.x86_64` |
| hy-smi / hyhal | `hy-smi 1.20.0`，宿主机 `/opt/hyhal -> /usr/local/hyhal` |
| DTK | `DTK-26.04-DCC2602-0317` |
| Docker | `18.09.0`（更新版本通常也可使用） |
| SGLang | `0.5.12+das.opt.dtk2604` |
| Torch | `2.9.0+das.opt1.dtk2604.2605281139.gd0fc8c` |
| TP1 / TP4 flash-attn | `2.8.3+das.opt1.dtk2604.torch290.2607280958.gebb4be` |
| TP2 embedded sidecar | `2.8.3+das.opt1.dtk2604.torch290.2606021702.ge93bd4` |
| KV Cache | BF16 |

普通用户不要在最终镜像里自行升级 Torch / SGLang / flash-attn；这些版本和 TP2 sidecar ABI 都属于当前 correctness 合同。

---

# 历史版本

- [v1.3.0 Release Notes](RELEASE_NOTES_v1.3.0.md)
- [v1.2.2 Release Notes](RELEASE_NOTES_v1.2.2.md)
- [v1.2.1 Release Notes](RELEASE_NOTES_v1.2.1.md)
- [v1.2.0 Release Notes](RELEASE_NOTES_v1.2.0.md)

历史版本继续保留用于复现和回滚，但**新部署请直接使用 v1.3.1 最终单镜像**。
