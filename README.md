# Qwen3.8-27B W8A8 on K100AI

面向 **Hygon K100AI / gfx928** 的 Qwen3.8-27B W8A8 SGLang 优化方案。

## 这个项目做了什么

- 针对 **K100AI + Qwen3.8-27B W8A8** 做专项推理优化，不只是把模型“跑起来”，目标是把它做成适合长期使用的本地大模型服务。
- 从早期 **vLLM** 阶段开始经过数百轮测试和调优，后续迁移到 **SGLang**，把已经验证有效的思路重新适配并继续优化。
- 完成 Qwen3.8 W8A8 在 K100AI / SourceFind SGLang 上的兼容与正确性修复，重点保证 **回答正常、工具调用正常、重复运行稳定**。
- 将 **DFlash2** 移植并适配到 K100AI，进一步提升实际生成速度。
- 分别优化 **TP1 / TP2 / TP4**，覆盖 1 卡、2 卡、4 卡三种部署方式，并针对不同并行度分别调优。
- 重点优化 **32K～128K Agent 长上下文**，同时持续验证到 **257.9K**，减少长对话和长任务中的等待时间。
- 优化 **缓存复用 / 长会话续写**，减少重复计算，让 Agent 连续工作时更实用。
- 针对 TP4 做 **2 / 3 / 4 并发**适配，最高按 4 并发优化，让并发增加时总吞吐能够继续上升。

当前统一支持：

- **TP1：1 张 K100AI**
- **TP2：2 张 K100AI**
- **TP4：4 张 K100AI**

当前最新 production profile release 是 **v1.3.0 / v30**。它复用 v1.2.2 完整 Docker 镜像作为 TP1/TP4 二进制基线，并为 TP1/TP2/TP4 冻结新的 correctness、cache-resume、工具/API、断连回收和显存守护合同。全新安装仍先取得 v1.2.2 完整镜像，再按 [`v1.3.0/`](v1.3.0/) 安装精确 overlay；TP2 必须保留验证过的 legacy SourceFind image，不能为了形式统一切换到 unified image。

> ⚠️ 本项目是社区研究成果，不是海光、SourceFind、Qwen、SGLang 或 DFlash2 官方发行版。请确保宿主机 K100AI 驱动、`/dev/kfd`、`/opt/hyhal` 和 Docker 本身工作正常。

## 验证环境 / 建议基线

本项目当前版本是在下面这套 K100AI 环境上开发和验收的。**建议宿主机环境尽量对齐，并且不要低于这套验证基线。** 更低版本不代表一定不能运行，只是本项目没有做过完整验证。

| 项目 | 本项目验证环境 |
|---|---|
| GPU | Hygon K100AI / gfx928 |
| OS | Kylin Linux Advanced Server V10 (Halberd) |
| Host kernel / amdgpu | `4.19.90-89.27.v2401.ky10.x86_64` |
| amdgpu 来源 | `kernel-modules-4.19.90-89.27.v2401.ky10.x86_64`（该环境未单独暴露 amdgpu `version:` 字段） |
| hy-smi / hyhal | `hy-smi 1.20.0`，宿主机 `/opt/hyhal -> /usr/local/hyhal` |
| DTK 基线 | `DTK-26.04-DCC2602-0317` |
| Docker | `18.09.0`（更新版本通常也可使用） |
| 镜像内 SGLang | `0.5.12+das.opt.dtk2604` |
| 镜像内 Torch | `2.9.0+das.opt1.dtk2604.2605281139.gd0fc8c` |
| 镜像内 flash-attn | `2.8.3+das.opt1.dtk2604.torch290.2607280958.gebb4be` |
| KV Cache | BF16 |

部署前建议先检查自己的宿主机：

```bash
uname -r
/usr/local/hyhal/bin/hy-smi --version || hy-smi --version
test -e /dev/kfd && echo '/dev/kfd: OK'
test -d /opt/hyhal && echo '/opt/hyhal: OK'
docker version
```

> 完整镜像已经固定 SGLang、Torch、flash-attn 等用户态运行环境，普通用户不要在容器内自行升级这些组件。真正需要重点确认的是宿主机 **K100AI 驱动 / 内核、hyhal、GPU 设备映射和 Docker** 是否正常。

## v1.3.0 / v30 更新

v1.3.0 是当前最新 correctness-first production profile release：

- TP1：`Final-v2 -> checkpoint8192 -> Hybrid14 -> v30 common`；
- TP2：`longtail-v1/{3,23} -> checkpoint8192 -> v30 common`，继续固定 legacy SourceFind image；
- TP4：保持既有 v30 FINAL，不在本版本返工；
- TP1/TP2 的 16K、64K、139,265 cache-resume 均为 deterministic cold==cached；
- greedy、sampling、min-p、seed、JSON/regex grammar、required/named tool、mixed c3 全部通过；
- thinking arithmetic 20/20；no-thinking 保持历史 18/20[8,17]，没有新增回归；
- stream/non-stream 客户端断连都能自动回收并继续健康 serving；
- 新增外置 idle-cache guard，只在服务完全 idle 时调用官方 `/flush_cache`。

发布包、安装边界和机器可读证据见 [v1.3.0 Release Notes](RELEASE_NOTES_v1.3.0.md) 与 [`v1.3.0/`](v1.3.0/)。v1.2.2 完整镜像仍是新安装用户需要下载的二进制基线。

## v1.2.2 更新

v1.2.2 是当前推荐正式版本，完整镜像已经包含 v1.2.1 的 TP4 raw-q8 修复和本次 DFlash2 non-greedy 修复。

- 支持 `temperature` / `top_k` / `top_p` / `min_p` 的 DFlash2 speculative sampling；
- 不再要求客户端必须把 `temperature` 固定为 0；
- 修复旧 K100AI backport 中 non-greedy 请求触发 hard raise、进而导致 TP4 scheduler 退出的问题；
- TP4 mixed c4（greedy + sampling）已实机验证 4/4 200 OK，四 rank verify audit 无分叉；
- greedy 路径保持不变，candidate 与 v1.2.1 production 的完整 message SHA 对齐；
- `sampling_seed`、DFlash-native grammar / `tool_choice=required`、non-greedy cache-resume 全矩阵仍属于后续范围，不在本版本过度声明。

新部署用户直接使用 v1.2.2 完整镜像即可。已经部署旧版本、且不想重新导入完整镜像的用户，才需要使用 [`hotfixes/v1.2.2/apply.sh`](hotfixes/v1.2.2/apply.sh) 做增量升级。

详细验证数据和已知边界见 [v1.2.2 Release Notes](RELEASE_NOTES_v1.2.2.md)。

## v1.2.1 TP4 raw-q8 hotfix

v1.2.1 是基于 v1.2.0 的 **TP4-only 性能修复**，TP1 / TP2、权重和用户态依赖版本均不变。

- 修正 SourceFind flash-attn 260728 raw `paged_attention` 的 TP4 layout ABI 适配；
- 恢复 exact TP4 DFlash **single raw-q8 verifier**，不再为兼容性长期承受 `2 × q4` 的长 KV 重复扫描成本；
- 16K / 64K / 128K / 257.9K isolated gate 与 v1.2.0 正确 reference **bitwise equal**；
- full-model canonical 输出 SHA 在 16K / 64K / 128K / 257.9K 与 v1.2.0 正确版完全一致；
- 128K 三次 Decode `87.90 / 87.33 / 87.36 tok/s`；257.9K 两次 `70.49 / 73.56 tok/s`；
- 已有 v1.2.0 完整镜像的用户无需重新下载 6.6GB 镜像，直接执行 [`hotfixes/v1.2.1/apply.sh`](hotfixes/v1.2.1/apply.sh) 生成本地 v1.2.1 镜像。

详细修复原理、验证范围和回滚方式见 [v1.2.1 Release Notes](RELEASE_NOTES_v1.2.1.md) 与 [hotfix README](hotfixes/v1.2.1/README.md)。

## v1.2.0 更新

- TP1 / TP2 / TP4 合并为 **同一个完整镜像**，通过 `PROFILE=tp1|tp2|tp4` 切换。
- 更新 SourceFind 新版 `flash-attn 2.8.3+...2607280958.gebb4be`。
- 修复新版 flash-attn 下 TP2 / TP4 可能出现的 **乱答、输出异常、工具调用异常**。
- TP2 继续优化短、中、长上下文，修复部分长度下速度突然下降的问题。
- TP4 继续强化长上下文、缓存复用和 Agent 场景，并完善 **2 / 3 / 4 并发**。
- 正常 OpenAI Compatible API、Reasoning、`tool_choice=auto` / 默认工具调用已验证可用。
- 启动时可直接修改 **端口** 和 **对外模型名称**。

## 性能参考

以下为前一轮正式十档结果中的代表点，本次主要更新正确性、兼容性和统一镜像，数值用于选 Profile：

| Profile | GPU | 64K Decode | 128K Decode | 257.9K Decode | 建议用途 |
|---|---:|---:|---:|---:|---|
| **TP1** | 1 | ~31.7 tok/s | ~33.9 tok/s | ~24.3 tok/s | GPU 最省、单用户 |
| **TP2** | 2 | ~73.3 tok/s | ~50.0 tok/s | ~53.1 tok/s | 性能 / GPU 成本平衡 |
| **TP4** | 4 | ~102.2 tok/s | ~88.7 tok/s | ~72.5 tok/s | 长上下文、Agent、并发 |

TP4 并发总吞吐粗略参考（不同请求并发，总吞吐，不是单请求速度）：

| 并发数 | 总吞吐 |
|---:|---:|
| 1 | ~95–97 tok/s |
| 2 | ~115–117 tok/s |
| 3 | ~141–152 tok/s |
| 4 | ~175–183 tok/s |

> 并发结果会受 prompt 长度、缓存命中、输出长度等影响，仅作为实际部署容量参考。

完整十档数据、TTFT、Decode 和历史测试结果见 [PERFORMANCE.md](PERFORMANCE.md)。原有测速 JSON 与图片继续保留在仓库中。

![TP1 / TP2 / TP4 十档性能对比](assets/tp1_tp2_tp4_10level.png)

---

## 1. 下载完整镜像

当前完整镜像基线：**v1.2.2**（最新 production profile release：**v1.3.0 / v30**）

夸克网盘：

**[Qwen3.8-K100AI-Unified-v1.2.2](https://pan.quark.cn/s/653e165c2fc7?pwd=DBia)**

提取码：`DBia`

镜像文件：

```text
Qwen3.8-K100AI-Unified-v1.2.2.tar.zst
```

SHA256：

```text
91818fcc5ae0fc1cfcec6b6b9cc2950ee991f293fd16221ccd80e91fa069850d
```

压缩包约 **6.5 GiB（7.03 GB）**，解压后的 Docker tar 约 **31.6 GiB（33.90 GB）**。

校验并导入：

```bash
IMAGE_ARCHIVE=Qwen3.8-K100AI-Unified-v1.2.2.tar.zst

echo "91818fcc5ae0fc1cfcec6b6b9cc2950ee991f293fd16221ccd80e91fa069850d  $IMAGE_ARCHIVE" | sha256sum -c -
zstd -t "$IMAGE_ARCHIVE"
zstd -dc "$IMAGE_ARCHIVE" | docker load
```

导入后的镜像：

```text
qwen38-k100ai-int8:unified-20260827-v1.2.2
```

---

## 2. 准备模型（两种下载方式二选一）

镜像 **不包含模型权重**。如果你已经部署过上一版，原来的 Target / Draft 可以直接继续使用。

运行时必须同时有两份模型：

- Target：`Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8`
- Draft：`z-lab/Qwen3.8-27B-DFlash2`

**不需要 BF16 / FP16 Base 权重。**

### 方式 A：夸克完整权重包（推荐）

一个压缩包里已经同时包含 Target + Draft：

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

解压并设置真实路径：

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

上面两条 `TARGET_MODEL` / `DRAFT_MODEL` 已经对应整合包的真实目录层级，正常按这个位置解压就不用再改。

### 方式 B：HuggingFace

```bash
python3 -m pip install -U huggingface_hub

# 国内网络可选：
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

两种方式只选一种，不要重复下载。

---

## 3. 启动

### 启动前真正需要确认 / 修改的值

下面启动命令中，普通用户只需要确认这几项：

1. `TARGET_MODEL`：Target 模型的**真实绝对路径**。如果按上面的夸克或 HuggingFace 命令设置，直接使用即可。
2. `DRAFT_MODEL`：Draft 模型的**真实绝对路径**。同上。
3. `R0 / R1 / R2 / R3`：你准备使用的 K100AI 在宿主机对应的真实 `/dev/dri/renderD*` 设备号。**这个必须按你自己的机器修改。**
4. `PORT`：API 端口。默认值可以直接用；如果端口被占用，再改成别的端口。
5. `MODEL_NAME`：客户端看到的模型名称，可以随便改；它只是 API 对外名称，不是模型文件夹名称。

`HIP_VISIBLE_DEVICES` 在下面示例中已经按容器内 GPU 顺序写好，通常**不要改**。

先确认准备使用的 GPU 对应哪个 `renderD*`：

```bash
hy-smi
ls -l /dev/dri/renderD*
```

下面使用 `renderD128` 开始举例，**请替换成你机器上的实际设备号**。

导入 v1.2.2 完整镜像后，只需要指定镜像名：

```bash
export IMAGE=qwen38-k100ai-int8:unified-20260827-v1.2.2
```

下面 TP1 / TP2 / TP4 都直接使用这一份镜像，通过 `PROFILE` 选择并行模式。

### TP1：1 张卡

```bash
export R0=/dev/dri/renderD128

docker run -d \
  --name qwen38-tp1 \
  --restart unless-stopped \
  --network host --ipc host \
  --security-opt label=disable \
  --device /dev/kfd:/dev/kfd \
  --device "$R0:$R0" \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$TARGET_MODEL:/models/target:ro" \
  -v "$DRAFT_MODEL:/models/draft:ro" \
  -e PROFILE=tp1 \
  -e HIP_VISIBLE_DEVICES=0 \
  -e PORT=8090 \
  -e MODEL_NAME=Qwen3.8-27B-W8A8-TP1 \
  "$IMAGE"
```

### TP2：2 张卡

> **v1.3.0 correctness warning：**下面的 unified-image 命令只保留为 v1.2.2 基线示例。v1.3.0 TP2 正式链必须使用 [`v1.3.0/`](v1.3.0/) 中固定的 legacy SourceFind launcher；unified image 在 TP2 冻结 arithmetic gate 上有质量回归，不得作为 v1.3.0 TP2 production。

```bash
export R0=/dev/dri/renderD128
export R1=/dev/dri/renderD129

docker run -d \
  --name qwen38-tp2 \
  --restart unless-stopped \
  --network host --ipc host \
  --security-opt label=disable \
  --device /dev/kfd:/dev/kfd \
  --device "$R0:$R0" \
  --device "$R1:$R1" \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$TARGET_MODEL:/models/target:ro" \
  -v "$DRAFT_MODEL:/models/draft:ro" \
  -e PROFILE=tp2 \
  -e HIP_VISIBLE_DEVICES=0,1 \
  -e PORT=8062 \
  -e MODEL_NAME=Qwen3.8-27B-W8A8-TP2 \
  "$IMAGE"
```

### TP4：4 张卡

```bash
export R0=/dev/dri/renderD128
export R1=/dev/dri/renderD129
export R2=/dev/dri/renderD130
export R3=/dev/dri/renderD131

docker run -d \
  --name qwen38-tp4 \
  --restart unless-stopped \
  --network host --ipc host \
  --security-opt label=disable \
  --device /dev/kfd:/dev/kfd \
  --device "$R0:$R0" \
  --device "$R1:$R1" \
  --device "$R2:$R2" \
  --device "$R3:$R3" \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$TARGET_MODEL:/models/target:ro" \
  -v "$DRAFT_MODEL:/models/draft:ro" \
  -e PROFILE=tp4 \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -e PORT=8068 \
  -e MODEL_NAME=Qwen3.8-27B-W8A8-TP4 \
  "$IMAGE"
```

---

## 4. 修改端口和模型名称

不需要修改镜像内部文件，直接改启动命令中的环境变量：

```bash
-e PORT=9000
-e MODEL_NAME=my-qwen38
```

例如：

```text
PORT=9000
MODEL_NAME=my-qwen38
```

API 地址就是：

```text
http://服务器IP:9000/v1
```

`/v1/models` 返回的模型 ID 就会是：

```text
my-qwen38
```

也兼容使用：

```bash
-e SERVED_MODEL_NAME=my-qwen38
```

---

## 5. 验证和管理

查看模型：

```bash
curl http://127.0.0.1:8068/v1/models
```

查看日志：

```bash
docker logs -f qwen38-tp4
```

停止 / 启动 / 重启：

```bash
docker stop qwen38-tp4
docker start qwen38-tp4
docker restart qwen38-tp4
```

---

## 已知限制

- 当前统一使用 **BF16 KV Cache**；不要自行切换 FP8 KV Cache。
- 正常 `tool_choice=auto` / 默认工具调用已验证可用。
- 当前 SGLang DFlash 不支持 `tool_choice=required` 对应的强制 grammar-constrained decoding，该模式会返回 400。
- TP1 / TP2 的极长上下文缓存复用仍在继续完善；TP4 当前完成度更高，但仍会继续扩大随机缓存与并发回归测试。

技术细节与完整数据： [TP1](TP1.md) · [TP2](TP2.md) · [TP4](TP4.md) · [PERFORMANCE](PERFORMANCE.md)

## License

项目自身代码见 [LICENSE](LICENSE)，第三方来源与许可说明见 [NOTICE.md](NOTICE.md)。
