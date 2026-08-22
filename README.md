# Qwen3.8-27B INT8/W8A8 on K100AI

> **SGLang performance optimization for Hygon K100AI**
> TP1 / TP4 · DFlash2 · Agent long context · gfx928 native kernels

> ⚠️ **免责声明 / 风险提示**
>
> 本项目是社区研究成果，不是海光、SourceFind、Qwen、SGLang 或 DFlash2 官方发行版。项目依赖宿主机已经正常工作的 K100AI 驱动、DTK/hyhal、Docker 与 GPU 设备映射。错误的 GPU/PCIe/驱动操作可能导致现有业务中断，极端情况下需要重启服务器恢复。
>
> **本仓库不会自动安装、替换或重新编译宿主机 `amdgpu.ko` / DKMS 驱动，也不会自动修改 GRUB、IOMMU、ACS 或执行 `setpci`。** 如果 `hy-smi`、`/dev/kfd`、`/opt/hyhal` 或官方 SourceFind 容器本身不正常，请先停止部署，不要让本项目替你"修驱动"。

## 这是什么

这是一个面向 **Hygon K100AI / gfx928** 的 Qwen3.8-27B **INT8/W8A8 推理优化总项目**。

仓库不再把某一个并行度或某一种投机算法写进项目名称。TP1、TP4、DFlash2、长上下文 prefill、native INT8 kernel 都是同一个优化栈里的不同 profile / 技术组件。

当前主要目标是：

- 在 K100AI 上稳定运行 Qwen3.8-27B SmoothQuant W8A8；
- 针对真实 Agent 的 32K–128K 长上下文优化 TTFT 与 decode；
- 提供单卡和多卡两种可复现配置；
- DFlash2 投机解码、Radix Cache、gfx928 native kernel 与 correctness 修复统一维护；
- 所有正式性能结果必须通过固定 prompt、output256、无污染、quality/correctness 门禁。

---

## 先选你的 Profile

| Profile | GPU | 正式验收范围 | 代表性能 | 适合谁 | 状态 |
|---|---:|---:|---|---|---|
| **[TP1 Agent128K](TP1.md)** | 1× K100AI | 512 → 128K | 16K **47.87 tok/s** · 64K **31.23** · 128K **32.88** | 卡少、单用户 Agent、优先省 GPU | **ACCEPTED** |
| **[TP4 Agent256K](TP4.md)** | 4× K100AI | 512 → 257.9K | 16K **113.10 tok/s** · 64K **102.21** · 128K **88.68** · 257.9K **72.49** | 长上下文、高 decode、长期主服务 | **CHAMPION** |

> TP1 和 TP4 的正式十档范围并不完全相同，因此不要直接拿"十档平均值"做横向排名。更有意义的是比较相同上下文下的 TTFT / decode，以及你的 GPU 成本。

---

## 部署方式选择（三选一）

> **重要：以下三种方式是替代关系，不是连续步骤。根据你的场景选择其中一种完成即可，不要三种都操作。**

| 方式 | 适合场景 | 是否需要 build | 推荐 |
|---|---|---:|---|
| **A. 统一镜像部署** | 有 Docker + K100AI 环境，能直接下载文件 | 否 | ⭐⭐⭐⭐⭐ |
| **B. 离线迁移部署** | 目标服务器无公网，通过网盘/U盘/内网传输 | 否 | ⭐⭐⭐⭐⭐ |
| **C. 源码构建部署** | 需要修改 patch、kernel 或审计构建过程 | 是 | ⭐⭐⭐ |

- 普通部署选 **A**（有网）或 **B**（无网），两者使用同一个镜像，只是下载/传输方式不同。
- 需要改代码或审计构建过程才选 **C**。

---

### 方式 A：统一镜像部署（推荐，有网环境）

**前提**：目标服务器有 Docker，能访问夸克网盘下载文件。

#### 1. 下载镜像压缩包

- 夸克网盘：[Qwen3.8-K100AI-Unified-20260822](https://pan.quark.cn/s/1e9abeef509a?pwd=duZx)
- 提取码：`duZx`
- 文件：`qwen38-k100ai-int8-unified-20260822.docker.tar.zst`
- 大小：约 **5.67 GiB**（压缩后）/ **31.86 GiB**（Docker 镜像）
- Docker tag：`qwen38-k100ai-int8:unified-20260822`
- SHA256：`e3e2874939b540a935191939fe6309e583a7bf1808f6341f07aba447740d7557`

#### 2. 校验完整性

```bash
# SHA256 校验
echo "e3e2874939b540a935191939fe6309e583a7bf1808f6341f07aba447740d7557  qwen38-k100ai-int8-unified-20260822.docker.tar.zst" | sha256sum -c -

# zstd 完整性
zstd -t qwen38-k100ai-int8-unified-20260822.docker.tar.zst
```

#### 3. 导入 Docker

```bash
zstd -dc qwen38-k100ai-int8-unified-20260822.docker.tar.zst | docker load
```

确认：

```bash
docker images | grep qwen38-k100ai-int8
# 应看到: qwen38-k100ai-int8  unified-20260822  ...  31.86GB
```

#### 4. 准备模型权重

镜像**不包含模型权重**（避免重复传输几十 GB）。需要单独准备并挂载：

| 挂载路径 | 内容 | 来源 |
|---|---|---|
| `/models/target` | Qwen3.8-27B SmoothQuant W8A8/INT8 target | `Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8` (rev `417ede1`) |
| `/models/draft` | Qwen3.8-27B DFlash2 draft | `z-lab/Qwen3.8-27B-DFlash2` (rev `50307d4`) |

#### 5. 启动服务

**TP4（4 卡，默认）**：

```bash
docker run -d \
  --name qwen38-tp4 \
  --network host --ipc host \
  --restart unless-stopped \
  --security-opt label=disable \
  --device /dev/kfd:/dev/kfd \
  --device /dev/dri/renderD128:/dev/dri/renderD128 \
  --device /dev/dri/renderD129:/dev/dri/renderD129 \
  --device /dev/dri/renderD130:/dev/dri/renderD130 \
  --device /dev/dri/renderD131:/dev/dri/renderD131 \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v /path/to/target:/models/target:ro \
  -v /path/to/draft:/models/draft:ro \
  -e PROFILE=tp4 \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -e PORT=8068 \
  qwen38-k100ai-int8:unified-20260822
```

**TP1（单卡）**：

```bash
docker run -d \
  --name qwen38-tp1 \
  --network host --ipc host \
  --restart unless-stopped \
  --security-opt label=disable \
  --device /dev/kfd:/dev/kfd \
  --device /dev/dri/renderD128:/dev/dri/renderD128 \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v /path/to/target:/models/target:ro \
  -v /path/to/draft:/models/draft:ro \
  -e PROFILE=tp1 \
  -e HIP_VISIBLE_DEVICES=0 \
  -e PORT=8090 \
  qwen38-k100ai-int8:unified-20260822
```

> **renderD 编号**：`renderD128`–`renderD131` 是示例，必须根据你机器的实际拓扑替换。用 `ls /dev/dri/` 查看。
>
> **模型路径**：`/path/to/target` 和 `/path/to/draft` 替换为你实际的模型目录。

#### 6. 验证

```bash
# 等待模型加载（TP4 约 3-5 分钟，TP1 约 2-3 分钟）
docker logs -f --tail=50 qwen38-tp4

# 健康检查
curl http://localhost:8068/health
# 应返回 200

# 测试请求
curl http://localhost:8068/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3.8-27B-W8A8-DFlash2-TP4-LongCtx-v8","messages":[{"role":"user","content":"你好"}],"max_tokens":32,"temperature":0}'
```

---

### 方式 B：离线迁移部署（无公网环境）

**前提**：目标服务器无法访问互联网，但可以通过网盘/U盘/内网传输文件。

**与方式 A 的区别**：镜像相同，只是下载和传输方式不同。

#### 1. 在有网络的机器上下载

从夸克网盘下载 `qwen38-k100ai-int8-unified-20260822.docker.tar.zst`（约 5.67 GiB）。

#### 2. 校验

```bash
echo "e3e2874939b540a935191939fe6309e583a7bf1808f6341f07aba447740d7557  qwen38-k100ai-int8-unified-20260822.docker.tar.zst" | sha256sum -c -
zstd -t qwen38-k100ai-int8-unified-20260822.docker.tar.zst
```

#### 3. 传输到目标服务器

```bash
# 方式一：rsync（内网，支持断点续传）
rsync -avP --partial qwen38-k100ai-int8-unified-20260822.docker.tar.zst user@target:/data/

# 方式二：U盘/移动硬盘
# 直接拷贝文件到目标服务器

# 方式三：scp
scp qwen38-k100ai-int8-unified-20260822.docker.tar.zst user@target:/data/
```

#### 4. 在目标服务器上导入

```bash
cd /data
zstd -dc qwen38-k100ai-int8-unified-20260822.docker.tar.zst | docker load
docker images | grep qwen38-k100ai-int8
```

#### 5. 准备模型权重 + 启动

同方式 A 的第 4、5、6 步。模型权重（target + draft，共约 60+ GB）也需要离线传输到目标服务器。

> **注意**：整个镜像导入过程不需要访问互联网。但模型权重仍需单独准备并挂载。

---

### 方式 C：源码构建部署

**前提**：需要修改 patch、kernel 或审计构建过程。普通部署不需要走这条路。

> 根目录的 `Dockerfile / build_image.sh / run.sh` 是 **TP4 Stable Profile** 源码路线。

#### 1. 准备模型

同方式 A 第 4 步。

#### 2. 配置环境

```bash
cp .env.example .env
# 修改 .env 中的：
#   TARGET_MODEL=/path/to/target
#   DRAFT_MODEL=/path/to/draft
#   RENDER0=/dev/dri/renderD128
#   RENDER1=/dev/dri/renderD129
#   RENDER2=/dev/dri/renderD130
#   RENDER3=/dev/dri/renderD131
```

#### 3. 构建镜像

```bash
# 使用仓库内已验证的预编译 native extensions（推荐，不启动编译容器）
bash build_image.sh

# 如果需要从源码重编 native extensions（备用路径）
REBUILD_NATIVE=1 bash build_image.sh
```

#### 4. 启动

```bash
bash run.sh
```

`run.sh` 会自动检查 `.env` 配置、模型目录、GPU 设备，然后启动 Docker 容器。

> 源码构建只是备用路径。临时编译容器无 GPU 设备、无网络、非 privileged，不修改宿主机驱动。

---

### Profile 详情

- **只有 1 张 K100AI** → 看 **[TP1 Agent128K](TP1.md)**
- **有 4 张 K100AI** → 看 **[TP4 Agent256K](TP4.md)**

TP4 离线算力服务器部署见：**[TP4 离线部署教程](TP4_OFFLINE_DEPLOY.md)**。

![TP4 longctx-v8 formal 10-level](assets/tp4_longctx_v8_10level.png)

---

## 项目分层

这个仓库以后按下面的逻辑维护，而不是把所有东西继续堆进一个 README：

| 层 | 内容 |
|---|---|
| **Profile** | TP1 / TP4：不同 GPU 数量、context、运行参数与验收结果 |
| **Common runtime** | Qwen3.8 W8A8 compatibility、correctness repair、Radix Cache、DFlash2 backport |
| **Native kernels** | K100AI/gfx928 INT8 GEMV、LDS-x 等用户态 HIP 扩展 |
| **Long-context** | chunked prefill、paged/varlen repair、U036、Agent prefix cache |
| **Evidence** | 固定十档、quality gate、needle、无污染与稳定性结果 |

这样以后新增 TP2、TP8、MTP 或新的 speculative algorithm，只增加一个 profile / 技术模块，不再重开仓库。

---

## 为什么不是两个 GitHub 仓库

TP1 和 TP4 的共同部分远大于差异：

- 同一套 Qwen3.8-27B W8A8 target；
- 同一份 DFlash2 draft；
- 同一个 SourceFind SGLang / DTK 基线；
- 同一批 gfx928 correctness 结论；
- 同一类 native INT8 kernel；
- 同一套 Agent 长上下文测试方法。

真正不同的是 **并行度、少量 shape-specific kernel、runtime patch 和启动参数**。拆成两个仓库反而会重复维护模型 revision、风险说明、DFlash2 backport、驱动边界和大量公共代码。

因此项目统一定位为：

> **Qwen3.8-27B INT8/W8A8 on K100AI**
>
> Profile = TP1 / TP4 / future TP2 / TP8

---

## 上游与固定版本

- Qwen base: `Qwen/Qwen3.8-27B`
- W8A8 target: `Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8`
- DFlash2 draft: `z-lab/Qwen3.8-27B-DFlash2`
- DFlash reference: `z-lab/dflash`
- SGLang base: SourceFind SGLang 0.5.12 / DTK 26.04 for K100AI

当前验证的 SourceFind image digest：

```text
sha256:366525b25f452f85eb0ea5813604a64f03c648627bc824bb498b56cf5a325dde
```

详细 revision、patch、native kernel、correctness 根因与完整性能表请进入对应 Profile 文档。

---

## 发布策略

- **`main`**：当前推荐稳定代码与文档；
- **TP1 / TP4**：在同一仓库内独立冻结 profile；
- **release tag**：只在 profile 通过正式 quality + 十档 + stability 门后发布；
- 实验分支和 microbench 不作为 README 的"冠军数据"。

当前 TP4 longctx-v8 已完成完整公开打包、冷十档、长上下文质量和稳定性门禁，并作为当前 `main` 推荐 Champion；TP1 已通过 Agent128K acceptance，继续作为单卡 profile 维护。

---

## 版本记录

| 版本 | 日期 | 主要更新 |
|---|---|---|
| **v1.0.1** | 2026-08-22 | 发布 TP1 / TP4 **统一完整 Docker 镜像**；`PROFILE=tp1\|tp4` 切换；增加夸克网盘直下与 SHA256；TP1 真机 smoke 通过；统一镜像纳入 DFlash2 sampling 防崩保护。 |
| **v1.0.0** | 2026-08-21 | TP4 longctx-v8 正式 Champion：128K 49.45s / 88.68 tok/s，257.9K 132.25s / 72.49 tok/s；完成 cold 十档、质量与稳定性门禁。 |
| **v0.2.0** | 2026-08-21 | 仓库重构为统一 K100AI INT8/W8A8 优化项目，TP1 / TP4 作为 profile 维护。 |
| **v0.1.3** | 2026-08-21 | 默认使用已验证的预编译 K100AI 用户态 native extensions，源码重编改为可选路径。 |
| **v0.1.2** | 2026-08-21 | 增加安全的 K100AI native extension 源码构建流程，不修改宿主机驱动。 |
| **v0.1.1** | 2026-08-21 | 增加离线算力服务器部署说明。 |
| **v0.1.0** | 2026-08-21 | 首个公开版本。 |

---

## License / third-party notice

项目自身代码见 [LICENSE](LICENSE)，第三方来源与许可说明见 [NOTICE.md](NOTICE.md)。
