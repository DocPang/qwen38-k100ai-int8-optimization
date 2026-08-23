# Qwen3.8-27B INT8/W8A8 on K100AI

> **SGLang performance optimization for Hygon K100AI**
> TP1 / TP2 / TP4 · DFlash2 · Agent long context · gfx928 native kernels

> ⚠️ **免责声明 / 风险提示**
>
> 本项目是社区研究成果，不是海光、SourceFind、Qwen、SGLang 或 DFlash2 官方发行版。项目依赖宿主机已经正常工作的 K100AI 驱动、DTK/hyhal、Docker 与 GPU 设备映射。错误的 GPU/PCIe/驱动操作可能导致现有业务中断，极端情况下需要重启服务器恢复。
>
> **本仓库不会自动安装、替换或重新编译宿主机 `amdgpu.ko` / DKMS 驱动，也不会自动修改 GRUB、IOMMU、ACS 或执行 `setpci`。** 如果 `hy-smi`、`/dev/kfd`、`/opt/hyhal` 或官方 SourceFind 容器本身不正常，请先停止部署，不要让本项目替你"修驱动"。

## 这是什么

这是一个面向 **Hygon K100AI / gfx928** 的 Qwen3.8-27B **INT8/W8A8 推理优化总项目**。

仓库不再把某一个并行度或某一种投机算法写进项目名称。TP1、TP2、TP4、DFlash2、长上下文 prefill、native INT8 kernel 都是同一个优化栈里的不同 profile / 技术组件。

当前主要目标是：

- 在 K100AI 上稳定运行 Qwen3.8-27B SmoothQuant W8A8；
- 针对真实 Agent 的 32K–128K 长上下文优化 TTFT 与 decode；
- 提供 TP1 / TP2 / TP4 三种可复现配置；
- DFlash2 投机解码、Radix Cache、gfx928 native kernel 与 correctness 修复统一维护；
- 所有正式性能结果必须通过固定 prompt、output256、无污染、quality/correctness 门禁。

---

## 先选你的 Profile

| Profile | GPU | 正式验收范围 | 代表性能 | 适合谁 | 状态 |
|---|---:|---:|---|---|---|
| **[TP1](TP1.md)** | 1× K100AI | 512 → 257.9K | 64K **72.66s / 31.68 tok/s** · 128K **174.68s / 33.90** · 257.9K **466.44s / 24.30** | GPU 最省、单用户 Agent | **ACCEPTED** |
| **[TP2](TP2.md)** | 2× K100AI | 512 → 257.9K | 64K **40.36s / 73.32 tok/s** · 128K **90.66s / 49.98** · 257.9K **234.28s / 53.10** | 性能 / GPU 成本平衡 | **ACCEPTED** |
| **[TP4](TP4.md)** | 4× K100AI | 512 → 257.9K | 64K **22.18s / 102.21 tok/s** · 128K **49.45s / 88.68** · 257.9K **132.25s / 72.49** | 长上下文、高 decode、长期主服务 | **CHAMPION** |

> 三个 Profile 使用同一套 target / draft、同一测试语料和 output256 口径。选择时优先看相同上下文下的 TTFT / decode，再结合 GPU 成本。

### TP1 / TP2 / TP4 正式十档

统一口径：**canonical corpus / output=256 / DFlash2 / cold / 每档独立 cache flush / contaminated=false**。

| 上下文 | TP1 TTFT | TP1 Decode | TP2 TTFT | TP2 Decode | TP4 TTFT | TP4 Decode |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 5.93s | 23.92 | 0.42s | 70.61 | 0.41s | 100.74 |
| 2K | 7.21s | 28.65 | 1.62s | 86.08 | 1.08s | 119.03 |
| 4K | 12.41s | 25.65 | 4.15s | 77.09 | 2.36s | 95.65 |
| 8K | 8.42s | 27.58 | 3.89s | 83.24 | 2.40s | 110.88 |
| 12K | 13.74s | 28.29 | 12.92s | 85.27 | 4.12s | 91.06 |
| 16K | 16.55s | 26.59 | 9.59s | 81.27 | 4.60s | 113.10 |
| 32K | 34.01s | 27.30 | 23.23s | 79.05 | 10.01s | 128.25 |
| 64K | 72.66s | 31.68 | 40.36s | 73.32 | 22.18s | 102.21 |
| 128K | 174.68s | 33.90 | 90.66s | 49.98 | 49.45s | 88.68 |
| 257.9K | 466.44s | 24.30 | 234.28s | 53.10 | 132.25s | 72.49 |

![TP1 / TP2 / TP4 十档性能对比](assets/tp1_tp2_tp4_10level.png)

完整 Total、scaling 与质量门见 **[PERFORMANCE.md](PERFORMANCE.md)**。机器可读统一数据：`results/tp1_tp2_tp4_10level_20260823.json`。

---

## 部署方式选择（三选一）

当前统一正式版：**v1.1.0（2026-08-23）**，同时支持 `PROFILE=tp1|tp2|tp4`。

> **重要：以下三种方式是替代关系，不是连续步骤。根据你的场景选择其中一种完成即可，不要三种都操作。**

| 方式 | 说明 | 需要传输的大小 | 是否需要 build | 推荐 |
|---|---|---:|---:|---|
| **A. 完整成品镜像** | 直接导入已验收的统一 Docker 镜像，load 即用 | 5.65 GiB（压缩） | 否 | ⭐⭐⭐⭐⭐ |
| **B. 官方镜像 + 补丁构建** | 拉取 SourceFind 官方基础镜像，打上小补丁（patchset + 预编译 .so），docker build | 官方镜像 + ~2.3 MB 补丁 | 是（docker build） | ⭐⭐⭐⭐ |
| **C. 全源码构建** | 同 B，但 native extensions 从源码重新编译 | 官方镜像 + 源码 | 是（docker build + 编译） | ⭐⭐ |

### 选定部署方式后，只下载对应内容

| 你选择的部署方式 | 需要下载 | 不需要下载 |
|---|---|---|
| **A. 完整成品镜像** | `unified-20260823` 完整镜像 + **一套模型权重** | SourceFind 官方基础镜像、B/C 构建流程 |
| **B. 官方镜像 + 补丁构建** | SourceFind 官方基础镜像 + `v1.1.0` 仓库 + **一套模型权重** | A 的完整成品镜像、源码重编 |
| **C. 全源码构建** | SourceFind 官方基础镜像 + `v1.1.0` 仓库 + **一套模型权重** | A 的完整成品镜像；native `.so` 会从源码重编 |

> **“一套模型权重”始终只有一套。** 下方夸克整合包与 HuggingFace 是两个下载来源，**二选一**，不要两边都下载。

- 普通部署选 **A**（最快）或 **B**（已有官方镜像时最省流量）。
- 需要修改 patch、kernel 或审计构建过程才选 **C**。
- 三种方式都支持离线：A 的镜像包和 B/C 的官方镜像都可以先在有网机器下载再拷贝到目标服务器。

### 所有方式共用：模型权重下载（二选一）

> **无论选择 A / B / C，都只需要准备一套模型权重。下面两个下载来源二选一即可，不要两个都下载。**
>
> 最终都应得到同样的两个模型目录：**W8A8 Target + DFlash2 Draft**。不需要 Qwen3.8 BF16/FP16 Base 权重。

固定版本：

| 模型 | 固定 revision |
|---|---|
| `Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8` | `417ede1e4524c8fdbb586ebdabc9cfc5d0760b3e` |
| `z-lab/Qwen3.8-27B-DFlash2` | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` |

#### 下载地址 1：夸克网盘整合权重包

已经把当前实际使用的 Target + Draft 整理成一个包：

- 文件夹：`Qwen3.8-K100AI-Weights-20260823`
- 下载：**[夸克网盘](https://pan.quark.cn/s/eb79a87216ba?pwd=Rcxc)**
- 提取码：`Rcxc`
- 压缩包：`qwen38-k100ai-w8a8-dflash2-weights-20260823.tar.zst`
- SHA256：`aa33b9d1ed1e31b1f5c3c6989a302299ecb957ff3f2768f233fdaab17f0073f5`

```bash
tar --use-compress-program=unzstd -xf qwen38-k100ai-w8a8-dflash2-weights-20260823.tar.zst
```

该整合包已经同时包含 W8A8 Target 和 DFlash2 Draft，**下载这个包后不要再执行下面的 HuggingFace 下载命令**。

#### 下载地址 2：HuggingFace

国内网络可使用群友实测可用的 HF Mirror；如果机器可以直接访问 HuggingFace，去掉下面两行 `export` 即可，仍然属于同一个 HuggingFace 下载方案。

```bash
python3 -m pip install -U huggingface_hub

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

hf download Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8 \
  --revision 417ede1 \
  --local-dir /home/models/Qwen3.8-27B-SmoothQuant-W8A8-INT8

hf download z-lab/Qwen3.8-27B-DFlash2 \
  --revision 50307d4 \
  --local-dir /home/models/Qwen3.8-27B-DFlash2
```

> 这里两条 `hf download` 是 **同一个 HuggingFace 方案内必须准备的两份模型**，不是两个可选方案。执行 HuggingFace 方案后，不需要再下载夸克整合包。

---

### 方式 A：完整成品镜像（推荐）

**说明**：使用已经验收的 **TP1 / TP2 / TP4 统一 Docker 运行时镜像**，导入后直接运行，不需要任何 build 步骤。

#### 1. 获取镜像压缩包

- 文件：`qwen38-k100ai-int8-unified-20260823.docker.tar.zst`
- 大小：**5.65 GiB（6,065,184,632 bytes）**（压缩后）/ **约 31.70 GB（Docker image `.Size`）**（Docker 镜像）
- Docker tag：`qwen38-k100ai-int8:unified-20260823`
- SHA256：`6d14588722b0fea0ab66a53e2810385d1f9999a9cd78c8e1d2e6640c744f2b14`
- 下载：**[夸克网盘：full_images](https://pan.quark.cn/s/e7626123faa0?pwd=M8Fr)** · 提取码：`M8Fr`

> 离线环境：先在有网机器下载完整镜像，再通过 rsync / U盘 / scp 拷贝到目标服务器。

#### 2. 校验完整性

```bash
# SHA256 校验
echo "6d14588722b0fea0ab66a53e2810385d1f9999a9cd78c8e1d2e6640c744f2b14  qwen38-k100ai-int8-unified-20260823.docker.tar.zst" | sha256sum -c -

# zstd 完整性
zstd -t qwen38-k100ai-int8-unified-20260823.docker.tar.zst
```

#### 3. 导入 Docker

```bash
zstd -dc qwen38-k100ai-int8-unified-20260823.docker.tar.zst | docker load
```

确认：

```bash
docker images | grep qwen38-k100ai-int8
# 应看到: qwen38-k100ai-int8  unified-20260823
```

#### 4. 挂载模型权重

镜像**不包含模型权重**。请先按上方“模型权重下载（二选一）”任选一个来源准备好同一套 Target + Draft，然后挂载：

| 容器路径 | 内容 |
|---|---|
| `/models/target` | Qwen3.8-27B SmoothQuant W8A8/INT8 Target |
| `/models/draft` | Qwen3.8-27B DFlash2 Draft |

> **不要重复下载。** 如果已经下载夸克整合包，就不要再从 HuggingFace 下载；反之亦然。

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
  qwen38-k100ai-int8:unified-20260823
```

**TP2（双卡）**：

```bash
docker run -d \
  --name qwen38-tp2 \
  --network host --ipc host \
  --restart unless-stopped \
  --security-opt label=disable \
  --device /dev/kfd:/dev/kfd \
  --device /dev/dri/renderD128:/dev/dri/renderD128 \
  --device /dev/dri/renderD129:/dev/dri/renderD129 \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v /path/to/target:/models/target:ro \
  -v /path/to/draft:/models/draft:ro \
  -e PROFILE=tp2 \
  -e HIP_VISIBLE_DEVICES=0,1 \
  -e PORT=8062 \
  qwen38-k100ai-int8:unified-20260823
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
  qwen38-k100ai-int8:unified-20260823
```

> **renderD 编号**：`renderD128`–`renderD131` 是示例，必须根据你机器的实际拓扑替换。用 `ls /dev/dri/` 查看。
>
> **模型路径**：`/path/to/target` 和 `/path/to/draft` 替换为你实际的模型目录。

#### 6. 验证

```bash
# 等待模型加载（不同 Profile / 机器环境会有差异）
docker logs -f --tail=50 qwen38-tp4

# 健康检查
curl http://localhost:8068/health
# 应返回 200

# 测试请求
curl http://localhost:8068/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3.8-27B-W8A8-DFlash2-TP4","messages":[{"role":"user","content":"你好"}],"max_tokens":32,"temperature":0}'
```

---

### 方式 B：官方镜像 + 补丁构建

**说明**：拉取 SourceFind 官方 K100AI 基础镜像，打上本项目的**小补丁**（runtime patchset + 预编译 native extensions），然后 `docker build`。适合已经有官方镜像、或不想传输 5.65 GiB 完整镜像的场景。

**补丁内容**（合计约 2.3 MB）：

| 文件 | 大小 | 内容 |
|---|---:|---|
| `qwen38-k100ai-patchset.tar.gz` | ~86 KB | runtime patches（sitecustomize、repair、SGLang overlay） |
| `native_ext/prebuilt/*.so` | ~2.1 MB | 7 个预编译 gfx928 用户态 HIP 扩展 |

#### 1. 获取官方基础镜像

```text
harbor.sourcefind.cn:5443/dcu/admin/base/custom@sha256:366525b25f452f85eb0ea5813604a64f03c648627bc824bb498b56cf5a325dde
```

- 该镜像包含 SourceFind SGLang 0.5.12 / DTK 26.04 运行环境；
- 需要能访问 `harbor.sourcefind.cn:5443`（海光内网/授权环境）；
- 离线环境：先在有网机器 `docker pull` 后 `docker save` 导出，拷贝到目标服务器 `docker load`，然后在 `.env` 中设置 `BASE_IMAGE` 指向本地 tag。

#### 2. 获取本项目仓库

```bash
# 在线：固定到当前正式版 v1.1.0
git clone --branch v1.1.0 --depth 1 https://github.com/DocPang/qwen38-k100ai-int8-optimization.git
cd qwen38-k100ai-int8-optimization

# 离线：在有网机器 clone v1.1.0 后打包拷贝
# tar czf qwen38-k100ai-int8-optimization.tar.gz qwen38-k100ai-int8-optimization/
```

#### 3. 配置环境

```bash
cp .env.example .env
# 修改 .env 中的：
#   BASE_IMAGE=<本地已导入的官方镜像 tag>（离线时必填）
#   TARGET_MODEL=/path/to/target
#   DRAFT_MODEL=/path/to/draft
#   RENDER0=/dev/dri/renderD128
#   RENDER1=/dev/dri/renderD129
#   RENDER2=/dev/dri/renderD130
#   RENDER3=/dev/dri/renderD131
```

#### 4. 构建镜像

```bash
bash build_image.sh
```

构建过程：
1. 校验 7 个预编译 `.so` 的 SHA256；
2. 以官方基础镜像为基底 `docker build`；
3. 解压 patchset 到 `/opt/qwen38-k100ai/`，执行 `install_into_image.sh`；
4. 拷贝预编译 native extensions 到 `/data/qwen38-27b-k100ai-int8-opt/native_ext/`。

> 构建容器无 GPU 设备、无网络、非 privileged，不修改宿主机驱动。

#### 5. 准备模型 + 启动

模型权重按上方统一的“模型权重下载（二选一）”准备一次即可；挂载、启动与验证同方式 A 的第 4、5、6 步。

---

### 方式 C：全源码构建

**说明**：同方式 B，但 native extensions 从 C++ 源码重新编译（需要 DTK/hyhal 编译环境）。适合需要修改 kernel 源码、审计编译过程、或预编译 `.so` 与你的 DTK 版本不兼容的场景。

#### 与方式 B 的唯一区别

```bash
# 方式 B：使用预编译 .so（默认）
bash build_image.sh

# 方式 C：从源码重编 7 个 native extensions
REBUILD_NATIVE=1 bash build_image.sh
```

`REBUILD_NATIVE=1` 会启动一个临时编译容器（无 GPU、无网络、非 privileged），在官方基础镜像内用 DTK 编译 7 个 HIP `.so`，输出到 `.build/native/`，然后正常 `docker build`。

> 编译容器不修改宿主机驱动。如果编译失败，回退到方式 B 的预编译 `.so`。

其余步骤（获取官方镜像、配置 .env、准备模型、启动、验证）同方式 B。

---

### Profile 详情

- **只有 1 张 K100AI** → 看 **[TP1](TP1.md)**
- **有 2 张 K100AI** → 看 **[TP2](TP2.md)**
- **有 4 张 K100AI** → 看 **[TP4](TP4.md)**

TP4 离线算力服务器部署见：**[TP4 离线部署教程](TP4_OFFLINE_DEPLOY.md)**。

![TP4 formal 10-level](assets/tp4_10level.png)

---

## 项目分层

这个仓库以后按下面的逻辑维护，而不是把所有东西继续堆进一个 README：

| 层 | 内容 |
|---|---|
| **Profile** | TP1 / TP2 / TP4：不同 GPU 数量、context、运行参数与验收结果 |
| **Common runtime** | Qwen3.8 W8A8 compatibility、correctness repair、Radix Cache、DFlash2 backport |
| **Native kernels** | K100AI/gfx928 INT8 GEMV、LDS-x 等用户态 HIP 扩展 |
| **Long-context** | chunked prefill、paged/varlen repair、U036、Agent prefix cache |
| **Evidence** | 固定十档、quality gate、needle、无污染与稳定性结果 |

这样以后新增 TP8、MTP 或新的 speculative algorithm，只增加一个 profile / 技术模块，不再重开仓库。

---

## 为什么不是三个 GitHub 仓库

TP1、TP2 和 TP4 的共同部分远大于差异：

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
> Profile = TP1 / TP2 / TP4 / future TP8

---

## 上游与固定版本

- W8A8 target: [Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8](https://huggingface.co/Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8)（rev `417ede1e4524c8fdbb586ebdabc9cfc5d0760b3e`）
- DFlash2 draft: [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)（rev `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`）
- DFlash reference: `z-lab/dflash`（算法参考，私有仓库，无需下载）
- SGLang base: SourceFind SGLang 0.5.12 / DTK 26.04 for K100AI

当前验证的 SourceFind image digest：

```text
sha256:366525b25f452f85eb0ea5813604a64f03c648627bc824bb498b56cf5a325dde
```

详细 revision、patch、native kernel、correctness 根因与完整性能表请进入对应 Profile 文档。

---

## 发布策略

- **`main`**：当前推荐稳定代码与文档；
- **TP1 / TP2 / TP4**：在同一仓库内独立冻结 profile；
- **release tag**：只在 profile 通过正式 quality + 十档 + stability 门后发布；
- 实验分支和 microbench 不作为 README 的"冠军数据"。

当前 TP1 / TP2 / TP4 均已完成正式十档；TP1 / TP2 通过 short semantic、Arithmetic20 与 257.9K semantic / P95 needle，TP4 通过长上下文质量与稳定性门禁。

---

## 版本记录

| 版本 | 日期 | 主要更新 |
|---|---|---|
| **v1.1.0** | 2026-08-23 | TP1 / TP2 / TP4 三档统一正式发布；TP1 更新为当前最优长上下文方案；TP2 首次正式公开；统一 `PROFILE=tp1\|tp2\|tp4`；更新完整镜像与十档对比。 |
| **v1.0.1** | 2026-08-22 | 发布 TP1 / TP4 **统一完整 Docker 镜像**；`PROFILE=tp1\|tp4` 切换；增加夸克网盘直下与 SHA256；TP1 真机 smoke 通过；统一镜像纳入 DFlash2 sampling 防崩保护。 |
| **v1.0.0** | 2026-08-21 | TP4 正式 Champion：128K 49.45s / 88.68 tok/s，257.9K 132.25s / 72.49 tok/s；完成 cold 十档、质量与稳定性门禁。 |
| **v0.2.0** | 2026-08-21 | 仓库重构为统一 K100AI INT8/W8A8 优化项目，TP1 / TP4 作为 profile 维护。 |
| **v0.1.3** | 2026-08-21 | 默认使用已验证的预编译 K100AI 用户态 native extensions，源码重编改为可选路径。 |
| **v0.1.2** | 2026-08-21 | 增加安全的 K100AI native extension 源码构建流程，不修改宿主机驱动。 |
| **v0.1.1** | 2026-08-21 | 增加离线算力服务器部署说明。 |
| **v0.1.0** | 2026-08-21 | 首个公开版本。 |

---

## License / third-party notice

项目自身代码见 [LICENSE](LICENSE)，第三方来源与许可说明见 [NOTICE.md](NOTICE.md)。

---

本说明由 Qwen3.8-27B 生成。
