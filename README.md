# Qwen3.8-27B INT8/W8A8 on K100AI

> **SGLang performance optimization for Hygon K100AI**
> TP1 / TP4 · DFlash2 · Agent long context · gfx928 native kernels

> ⚠️ **免责声明 / 风险提示**
>
> 本项目是社区研究成果，不是海光、SourceFind、Qwen、SGLang 或 DFlash2 官方发行版。项目依赖宿主机已经正常工作的 K100AI 驱动、DTK/hyhal、Docker 与 GPU 设备映射。错误的 GPU/PCIe/驱动操作可能导致现有业务中断，极端情况下需要重启服务器恢复。
>
> **本仓库不会自动安装、替换或重新编译宿主机 `amdgpu.ko` / DKMS 驱动，也不会自动修改 GRUB、IOMMU、ACS 或执行 `setpci`。** 如果 `hy-smi`、`/dev/kfd`、`/opt/hyhal` 或官方 SourceFind 容器本身不正常，请先停止部署，不要让本项目替你“修驱动”。

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

> TP1 和 TP4 的正式十档范围并不完全相同，因此不要直接拿“十档平均值”做横向排名。更有意义的是比较相同上下文下的 TTFT / decode，以及你的 GPU 成本。

### 完整统一镜像（推荐）

如果不想重新构建 30GB 级基础镜像，可以直接使用已经验收的 **TP1 / TP4 统一 Docker 运行时镜像**：

- 夸克网盘：[Qwen3.8-K100AI-Unified-20260822](https://pan.quark.cn/s/1e9abeef509a?pwd=duZx)
- 提取码：`duZx`
- 文件：`qwen38-k100ai-int8-unified-20260822.docker.tar.zst`
- 压缩后：约 **5.67 GiB**
- Docker tag：`qwen38-k100ai-int8:unified-20260822`
- SHA256：`e3e2874939b540a935191939fe6309e583a7bf1808f6341f07aba447740d7557`

```bash
zstd -dc qwen38-k100ai-int8-unified-20260822.docker.tar.zst | docker load
```

同一只镜像通过 `PROFILE=tp1` / `PROFILE=tp4` 切换单卡与四卡运行参数。模型权重仍采用外部挂载，不重复塞进镜像。完整校验、profile 参数和注意事项见 **[full_images/README.md](full_images/README.md)**。

### 我只有 1 张 K100AI

看 **[TP1 Agent128K](TP1.md)**。

当前正式验收版本使用 DFlash2 + q8split + first-round corrected Triton，context length 为 147456，已经完成 64K/128K needle、Arithmetic20、critical case 与十档性能门禁。

### 我有 4 张 K100AI

直接看 **[TP4 Agent256K](TP4.md)**。

这是当前公开打包最成熟的版本，支持 262144 context、DFlash2、Radix Cache、TP4 native kernel 与 longctx-v8 分段 prefill。正式 cold output256 十档已经完整通过；128K 为 **49.45s TTFT / 88.68 tok/s**，257.9K 为 **132.25s / 72.49 tok/s**，并通过 128K needle、257900-token exact retrieval、三轮 257.9K 确定性与 restart/OOM 门禁。

![TP4 longctx-v8 formal 10-level](assets/tp4_longctx_v8_10level.png)

离线算力服务器见：**[TP4 离线部署教程](TP4_OFFLINE_DEPLOY.md)**。

---

## 源码构建 TP4 Stable Profile

> 根目录的 `Dockerfile / build_image.sh / run.sh` 继续保留为可审计、可重建的 **TP4 Stable Profile** 源码路线。若只是部署，优先使用上面的统一预构建镜像；若需要自行重建、修改 patch 或 native kernel，再使用本节。TP1 / TP4 的统一预构建入口见 `full_images/`。

先准备两份模型：

- Target：`Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8`
  - validated revision: `417ede1e4524c8fdbb586ebdabc9cfc5d0760b3e`
- Draft：`z-lab/Qwen3.8-27B-DFlash2`
  - validated revision: `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`

然后：

```bash
cp .env.example .env
# 修改 target/draft 模型路径与本机 4 个 renderD 设备号

bash build_image.sh
bash run.sh
```

默认构建使用仓库内已经验证过的 4 个 gfx928 用户态 HIP `.so`，并校验 SHA256；**不会启动编译容器**。如果你明确需要从源码重编：

```bash
REBUILD_NATIVE=1 bash build_image.sh
```

源码重编只是备用路径。临时编译容器无 GPU 设备、无网络、非 privileged，不修改宿主机驱动。

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
- 实验分支和 microbench 不作为 README 的“冠军数据”。

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
