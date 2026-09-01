# TP4 Profile：Qwen3.8-27B W8A8 + DFlash2 on K100AI

[← 返回项目总览](README.md) · [TP1](TP1.md) · [TP2](TP2.md) · [三档最终十档](PERFORMANCE.md)

> **v1.3.0 / v30 FINAL（2026-09-01）**：TP4 production 已完成 request/lifecycle、工具、并发、公网与断连闭环；v1.3.0 发布不改动 GPU4-7 的既有 v30 服务。机器可读 authority 见 [`results/v1.3.0/TP4_V30_CHAMPION_MANIFEST_20260901.json`](results/v1.3.0/TP4_V30_CHAMPION_MANIFEST_20260901.json)。

> **当前公开说明（2026-08-27）**：TP4 与 TP1 / TP2 统一打包为 `PROFILE=tp1|tp2|tp4` 系列。v1.2.1 在 v1.2.0 RC2 基础上提供 TP4 raw-q8 性能 hotfix，修正 SourceFind flash-attn 260728 的 layout ABI 并恢复 single q8 verifier；TP1 / TP2 不变。已有 v1.2.0 镜像无需重新下载整包，见 [`hotfixes/v1.2.1`](hotfixes/v1.2.1/README.md)。本文保留 TP4 的完整技术说明，但不再使用研发期迭代号作为产品名称。

> ⚠️ **免责声明 / 风险提示**
>
> 本项目是社区研究成果，不是海光、SourceFind、Qwen、SGLang 或 DFlash2 官方发行版。部署会直接访问 GPU 设备，并依赖宿主机驱动、DTK、hyhal、PCIe/P2P 与 `renderD*` 设备映射。错误配置可能导致模型启动失败、GPU 不可用、现有业务中断，极端情况下可能需要重启服务器恢复。
>
> **请先备份宿主机现有配置。不要为了照抄本项目而直接覆盖驱动、修改 GRUB/IOMMU/ACS、执行未知 `setpci` 命令或在生产服务器上盲目试验。** 本仓库只在 Docker/用户态叠加优化，不会自动修改宿主机驱动。
>
> **特别说明：本项目不包含、不编译、也不安装 `amdgpu.ko`、DKMS 或任何宿主机 GPU 内核驱动。** v1.1.1 部署仓库提供 7 个已验证的 K100AI/gfx928 **用户态 PyTorch/HIP `.so`**（推理 runtime 与 v1.1.0 已验收版本相同），覆盖 TP1/TP2/TP4 的公共与 profile-specific native 依赖；对应 `.hip` 源码全部公开。只有用户主动选择源码重编时才会启动临时编译容器，而且该容器不映射 `/dev/kfd` 或 `renderD*`、不使用 privileged、无网络、只读挂载 `/opt/hyhal`。如果现有 `hy-smi`、`/opt/hyhal` 或官方容器环境本身不正常，请停止部署，不要让本项目替你“修驱动”。

> 状态：**当前长上下文方案 Champion**。正式 cold output256 十档、128K needle、257900-token exact retrieval、三轮 257.9K 确定性和 restart/OOM 门禁均已通过。

## 摘要

这项工作的目标很直接：在 **海光 K100AI** 上，把 Qwen3.8-27B 的 W8A8 版本做成一个适合长期 Agent 使用的高性能 SGLang 服务，并把 DFlash2 投机解码移植过来，同时解决长上下文、TP4、多卡通信和 gfx928 上若干实际运行问题。

最终稳定版本基于 SourceFind 的 K100AI SGLang 0.5.12 / DTK 26.04 镜像，目标模型使用 Qwen3.8-27B SmoothQuant W8A8，DFlash2 使用 z-lab 的 Qwen3.8-27B-DFlash2 草稿模型。稳定版在 TP4、BF16 KV、256K context、1M total-token KV budget、Radix Cache 开启的配置下完成了 512→257.9K 的十档 full-model 测试。

这不是单纯“加一个 DFlash2 参数”。整个过程包含：

- Qwen3.8 W8A8 在 SourceFind SGLang 0.5.12 上的兼容修复；
- gfx928 paged-varlen FlashAttention correctness 修复；
- K100AI 专用 INT8 GEMV / LDS-x / SwiGLU / RMSNorm / GDN 路径优化；
- TP4 rank-local 优化；
- 长上下文 prefill / decode 的专项调优；
- DFlash2 从上游实现向 K100AI + SourceFind SGLang 的 backport；
- DFlash2 在 TP4 下的 vocab-parallel selector 和 q=8 verifier 修复；
- PCIe / ACS / IOMMU / P2P 环境排查与验证；
- 128K Agent prefix / Radix Cache 使用方式的实际验证。

普通用户**不需要手工打 SGLang 补丁，也不需要先编译 native 扩展**。系列仓库直接提供 7 个已经在固定 SourceFind 镜像 / DTK / gfx928 环境验证过的用户态 `.so`，并附 SHA256；`bash build_image.sh` 默认校验这些二进制后直接构建派生镜像。7 个 `.hip` 源码和 `build_native.sh` 仍然公开，只有在你主动设置 `REBUILD_NATIVE=1` 时才会启动一次性、无 GPU 权限的编译容器重新构建。宿主机驱动、DTK、ACS、PCIe/IOMMU 都不会自动修改。

---

# 部署

从零部署请统一按 **[README 的完整 A / B / C 教程](README.md#从零部署先选一种方式三选一)** 操作；不要把本技术页当成第二套部署流程。

> **算力服务器不能联网？** 请看：[TP4 离线算力服务器部署教程](TP4_OFFLINE_DEPLOY.md)

TP4 的 Profile 特有设置只有这些：

| 项目 | TP4 |
|---|---|
| `PROFILE` | `tp4` |
| GPU 数 | 4 |
| render node | 映射 `RENDER0-3` 对应的 4 个设备 |
| 默认端口 | `8068` |
| served model | `Qwen3.8-27B-W8A8-DFlash2-TP4` |
| 构建后本地镜像（B/C） | `qwen38-k100ai-int8-series:local` |
| 完整成品镜像（A） | `qwen38-k100ai-int8:unified-20260823` |

- **方式 A**：`docker run` 设置 `PROFILE=tp4` 后，Docker 自动执行 `/opt/qwen38-k100ai/start.sh` → `/opt/qwen38-k100ai/entrypoint.tp4.sh`。
- **方式 B / C**：仓库根目录 `.env` 设置 `PROFILE=tp4`、正确的 `RENDER0-3` 和模型绝对路径，然后执行 `bash run.sh`。

`renderD*` 不能照抄验证机编号，必须先用 `ls -l /dev/dri/`、`hy-smi` 和服务器拓扑确认。

---

# 1. 验证环境

本项目最终稳定版本验证环境如下。

| 项目 | 验证值 |
|---|---|
| GPU | Hygon K100AI |
| GPU ISA | gfx928 |
| TP | 4 |
| Host kernel | `4.19.90-89.27.v2401.ky10.x86_64` |
| DTK | `DTK-26.04-DCC2602-0317` |
| SGLang | SourceFind 0.5.12 系列 |
| Target dtype | W8A8 / BF16 preserved layers |
| KV cache | BF16 |
| Context length | 262144 |
| Page size | 64 |
| Chunked prefill | 16384（128K 后自动切回 8192） |
| Max prefill tokens | 16384 |
| Max total tokens | 1048576 |
| CUDA Graph | bs=1 |
| Radix Cache | 开启 |
| DFlash2 draft block | 8 |
| P2P | 开启 |
| Custom all-reduce | 开启 |

注意：不同 K100AI 服务器的内核、DTK、PCIe switch、IOMMU group、renderD 设备号可能并不完全一样。因此下面的宿主机部分应当**先检查、先备份，再决定是否需要调整**。

---

# 2. 需要从网上或厂商仓库获取的原始文件

我们的补丁包不重新分发大模型和基础 Docker 镜像。建议严格固定版本，不要直接跟随 Hugging Face HEAD。

## 2.1 SourceFind SGLang 镜像

稳定版实际使用：

```text
harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-ubuntu22.04-dtk26.04-py3.10-20260620
```

测试机解析到的 repository digest：

```text
sha256:366525b25f452f85eb0ea5813604a64f03c648627bc824bb498b56cf5a325dde
```

推荐优先按 digest 拉取：

```bash
docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom@sha256:366525b25f452f85eb0ea5813604a64f03c648627bc824bb498b56cf5a325dde
```

SourceFind Harbor 是否允许所有外部用户匿名 pull，取决于厂商侧权限。如果无法拉取，需要从海光/SourceFind 正常渠道取得对应 SGLang 0.5.12 + DTK26.04 镜像。

## 2.2 Qwen3.8-27B W8A8 目标模型

Hugging Face：

```text
https://huggingface.co/Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8
```

我们验证使用的 revision：

```text
417ede1e4524c8fdbb586ebdabc9cfc5d0760b3e
```

下载时请使用本文上方的“夸克整合包 / HuggingFace”二选一方案。HuggingFace 方案固定到 `417ede1`（完整 revision 如上），不要直接下载当前 HEAD 后和本文成绩比较，因为上游仓库已经继续发生过变化。

## 2.3 DFlash2 草稿模型

Hugging Face：

```text
https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2
```

我们实际验证的 revision：

```text
50307d4c4cde6860d4eee73e2547cd786fe8e8a4
```

实际使用的权重只有约 3.6GB，服务器上的文件已和该 revision 做过 SHA256 字节级核对：

```text
config.json
873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980

model.safetensors
67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c
```

下载时同样使用本文上方的二选一方案。HuggingFace 方案固定到 `50307d4`（完整 revision 如上）；如果已经下载夸克整合包，就不要再单独下载 Draft。

该 checkpoint 的核心配置为：

- `DFlash2DraftModel`
- 5 层 draft network
- target layer ids：`[5,19,33,47,61]`
- block size：8
- sliding window：2048
- selector rank：256
- selector top-k：16
- conv kernel：2
- conv group：16

## 2.4 模型 metadata

正式 W8A8 serving **不需要下载 Qwen3.8 BF16/FP16 Base 权重**。运行需要的：

```text
preprocessor_config.json
video_preprocessor_config.json
```

已经直接打进统一镜像和 patchset，用户只需准备 W8A8 target 与 DFlash2 draft 两份模型。

## 2.5 DFlash 上游参考代码

参考实现：

```text
https://github.com/z-lab/dflash
```

我们移植时保存的参考 commit：

```text
07ebd93db9f472af339b644bb70221ad8428328a
```

我们并不是直接用该仓库启动服务，而是将 DFlash2 所需部分 backport 到 SourceFind SGLang 0.5.12。

---

# 3. 安装前：先备份驱动和宿主机现场

这是本文最不建议“一键化”的部分。

如果你的服务器已经有可工作的 K100AI 驱动和 DTK，**优先不要动它**。先记录现场，再判断和验证环境的差异。

建议建立一个备份目录：

```bash
mkdir -p /root/k100ai_before_qwen38
```

保存内核和模块信息：

```bash
uname -a > /root/k100ai_before_qwen38/uname.txt
modinfo amdgpu > /root/k100ai_before_qwen38/amdgpu_modinfo.txt
lsmod > /root/k100ai_before_qwen38/lsmod.txt
```

保存 DTK / hyhal 环境：

```bash
ls -la /opt/dtk* > /root/k100ai_before_qwen38/dtk_dirs.txt
readlink -f /opt/dtk > /root/k100ai_before_qwen38/dtk_link.txt
readlink -f /opt/hyhal > /root/k100ai_before_qwen38/hyhal_link.txt
```

保存 GPU 和拓扑：

```bash
/usr/local/hyhal/bin/hy-smi > /root/k100ai_before_qwen38/hysmi.txt 2>&1 || true
/usr/local/hyhal/bin/hy-smi --showtopo > /root/k100ai_before_qwen38/hysmi_topo.txt 2>&1 || true
lspci -tv > /root/k100ai_before_qwen38/lspci_tree.txt
lspci -vvv > /root/k100ai_before_qwen38/lspci_vvv.txt
```

保存 IOMMU group：

```bash
find /sys/kernel/iommu_groups -type l 2>/dev/null | sort \
  > /root/k100ai_before_qwen38/iommu_groups.txt
```

保存可能影响驱动/PCIe 的系统配置：

```bash
cp -a /etc/modprobe.d /root/k100ai_before_qwen38/ 2>/dev/null || true
cp -a /etc/default/grub /root/k100ai_before_qwen38/ 2>/dev/null || true
```

如果准备真的升级/替换驱动或 DTK，最好再做**系统盘快照或完整可回滚备份**。

本文不提供自动修改以下内容的脚本：

- 内核 / amdgpu 驱动；
- DTK 全局软链接；
- GRUB / IOMMU 参数；
- PCIe ACS；
- `setpci`；
- renderD 设备映射。

这些配置一旦处理错误，可能影响整机多卡通信甚至造成宿主机异常。

本项目确实对 PCIe / ACS / IOMMU / P2P 做过专项检查和 A/B，但这属于**宿主机通信环境调优**，不是我们重新开发了一套 GPU 驱动。

---

# 4. 我们实际做了哪些优化

## 4.1 W8A8 兼容和 correctness

SourceFind SGLang 0.5.12 对这个 Qwen3.8 SmoothQuant checkpoint 的 compressed-tensors ignore 规则并不能直接正确匹配全部模块，所以首先补了 W8A8 compatibility 和 sparse tune-cache fallback。

更关键的是，我们定位到 gfx928 上 vendor vLLM-style paged-varlen FlashAttention 的一个 correctness 问题：在特定 multi-token prefill 路径中，`q>=5` 分支可能不写输出。这个问题经过 full-model A/B、isolated reference、sentinel-write 等测试确认。

因此最终服务不是简单走原始 paged prefill，而是保留正确的 decode consumer，同时对 multi-token prefill 做经过验证的替代路由。

## 4.2 K100AI INT8 decode kernel

我们针对 Qwen3.8 的真实 M=1 shape 写了多组 K100AI/gfx928 HIP INT8 GEMV：

- output projection；
- gate/up 和 full-QKV 等 body shape；
- deep down projection；
- TP4 row-parallel rank-local shape。

同时加入：

- SwiGLU → INT8 producer fusion；
- RMSNorm → dynamic INT8 producer；
- GDN QKVZ / BA activation quant 复用；
- TP4 rank-local LDS-x；
- compact lm-head shortlist。

这些优化主要改善 decode 端，而不是单纯改善 TTFT。

## 4.3 TP4

TP4 不是把 TP1 参数直接乘四。

我们针对 rank-local shape 重新做了：

- K5120 LDS-x；
- row-parallel LDS-x；
- TP4 compact head；
- GDN RMS→QKVZ；
- BA24 INT8 shadow；
- P2P / custom all-reduce A/B。

早期稳定十档曾使用 P2P ON、custom all-reduce OFF；当前长上下文方案已完成 matched 验收并将 **P2P ON + custom all-reduce ON** 固化为当前正式配置。

## 4.4 长上下文

Qwen3.8-27B 是 64 层 hybrid 结构，其中 48 层 linear attention、16 层 full attention。

我们补了 16K→128K 每 8K 一档的 dense prefill curve，确认所谓“64K 断崖”并不是一个离散阈值，而主要来自 full attention 的累计 `O(N²)` prefill 成本；decode 侧则会随着历史 KV 变长产生约 `O(N)`/token 的额外成本。

在此基础上形成 U036 长 KV 路径，专门针对 TP4 rank-local QH6/KVH1/D256 的 q=8192 full-attention prefill 调 kernel geometry。

## 4.5 DFlash2 移植

当前 SourceFind SGLang 0.5.12 并不原生支持我们使用的 DFlash2 checkpoint，所以做了 backport。

主要包括：

- DFlash2 draft model config/weight 支持；
- selector walk Triton kernel；
- local convolution；
- candidate selection；
- TP-safe vocab-parallel selector；
- target hidden-state / draft cache 对接；
- TP4 下的权重加载审计。

另一个 K100AI 特有问题是：DFlash2 target verifier 的 q=8 会碰到前面提到的 gfx928 q>=5 paged-varlen no-write wrapper 分支。当前方案不再走这个 wrapper：在严格 `batch1 / q=8 / QH6 / KVH1 / D256 / page64 / BF16` geometry 下直接调用底层 raw `paged_attention`。该 q8 路径在多个长 KV 点与原 `2×native q4` workaround 做过 bitwise gate；如果 geometry 不匹配则 fail-closed，而不是泛化放宽。

---

# 5. 补丁成果包

论坛附件只需要一个文件：

```text
qwen38-k100ai-patchset.tar.gz
```

当前 TP4 patchset SHA256：

```text
d25f5412f34a47138c3f249991c5625fd567a6e79d1ef000015217acf7c889d9
```

这个包里不是模型，也不是 Docker 镜像，只是我们的修改成果。

## 5.1 SGLang 主补丁

```text
sglang_dflash2_k100ai.patch
```

相对固定 SourceFind SGLang 镜像，它只涉及少量 DFlash2 核心文件。

| 文件 | 用途 |
|---|---|
| `sglang/kernels/ops/speculative/dflash2_selector.py` | 新增 DFlash2 selector walk Triton kernel |
| `sglang/kernels/ops/speculative/__init__.py` | speculative kernel package 入口 |
| `sglang/srt/models/dflash.py` | DFlash2DraftModel、checkpoint config、draft forward、TP selector |
| `sglang/srt/speculative/dflash_worker.py` | DFlash2 draft/verify 工作流、cache/hidden-state 协调 |
| `sglang/srt/speculative/dflash_utils.py` | DFlash/DFlash2 candidate、采样和辅助逻辑 |
| `sglang/srt/layers/attention/triton_ops/extend_attention.py` | DFlash2 draft attention / K100AI Triton attention 适配 |

我们已经验证过该 unified patch 可以在固定镜像抽出的 pristine SGLang 上直接 `patch -p1` 应用。

## 5.2 Target runtime patch

补丁包内的 `runtime_patch/` 是 target 模型从 correctness 到 TP4 性能优化形成的依赖链。每层都只做一个或少数几个明确改动，便于回退和定位。

| patch | 作用 |
|---|---|
| `runtime_patch_sglang_w8a8_compat` | 修 Qwen3.8 W8A8 compressed-tensors ignore 匹配 |
| `runtime_patch_sglang_n4_sparse_w8a8_cache` | sparse W8A8 tune-cache miss 安全 fallback |
| `runtime_patch_sglang_n5_compact_head` | M=1 compact lm-head shortlist |
| `runtime_patch_sglang_n5_compact_head_gdnint8` | GDN QKVZ+BA INT8 fused input path |
| 功能模块 | 作用 |
|---|---|
| W8A8 compatibility / producer fusion | W8A8 兼容、SwiGLU/RMSNorm producer 优化 |
| native INT8 GEMV | output / body / GDN / deep-down / K5120 等 M=1 路径 |
| TP4 rank-local LDS-x | TP4 row-parallel 与 K5120 shape 优化 |
| gfx928 paged-varlen repair | 修复 multi-token paged attention correctness |
| long-context scheduler | 128K 内 q16K，超长上下文切 q8K，257.9K partial-tail 修复 |
| DFlash2 verifier | 固定 geometry 下的 q=8 verifier 与 selector 路径 |

这些目录看起来层数较多，主要是因为研究期间坚持单变量 A/B 和可回退。正式长期维护版后续可以再扁平化，但当前 Draft 先保留已经实测过的组合关系。

## 5.3 K100AI 用户态 HIP native extension（不是驱动）

系列仓库提供 7 个已验证的用户态 native extensions 及对应 HIP 源码，位于 [`native_ext/`](native_ext/)；TP4 使用其中与自身 shape 对应的子集：

| 文件 | 用途 |
|---|---|
| `native_ext/k100_int8_gemv_v7.hip` | output projection M=1 INT8 GEMV |
| `native_ext/k100_int8_gemv_generic_v2.hip` | gate_up/full_qkv/GDN 等通用 shape |
| `native_ext/k100_int8_gemv_deep_v4.hip` | K17408 deep-down projection |
| `native_ext/k100_int8_gemv_tp4_row_ldsx_v1.hip` | TP4 row-parallel rank-local LDS-x |
| `native_ext/build_native.py` | 用 SourceFind 镜像内 PyTorch/hipcc 编译上述 4 个扩展 |
| `build_native.sh` | 最小权限编译 wrapper：无网络、无 GPU 设备、只读 `/opt/hyhal` |

v1.1.1 部署仓库默认提交 7 个已验证的预编译 `.so`，位于 `native_ext/prebuilt/`，其 SHA256 记录在 `native_ext/PREBUILT_SHA256SUMS`。Dockerfile 默认使用这些文件。`build_native.sh` 作为备用，会把重新编译的产物写到 `.build/native/`；Dockerfile 会在构建时用这些本机构建结果覆盖预编译版本。

再次强调：这些 `.so` 是 PyTorch 可加载的**用户态扩展**，不是 `amdgpu.ko`，不会通过 DKMS 安装，不会替换宿主机驱动。

## 5.4 其他必要文件

| 文件 | 用途 |
|---|---|
| `launch_sglang_require_sitecustomize.py` | 强制 sitecustomize 加载失败时直接终止，避免“补丁没生效但服务照样启动” |
| `qwen38_chat_template.jinja` | 本项目验证使用的 Qwen3.8 chat template |
| `ninja_native_wrapper.sh` | 规避启动链中 Python helper / ninja 的历史递归问题 |
| `reference_start_command.sh` | 稳定版启动参数参考，不建议不看内容就直接执行 |

---

# 6. Docker 构建与部署方式

本仓库不重新分发 31.7GB 的 SourceFind 基础镜像，而是用：

```dockerfile
FROM harbor.sourcefind.cn:5443/dcu/admin/base/custom@sha256:366525b25f452f85eb0ea5813604a64f03c648627bc824bb498b56cf5a325dde
```

作为固定底座。默认构建流程很简单：

1. 校验 `native_ext/prebuilt/` 中 7 个已验证用户态 `.so` 的 SHA256；
2. `docker build` 对镜像内原始 SGLang 应用 `sglang_dflash2_k100ai.patch`；
3. 放入 gfx928 correctness、W8A8、TP4、U036、BA24、q16/当前长上下文方案 等 runtime patch；
4. 安装 7 个 当前长上下文方案 exact tune-cache JSON（M8192/M16384/M3968 family）；
5. 把 7 个预编译 `.so` 放入固定 runtime 路径并逐个检查；
6. 保存 U036/q16 所需的 SourceFind Triton GQA 原始实现并设置运行时入口。

普通用户直接执行：

```bash
bash build_image.sh
```

这一步默认**不会启动编译容器，也不会映射 GPU**。如果你明确希望从源码重编，再执行：

```bash
REBUILD_NATIVE=1 bash build_image.sh
```

重编属于可选备用路径，不是正常部署前置步骤。

## 6.1 `.env` 需要修改什么

```text
TARGET_MODEL=/你的/Qwen3.8-27B-SmoothQuant-W8A8-INT8
DRAFT_MODEL=/你的/Qwen3.8-27B-DFlash2
RENDER0=/dev/dri/renderDxxx
RENDER1=/dev/dri/renderDxxx
RENDER2=/dev/dri/renderDxxx
RENDER3=/dev/dri/renderDxxx
PORT=8068
```

TP4 正式默认：

```text
U036_PROFILE=ranklocal_bm64_w4_preloadv
U036_SPLIT_KV=4
CHUNKED_PREFILL_SIZE=16384
CUSTOM_AR=1
```

镜像内还会固定下面的长上下文策略：

```text
q16K KV split = 4
q16 numerical repair = only KV==131072
long chunk switch = prefix>=131072 -> 8K chunks
257900 tail = q3948 fixed split8
DFlash2 q8 verifier = native raw paged_attention
selector top-k = 16
```

这些已经作为**同一个版本**完成 512→257.9K cold output256 十档，不再是实验开关。`q=3948` 尾块的 causal block 映射也已从 floor 修正为 ceil，解决旧实验中出现过的重复写、漏尾 block、随机 SHA 与 SIGSEGV 风险。

---

# 7. 镜像内置的稳定版启动参数

以下是正式 cold 十档与长上下文质量门实际使用的核心配置。

```text
TP=4
PP=1
attention_backend=fa3
kv_cache_dtype=bfloat16
page_size=64
context_length=262144
mem_fraction_static=0.90
chunked_prefill_size=16384
max_prefill_tokens=16384
max_total_tokens=1048576
max_running_requests=4
cuda_graph_bs=1
Radix Cache=ON
P2P=ON
custom all-reduce=ON
DFlash2 draft tokens=8
DFlash2 steps=1
```

关键环境变量：

```bash
export HSA_FORCE_FINE_GRAIN_PCIE=1
export W8A8_SUPPORT_METHODS=1
export SGLANG_KV_LAYOUT_DCU_FA=true
export SGLANG_Q38_TP4_COMPACT_HEAD_M1=1
export SGLANG_Q38_TP4_ROW_LDSX_M1=1
export SGLANG_Q38_TP4_K5120_LDSX_M1=1
export SGLANG_Q38_GDN_BA_FUSED_M1=1
export SGLANG_Q38_SWIGLU_INT8_M1=1
export SGLANG_Q38_RMS_GDN_INT8_M1=1
export SGLANG_Q38_TP4_RMS_QKVZ_M1=1
export SGLANG_Q38_TP4_BA24_INT8_M1=1
export SGLANG_Q38_NATIVE_OUT_GEMV_M1=1
export SGLANG_Q38_NATIVE_BODY_GEMV_M1=1
export SGLANG_Q38_NATIVE_GDN_SPLIT_M1=1
export SGLANG_Q38_DEEP_DOWN_GEMV_M1=1
```

当前派生镜像已经把 SGLang patch 安装进 site-packages；运行时只需要让冻结的 DFlash2/q16 composition 先于其它 Python 路径：

```bash
export PYTHONPATH=/data/qwen38-dflash2-k100ai/runtime_patch_dflash_tp4_q16k_agent128k_v1${PYTHONPATH:+:$PYTHONPATH}
export SGLANG_REQUIRED_SITECUSTOMIZE_PREFIX=/data/qwen38-dflash2-k100ai/runtime_patch_dflash_tp4_q16k_agent128k_v1
```

SGLang server 的核心参数（使用 SourceFind SGLang 0.5.12 自带的原生 CLI）：

```bash
sglang serve \
  --model-path /data/my_models/Qwen/Qwen3.8-27B-SmoothQuant-W8A8-INT8 \
  --host 0.0.0.0 \
  --port 8068 \
  --served-model-name Qwen3.8-27B-W8A8-DFlash2-TP4 \
  --chat-template /data/qwen38-dflash2-k100ai/runtime_assets/qwen38_chat_template.jinja \
  --dtype bfloat16 \
  --kv-cache-dtype bfloat16 \
  --tp-size 4 \
  --pp-size 1 \
  --attention-backend fa3 \
  --page-size 64 \
  --mamba-scheduler-strategy extra_buffer \
  --max-mamba-cache-size 16 \
  --cuda-graph-bs 1 \
  --disable-piecewise-cuda-graph \
  --context-length 262144 \
  --mem-fraction-static 0.90 \
  --chunked-prefill-size 16384 \
  --max-prefill-tokens 16384 \
  --pack-paged-kv-to-varlen auto \
  --pack-paged-kv-to-varlen-min-q-tokens 2048 \
  --pack-paged-kv-to-varlen-min-kv-tokens 8192 \
  --max-total-tokens 1048576 \
  --max-running-requests 4 \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path /data/qwen38-dflash2-k100ai/models/Qwen3.8-27B-DFlash2 \
  --speculative-draft-model-quantization unquant \
  --speculative-draft-attention-backend triton \
  --speculative-num-steps 1 \
  --speculative-num-draft-tokens 8 \
  --enable-metrics
```

> 正式 `unified-20260823` 镜像 entrypoint 仍会在 `sglang.launch_server` 前增加 `launch_sglang_require_sitecustomize.py` 的 fail-closed 检查，防止 runtime patch 加载失败后静默退回 stock SGLang；该包装器不修改 server 参数或推理逻辑。手工排障时可直接使用上面的原生 `sglang serve`。

注意：真正使用 Docker 时还需要按自己的机器传入 `/dev/kfd` 和正确的 `/dev/dri/renderD*`。测试机 GPU4-7 对应的设备号不能假设在所有服务器上都一样，所以本文不建议直接复制固定 renderD 编号。

---

# 8. TP4 正式十档结果

统一 contract：**每档先 flush Radix、single request、cold prefill、output256、512→257.9K**。十档均 `contaminated=false`，整轮完成后容器 `restart=0 / OOM=false / health=200`。

| Context | TTFT | Decode tok/s | Total |
|---:|---:|---:|---:|
| 512 | 0.409s | 100.74 | 2.940s |
| 2K | 1.078s | 119.03 | 3.220s |
| 4K | 2.363s | 95.65 | 5.029s |
| 8K | 2.400s | 110.88 | 4.700s |
| 12K | 4.119s | 91.06 | 6.920s |
| 16K | 4.603s | 113.10 | 6.858s |
| 32K | 10.012s | **128.25** | 12.000s |
| 64K | 22.184s | **102.21** | 24.679s |
| 128K | **49.450s** | **88.68** | **52.325s** |
| 257.9K | **132.249s** | **72.49** | **135.767s** |

摘要：

- 全十档 Decode 中位：**101.48 tok/s**；
- 16K：**113.10 tok/s**；
- 64K：**102.21 tok/s**；
- 128K：**49.45s TTFT / 88.68 tok/s**；
- 257.9K：**132.25s TTFT / 72.49 tok/s**。

正式机器可读证据：[results/tp4_10level_20260821.json](results/tp4_10level_20260821.json)。

![TP4 当前长上下文方案 formal 10-level](assets/tp4_10level.png)

这组数据来自**同一个正式配置**，不是从不同实验里挑最好点拼接。

---

# 9. Agent 128K / Radix Cache 的实际意义

这套配置并不是只追求“冷 128K benchmark”。

在长期 Agent 场景里，大部分 workspace / system prompt / tool context 会被重复使用，所以 Radix/prefix cache 的收益很明显。

我们实际测到同一个 128K workspace 再请求时：

- `cached-token=122880`
- `new-token=8192`
- prefix hit = **93.75%**
- TTFT ≈ **7.58s**
- Decode ≈ **67.75 tok/s**
- Total ≈ **11.34s**

因此这套服务更适合“长驻 Agent / 多轮 workspace”，而不是每次都完全不同的 128K 冷 prompt。

---

# 10. 长上下文实现

当前方案不再使用“整条曲线固定一个 chunk 大小”的策略，而是按已经验证的上下文区间做确定性路由：

1. **128K 以内优先 q16K prefill**，full-attention 使用 `BM64/BN64/w4/preloadV` 和长 KV split4；
2. **仅在 `KV==131072`** 对 q16 做 scheduler-equivalent `2×q8` 数值 repair，避免阈值策略外溢到更长上下文；
3. **prefix≥128K 后自动切回 8K chunk**，因为实测超长 KV 区间 q8K 比继续 q16K 更高效；
4. 257900-token canonical prompt 的最后有效 full-attention query 是 **q=3948**。该尾块使用 split8，并把 vendor causal two-pass 的第二遍 block 映射从 floor 改成 ceil：`ceil(Q/BLOCK_M)-1-pid`；
5. DFlash2 q=8 verifier 在固定 TP4 `QH6/KVH1/D256/page64/BF16` geometry 下直接调用 raw `paged_attention`。该路径在 16K、64K、128K、131328、257.9K KV 点都与原来的 `2×q4` workaround 做过 bitwise gate；
6. 镜像内置 7 个 exact tune-cache JSON，覆盖当前长上下文 profile 使用的 M8192/M16384/M3968 W8A8 family。

q3948 的 ceil-block 修复是稳定性的关键。旧实验用 floor block count 时会产生**一个 block 重复写、最后 partial block 漏写**，从而出现同输入 SHA 随机变化、DFlash acceptance 归零，甚至一次 scheduler `SIGSEGV(-11)`。修复后 isolated split8 连续输出 bitwise identical，full-model 257.9K 三轮也得到完全相同 SHA：

| Run | TTFT | Decode | SHA |
|---:|---:|---:|---|
| 1 | 132.309s | 70.95 tok/s | `25caec31…1f9a` |
| 2 | 132.301s | 72.52 tok/s | `25caec31…1f9a` |
| 3 | 132.345s | 72.21 tok/s | `25caec31…1f9a` |

三轮均 `restart=0 / OOM=false`。原始 artifacts：

- [stability run1](results/tp4_257900_stability_1_20260821.json)
- [stability run2](results/tp4_257900_stability_2_20260821.json)
- [stability run3](results/tp4_257900_stability_3_20260821.json)

质量门同样使用真实长上下文：

- 128K thinking needle：精确返回 `Q38U036V2P0K100AI`，见 [artifact](results/tp4_128k_midneedle_20260821.json)；
- 257900-token exact retrieval：在 95% 位置植入唯一 access code，模型精确返回 `Q38QTAIL8K100AI`，见 [artifact](results/tp4_257900_p95_needle_20260821.json)。

---

# 11. 已知限制

1. 本项目针对 K100AI / gfx928 和 SourceFind SGLang 0.5.12 做了大量 shape-specific 优化，不保证直接适用于其他 GPU。
2. 厂商 Harbor 镜像的获取权限可能因用户环境不同。
3. 默认预编译 `.so` 只针对本文锁定的 SourceFind 镜像 / DTK / gfx928 组合验证；如果你使用不同底座、不同 PyTorch/DTK ABI，建议执行 `REBUILD_NATIVE=1 bash build_image.sh` 重新编译，不要强行混用。
4. 宿主机驱动、DTK、ACS、IOMMU、renderD 映射不要盲目照抄测试机；本项目不会替你安装或修复这些系统组件。
5. DFlash2 提高 decode，但冷长 prompt 仍需生成 draft-side KV，因此它不是免费的 TTFT 加速器。
6. `q3948 split8` 和 raw q8 paged verifier 都是**严格 shape/geometry fail-closed** 的 K100AI 特化；换模型、换 head ratio、换 page size 或 dtype 时不要直接放宽限制，必须重新做 correctness/stability gate。
7. 当前 257900-tail fast path只对精确 audited geometry 生效；其它非整齐超长尾块会回退到正确 parent，而不是猜测使用同一 split。

---

# 12. 结语

这次工作真正的难点不是某一个 kernel，而是把多个层次同时闭合：

**W8A8 checkpoint → SGLang correctness → gfx928 attention → K100AI INT8 kernel → TP4 rank-local → 长上下文 → DFlash2 → Agent prefix cache。**

为了方便他人复现，仓库继续只分发用户态 patchset、tune cache、native extension 与机器可读 evidence；基础镜像、target 权重和 DFlash2 权重仍从上游固定版本获取。

**TP4 已完成整条 cold 十档、128K/257.9K 质量门和多轮稳定性门。**
