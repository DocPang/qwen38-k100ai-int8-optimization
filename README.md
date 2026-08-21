# Qwen3.8-27B W8A8 + DFlash2 在海光 K100AI 上的 TP4 优化与复现说明

> 版本：v0.1 · 2026-08-21  
> 状态：稳定十档配置已验证；最新 U036 split4 长上下文 profile 已包含但默认不开启。

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

普通用户**不需要手工打 SGLang 补丁**。本仓库用一个约 448KB 的 patchset 在 `docker build` 时自动基于固定 SourceFind 镜像生成本地优化镜像；如果基础镜像已经存在就直接复用，不存在则 Docker 自动拉取。宿主机驱动、DTK、ACS、PCIe/IOMMU 仍然不自动修改，避免不同机器上的高风险操作。

---

# 最简单的部署方式

先下载两份模型：

```bash
hf download Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8 \
  --revision 417ede1e4524c8fdbb586ebdabc9cfc5d0760b3e \
  --local-dir /data/models/Qwen3.8-27B-SmoothQuant-W8A8-INT8

hf download z-lab/Qwen3.8-27B-DFlash2 \
  --revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4 \
  --local-dir /data/models/Qwen3.8-27B-DFlash2
```

然后：

```bash
cp .env.example .env
# 编辑 .env：只改模型路径和本机 4 个 renderD 设备号

docker compose up -d --build
```

Docker 会自动执行：

`复用/拉取固定 SourceFind 基础镜像 → 应用 K100AI/SGLang/DFlash2 patch → 安装已验证 native kernel → 生成本地优化镜像 → 启动 TP4 服务`。

默认端口为 `8068`，默认使用完整十档验证过的稳定 profile。基础镜像约 31.7GB，但本仓库自身只有不到 1MB；后续更新补丁时 Docker 会复用已有基础层，不需要重新下载 31.7GB。 本项目在验证机上的实际派生镜像大小为 `31,700,057,086` bytes，对比基础镜像 `31,696,485,809` bytes，虚拟大小仅增加约 **3.6MB**。

> **重要：** `.env.example` 中的 `renderD132-135` 只是验证机示例。请先用 `ls -l /dev/dri/`、`hy-smi`、PCIe 拓扑确认你自己的 4 张 K100AI 对应设备号。

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
| Chunked prefill | 8192 |
| Max prefill tokens | 16384 |
| Max total tokens | 1048576 |
| CUDA Graph | bs=1 |
| Radix Cache | 开启 |
| DFlash2 draft block | 8 |
| P2P | 开启 |
| Custom all-reduce | 稳定十档版本关闭 |

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

建议：

```bash
hf download Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8 \
  --revision 417ede1e4524c8fdbb586ebdabc9cfc5d0760b3e \
  --local-dir /data/my_models/Qwen/Qwen3.8-27B-SmoothQuant-W8A8-INT8
```

不要直接下载当前 HEAD 后和本文成绩比较，因为上游仓库已经继续发生过变化。

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

下载：

```bash
hf download z-lab/Qwen3.8-27B-DFlash2 \
  --revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4 \
  --local-dir /data/qwen38-dflash2-k100ai/models/Qwen3.8-27B-DFlash2
```

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

## 2.4 Qwen 官方 BF16 仓库

官方模型：

```text
https://huggingface.co/Qwen/Qwen3.8-27B
```

正式 W8A8 serving 不需要再次下载完整 52GB BF16 权重，但我们的稳定启动环境会从官方目录补两个 preprocessing metadata 文件：

```text
preprocessor_config.json
video_preprocessor_config.json
```

如果只部署纯文本，也可以根据自己的 SGLang 版本判断是否需要它们。

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

稳定 DFlash2 十档版本使用 P2P ON、custom all-reduce OFF。后续 CAR 属于另一个研究分支，不和本文稳定十档混用。

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

另一个 K100AI 特有问题是：DFlash2 target verifier 的 q=8 会碰到前面提到的 gfx928 q>=5 paged-varlen no-write 路径。因此最终 TP4 patch 将严格符合条件的 batch1 causal q=8 verification 拆成两次 native q=4，避免错误分支。

---

# 5. 补丁成果包

论坛附件只需要一个文件：

```text
qwen38-k100ai-patchset.tar.gz
```

当前 Draft SHA256：

```text
bf629b2cd5a4323da300a1aaf1eee048d45f5c1f608309ec6798721bb00d9b62
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
| `runtime_patch_sglang_n6_swiglu_exact` | SwiGLU→INT8 producer fusion |
| `runtime_patch_sglang_n7_rms_gdn_exact` | GDN input RMSNorm→INT8 producer |
| `runtime_patch_sglang_u004_native_out_gemv` | native output-projection INT8 GEMV |
| `runtime_patch_sglang_u008_native_body_gemv` | gate_up/full_qkv 等 native body GEMV |
| `runtime_patch_sglang_u010_native_gdn_split` | native GDN QKVZ/BA split consumer |
| `runtime_patch_sglang_u016_deep_down` | deep MLP down-projection GEMV |
| `runtime_patch_sglang_u019_k5120_full5` | K5120 full-prefetch GEMV 路径 |
| `runtime_patch_sglang_u022_k5120_ldsx` | K5120 activation LDS-x |
| `runtime_patch_sglang_tp4_k5120_ldsx_v1` | TP4 rank-local K5120 shape |
| `runtime_patch_sglang_tp4_compact_head_v1` | TP4 local compact head |
| `runtime_patch_sglang_tp4_row_ldsx_v1` | TP4 row-parallel LDS-x |
| `runtime_patch_sglang_gfx928_paged_varlen_repair_v1/repair.py` | gfx928 multi-token paged-varlen correctness repair |
| `runtime_patch_sglang_tp4_row_ldsx_varlenfix_v1` | 将 TP4 decode 优化和 paged-varlen repair 组合 |
| `runtime_patch_sglang_tp4_u036_longkv_v1` | TP4 q8K long-KV prefill U036 路径 |
| `runtime_patch_sglang_tp4_u036_rmsqkvz_v1` | U036 + RMS→QKVZ prequant producer |
| `runtime_patch_sglang_tp4_u036_rmsqkvz_ba24_v1` | 最终 BA24 target 组合 |
| `dflash_tp4_agent128k_sitecustomize.py` | 在 BA24 target 上再安装 DFlash2 q=8 verifier 修复 |

这些目录看起来层数较多，主要是因为研究期间坚持单变量 A/B 和可回退。正式长期维护版后续可以再扁平化，但当前 Draft 先保留已经实测过的组合关系。

## 5.3 HIP native extension

| 文件 | 用途 |
|---|---|
| `k100_int8_gemv_v7.hip` | output projection M=1 INT8 GEMV |
| `k100_int8_gemv_generic_v2.hip` | gate_up/full_qkv/GDN 等通用 shape |
| `k100_int8_gemv_deep_v4.hip` | K17408 deep-down projection |
| `k100_int8_gemv_tp4_row_ldsx_v1.hip` | TP4 row-parallel rank-local LDS-x |

包内同时保留了我们验证过的 gfx928 `.so`，方便比对；更推荐在自己的固定镜像和驱动环境中从 `.hip` 重新编译。

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

作为固定底座。`qwen38-k100ai-patchset.tar.gz` 会在 build 阶段自动完成以下工作：

1. 对镜像内原始 SGLang 应用 `sglang_dflash2_k100ai.patch`；
2. 放入 gfx928 correctness、W8A8、TP4、U036、BA24 等 runtime patch；
3. 放入 K100AI native INT8 GEMV/LDS-x 扩展；
4. 保存 U036 所需的 SourceFind Triton GQA 原始实现；
5. 安装 DFlash2 TP4 q=8 verifier 组合层；
6. 对关键 Python 文件做 build-time syntax gate；
7. 设置运行时入口，直接启动 Qwen3.8-27B W8A8 + DFlash2 TP4 服务。

所以普通使用者只需要 `docker compose up -d --build`。想审计成果时，再解压 patchset 查看源码和 unified patch。

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

稳定十档 profile 默认：

```text
U036_PROFILE=legacy_bm64_w8
U036_SPLIT_KV=1
```

如果要试最新 64K/128K 长上下文优化：

```text
U036_PROFILE=ranklocal_bm64_w4_preloadv
U036_SPLIT_KV=4
```

后者已经完成 64K/128K output256 和 128K needle gate，但本文没有把它与旧稳定十档拼成一条“假十档”。完整十档重新冻结后再更新默认值。

---

# 7. 镜像内置的稳定版启动参数

以下是 2026-08-21 11:37 完成十档时的核心配置，不是下午继续试验的 CAR/split4 分支。

```text
TP=4
PP=1
attention_backend=fa3
kv_cache_dtype=bfloat16
page_size=64
context_length=262144
mem_fraction_static=0.90
chunked_prefill_size=8192
max_prefill_tokens=16384
max_total_tokens=1048576
max_running_requests=4
cuda_graph_bs=1
Radix Cache=ON
P2P=ON
custom all-reduce=OFF
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

`PYTHONPATH` 顺序必须让 DFlash2 composition patch 在前、patched SGLang overlay 在后，例如：

```bash
export PYTHONPATH=/data/qwen38-dflash2-k100ai/runtime_patch_dflash_tp4_agent128k_v1:/data/qwen38-dflash2-k100ai/work/sourcefind_sglang_overlay_tp4/python
```

SGLang server 的核心参数：

```bash
python3 -u /data/qwen38-27b-k100ai-int8-opt/scripts/launch_sglang_require_sitecustomize.py \
  --model-path /data/my_models/Qwen/Qwen3.8-27B-SmoothQuant-W8A8-INT8 \
  --host 0.0.0.0 \
  --port 8068 \
  --served-model-name Qwen3.8-27B-W8A8-DFlash2-TP4-Agent128K \
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
  --disable-custom-all-reduce \
  --context-length 262144 \
  --mem-fraction-static 0.90 \
  --chunked-prefill-size 8192 \
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

注意：真正使用 Docker 时还需要按自己的机器传入 `/dev/kfd` 和正确的 `/dev/dri/renderD*`。测试机 GPU4-7 对应的设备号不能假设在所有服务器上都一样，所以本文不建议直接复制固定 renderD 编号。

---

# 8. 稳定版十档结果

统一 contract：single request、output256、512→257.9K。

| Context | TTFT | Decode tok/s | Total |
|---:|---:|---:|---:|
| 512 | 5.968s | 112.30 | 8.239s |
| 2K | 5.653s | 114.40 | 7.882s |
| 4K | 7.010s | 105.72 | 9.422s |
| 8K | 2.448s | 112.82 | 4.708s |
| 12K | 3.881s | 123.13 | 5.952s |
| 16K | 4.776s | 108.85 | 7.119s |
| 32K | 10.422s | 113.13 | 12.676s |
| 64K | 24.291s | 66.10 | 28.149s |
| 128K | 63.715s | 68.59 | 67.433s |
| 257.9K | 242.208s | 53.25 | 246.997s |

摘要：

- 512–16K Decode 中位：约 **112.56 tok/s**；
- 全十档 Decode 中位：约 **110.58 tok/s**；
- 16K：**108.85 tok/s**；
- 64K：**66.10 tok/s**；
- 128K：**68.59 tok/s**；
- 257.9K：**53.25 tok/s**。

这组数据的价值是同一个版本完成完整十档，不是从不同实验里挑最好点拼接。

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

# 10. 最新长上下文优化进展（尚未冻结）

在稳定十档之后，我们又继续优化了 U036 TP4 long-prefill。

当前研究头主要变化：

```text
U036 profile = ranklocal_bm64_w4_preloadv
SGLANG_Q38_TP4_U036_SPLIT_KV=4
```

已完成的 full-model output256 点：

| Context | 稳定十档 TTFT | 最新研究 TTFT | 最新 Decode |
|---:|---:|---:|---:|
| 64K | 24.29s | **22.70s** | **83.66 tok/s** |
| 128K | 63.71s | **52.89s** | **79.29 tok/s** |

128K TTFT 相比稳定十档进一步下降约 17%。128K 中段 needle 语义检查已经完成。

257.9K 目前已经完成 output1 TTFT 点约 245.82s，但还没有按同一新配置重新完成 output256 十档，所以**本文暂不把 split4 当成正式发布默认配置**。等完整十档和稳定性门完成后，只需要更新 U036 patch 和本节数据，不需要推翻整个复现方法。

---

# 11. 已知限制

1. 本项目针对 K100AI / gfx928 和 SourceFind SGLang 0.5.12 做了大量 shape-specific 优化，不保证直接适用于其他 GPU。
2. 厂商 Harbor 镜像的获取权限可能因用户环境不同。
3. 预编译 `.so` 只应在 ABI/DTK/PyTorch 接近的环境中使用；不同环境建议重编。
4. 驱动、DTK、ACS、IOMMU、renderD 映射不要盲目照抄测试机。
5. DFlash2 提高 decode，但冷长 prompt 仍需生成 draft-side KV，因此它不是免费的 TTFT 加速器。
6. 最新 split4 长上下文优化还在收尾，正式发布时可能替换本文稳定十档中的 U036 配置。

---

# 12. 结语

这次工作真正的难点不是某一个 kernel，而是把多个层次同时闭合：

**W8A8 checkpoint → SGLang correctness → gfx928 attention → K100AI INT8 kernel → TP4 rank-local → 长上下文 → DFlash2 → Agent prefix cache。**

为了方便他人复现，论坛发布时只需要提供本文和一个小型 patchset；镜像、target 权重和 DFlash2 权重仍从上游固定版本获取。

等最新长上下文 TTFT 分支完成整条十档后，再更新本文的性能表和 U036 patch，即可形成正式 v1.0。
