# TP1 Profile：Qwen3.8-27B W8A8 + DFlash2 on K100AI

> **1× K100AI · 262K context · 单卡最终发布版**
> 状态：**ACCEPTED / FINAL**（2026-08-23）

> 当前公开版本已经替代此前的单卡 128K 方案，并把正式验收范围扩展到 257.9K。旧研究期证据不再放入公开仓库。

## 1. 最终配置

- Target：Qwen3.8-27B SmoothQuant W8A8 INT8；
- Draft：Qwen3.8-27B DFlash2；
- TP=1；
- context length = **262144**；
- KV cache = BF16；
- page size = 64；
- chunked prefill = 8192；
- max prefill tokens = 16384；
- CUDA Graph bs=1；
- speculative draft tokens = 8；
- max running requests = 1（正式验收口径）。

## 2. 当前长上下文方案

TP1 当前正式方案将 TP2/TP4 后续研究中已经证明有效的长上下文技术重新回移到单卡，同时保留 TP1 自身的 correctness 路径：

- **BM128/BN64/w8 long-KV**：q=8192，KV 32K → 253952；
- **exact 257900 qtail**：q=3948 / KV=257900；
- vendor causal block 起点使用 **ceil-causal** 修复；
- DFlash2 q=8 verifier 保留已验证的 **2× native q4 q8split**；
- **Early-Triton N=1**：真实请求首个 TARGET_VERIFY round 的 layer 7/15 走 corrected Triton，后续恢复 q8split CUDA Graph；
- gfx928 native INT8 GEMV、RMS→INT8、SwiGLU→INT8、GDN/compact-head 等公共优化栈继续保留。

### 为什么 TP1 没有启用 raw-q8 verifier

TP2/TP4 上 raw-q8 verifier 有明确收益，但 TP1 的 QH24/KVH4 几何没有通过完整晋级门：64K 可行，128K 出现数值漂移，257.9K 隔离验证出现 VMFault。因此 TP1 **主动保留 q8split**，不为了形式统一强行启用不稳定路径。

## 3. 正式质量验收

| Gate | 结果 |
|---|---|
| Short semantic | **PASS** |
| Arithmetic20 | **18/20** |
| 已知 miss | case8 / case17，仅历史基线问题 |
| Critical case15 | **303 / PASS** |
| 257.9K canonical semantic | **PASS**，输出 `Q38LONGSEMANTICOK` |
| 257.9K P95 needle | **PASS** |
| contamination | **0 / PASS** |
| Final manifest | **accept=true** |
| runtime | restart=0 · OOM=false |

Authority：[`results/tp1_acceptance_20260823.json`](results/tp1_acceptance_20260823.json)

## 4. 正式十档性能

统一口径：canonical corpus / output256 / DFlash2 / cold / 每档独立 cache flush / uncontaminated。

| Prompt | TTFT | Decode | Total |
|---:|---:|---:|---:|
| 512 | 5.93s | 23.92 tok/s | 16.59s |
| 2K | 7.21s | 28.65 | 16.11s |
| 4K | 12.41s | 25.65 | 22.35s |
| 8K | 8.42s | 27.58 | 17.67s |
| 12K | 13.74s | 28.29 | 22.75s |
| 16K | 16.55s | 26.59 | 26.14s |
| 32K | 34.01s | 27.30 | 43.35s |
| 64K | **72.66s** | **31.68** | 80.71s |
| 128K | **174.68s** | **33.90** | 182.20s |
| 257.9K | **466.44s** | **24.30** | 476.93s |

### 相比旧 TP1 的关键修正

旧版 128K TTFT 为 **231.13s**。当前正式结果为 **174.68s**，下降约 **24.4%**；同时首次把 TP1 正式验收范围扩展到 **257.9K**。

64K 也从旧版 85.13s 降到 **72.66s**。

更重要的是长上下文 scaling 恢复合理：

- 64K：TP1 / TP2 TTFT ≈ **1.80×**；
- 128K：≈ **1.93×**；
- 257.9K：≈ **1.99×**。

这说明旧版 128K 的异常变陡已经被修正。

## 5. 部署

统一镜像/统一构建版本使用：

```bash
PROFILE=tp1
HIP_VISIBLE_DEVICES=0
PORT=8090
```

或 `.env` 中设置 `PROFILE=tp1` 后：

```bash
bash run.sh
```

只映射 **1 个**对应的 render node，详见 [README](README.md)。

## 6. 证据

- [Acceptance manifest](results/tp1_acceptance_20260823.json)
- [257.9K semantic](results/tp1_257900_semantic_20260823.json)
- [257.9K P95 needle](results/tp1_257900_p95_needle_20260823.json)
- [Arithmetic20](results/tp1_arithmetic20_20260823.json)
- [TP1/TP2/TP4 统一十档](PERFORMANCE.md)

返回：[项目首页](README.md) · [TP2](TP2.md) · [TP4](TP4.md)
