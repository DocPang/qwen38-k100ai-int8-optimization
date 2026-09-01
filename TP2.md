# TP2 Profile：Qwen3.8-27B W8A8 + DFlash2 on K100AI

> **v1.3.0 / v30 FINAL（2026-09-01）**：当前正式链为 `longtail-v1/{3,23} -> Mamba checkpoint8192 -> TP124 v30 common`。16K/64K/139,265 cache-resume PASS；function matrix 9/9；thinking arithmetic 20/20；no-thinking 保持历史 18/20[8,17]；stream/non-stream disconnect 自动回收；257.9K formal10 Decode 59.03 tok/s。TP2 必须固定 legacy SourceFind image，不能使用 unified image。正式 manifest 见 [`results/v1.3.0/TP2_V30_RELEASE_MANIFEST_20260901.json`](results/v1.3.0/TP2_V30_RELEASE_MANIFEST_20260901.json)。以下 2026-08-22 内容保留为 v1.2.x 基线说明。

> **2× K100AI · 262K context · 双卡最终发布版**  
> v1.2.x 基线状态：**ACCEPTED / FINAL**（2026-08-22）

## 1. 最终配置

- Target：Qwen3.8-27B SmoothQuant W8A8 INT8；
- Draft：Qwen3.8-27B DFlash2；
- TP=2；
- context length = **262144**；
- KV cache = BF16；
- page size = 64；
- chunked prefill = 8192；
- max prefill tokens = 16384；
- CUDA Graph bs=1；
- custom all-reduce = enabled；
- P2P = enabled；
- speculative draft tokens = 8；
- shared-scale layer range = **32–47**；
- TP2 compact head top-k = 1024。

## 2. 当前技术栈

TP2 是本系列中的双卡长上下文方案：

- **RowParallel W8A8 shared-scale correctness repair**：rank-local activation scale 改为跨 rank MAX shared scale；
- layer-selective shared-scale：正式冻结在 **layers 32–47**；
- TP2 row-LDSx / K5120-LDSx native INT8 kernel；
- **BM128/BN64/w8 long-KV**：q=8192，KV 32K → 253952；
- **exact 257900 qtail**：q=3948 / KV=257900 + ceil-causal；
- **raw-q8 verifier**：TP2 QH12/KVH2/D256/page64/BF16 隔离门验证为 bitwise equal，并替换旧 2×q4 verifier；
- DFlash2 + CUDA Graph + custom AR + P2P；
- gfx928 native M=1 INT8 acceleration stack。

raw-q8 isolated gate 在 64K / 128K / 257.9K 分别约有 **2.22× / 2.26× / 2.04×** attention speedup，并保持与 q8split bitwise equality。

## 3. 正式质量验收

| Gate | 结果 |
|---|---|
| Short semantic | **PASS** |
| Arithmetic20 | **18/20** |
| 已知 miss | case8 / case17 |
| Critical case15 | **303 / PASS** |
| 257.9K canonical semantic | **PASS** |
| 257.9K P95 needle | **PASS** |
| contamination | **0 / PASS** |
| Final manifest | **accept=true** |
| runtime | restart=0 · OOM=false |

Authority：[`results/tp2_acceptance_20260822.json`](results/tp2_acceptance_20260822.json)

## 4. 正式十档性能

统一口径：canonical corpus / output256 / DFlash2 / cold / 每档独立 cache flush / uncontaminated。

| Prompt | TTFT | Decode | Total |
|---:|---:|---:|---:|
| 512 | 0.42s | 70.61 tok/s | 4.03s |
| 2K | 1.62s | 86.08 | 4.58s |
| 4K | 4.15s | 77.09 | 7.45s |
| 8K | 3.89s | 83.24 | 6.95s |
| 12K | 12.92s | 85.27 | 15.91s |
| 16K | 9.59s | 81.27 | 12.73s |
| 32K | 23.23s | 79.05 | 26.45s |
| 64K | **40.36s** | **73.32** | 43.83s |
| 128K | **90.66s** | **49.98** | 95.76s |
| 257.9K | **234.28s** | **53.10** | 239.09s |

## 5. 长上下文表现

当前正式方案修复了早期 TP2 在 64K 后的异常退化：

- 64K TTFT：旧 RC1 ≈ 101.17s → **40.36s**；
- 128K：旧路径 ≈ 245.03s → **90.66s**；
- 257.9K：旧路径 ≈ 1122.58s → **234.28s**。

最终 TP2 相对 TP4 的 TTFT 比例稳定在：

- 64K：**1.82×**；
- 128K：**1.83×**；
- 257.9K：**1.77×**。

## 6. 部署

从零部署请统一按 **[README 的完整 A / B / C 教程](README.md#从零部署先选一种方式三选一)** 操作，本页只补充 TP2 的 Profile 差异：

| 项目 | TP2 |
|---|---|
| `PROFILE` | `tp2` |
| GPU 数 | 2 |
| render node | 映射 `RENDER0-1` 对应的 2 个设备 |
| 默认端口 | `8062` |
| served model | `Qwen3.8-27B-W8A8-DFlash2-TP2` |

- **方式 A**：完整镜像 `docker run` 时设置 `PROFILE=tp2`，Docker 会自动进入镜像内 `/opt/qwen38-k100ai/entrypoint.tp2.sh`。
- **方式 B / C**：仓库根目录 `.env` 设置 `PROFILE=tp2`、正确的 `RENDER0/1`，然后执行 `bash run.sh`。

## 7. 证据

- [Acceptance manifest](results/tp2_acceptance_20260822.json)
- [257.9K semantic](results/tp2_257900_semantic_20260822.json)
- [257.9K P95 needle](results/tp2_257900_p95_needle_20260822.json)
- [Arithmetic20](results/tp2_arithmetic20_20260822.json)
- [TP1/TP2/TP4 统一十档](PERFORMANCE.md)

返回：[项目首页](README.md) · [TP1](TP1.md) · [TP4](TP4.md)
