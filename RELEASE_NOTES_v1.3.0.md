# v1.3.0 Release Notes — TP1 / TP2 / TP4 v30

发布日期：2026-09-01

v1.3.0 是 correctness-first 的 v30 production profile release。它冻结 TP1、TP2、TP4 三种拓扑的最终功能合同、cache-resume 合同、生产启动入口、显存高水位守护和机器可读证据；不会为了追逐单点性能重新引入已淘汰的 oldFA、noHybrid 或 tailpad 路线。

## 发布状态

| Profile | 状态 | 生产入口 | 关键边界 |
|---|---|---|---|
| TP1 | FINAL | GPU2 / 8042 / `q38-tp1-v30-prod-gpu2` | unified v1.2.2 image，max-running=1，mem=0.84 |
| TP2 | FINAL | GPU0-1 / 8062 / `q38-tp2-v30-prod-gpu01` | legacy SourceFind image，max-running=4，mem=0.88 |
| TP4 | FINAL | GPU4-7 / 8068 / `q38-tp4-v30-prod` | 本版本不改动既有 TP4 v30 production |

TP1、TP2 的 release manifests 均为 `accept=true`。发布时三容器均为 running、restart=0、OOM=false，端点 idle，最近 30 分钟 critical log 计数为 0。

## TP1 v30

组合：`Final-v2 -> Mamba checkpoint8192 -> Hybrid14 -> TP124 v30 common`。

- 16K、64K、139,265 cache-resume：2× cold deterministic，cached output IDs 与 cold 完全一致；
- function matrix：9/9；
- no-thinking arithmetic：18/20，只错冻结基线 case 8/17；
- thinking arithmetic：20/20；
- stream / non-stream disconnect：自动回收、post-abort smoke PASS，无显式 cleanup；
- real Hermes tool / mixed c3：已有 v30 exact PASS authority。

完整 same-build formal10 中 128K / 257.9K 曾落到 20.89 / 16.47 tok/s。完全 idle、cache guard 回收高水位后的同 prompt cold repeat 为：

| Context | TTFT | Decode | 相对历史 |
|---:|---:|---:|---:|
| 128K | 175.32s | 32.16 tok/s | Decode 约 -5.1% |
| 257.9K | 470.29s | 22.76 tok/s | TTFT +0.83%，Decode -6.34% |

因此该低谷按偶发 runtime/speculative 状态记录，不作为功能发布 blocker。仓库保留完整 formal10 与 targeted repeat，绝不拼接成“最佳十档”。

## TP2 v30

组合：`longtail-v1 / {3,23} -> Mamba checkpoint8192 -> TP124 v30 common`。

- 16K、64K、139,265 cache-resume PASS；
- function matrix 9/9；
- no-thinking arithmetic 18/20，仅冻结基线 case 8/17；
- thinking arithmetic 20/20；
- stream / non-stream disconnect PASS；
- formal10 10/10 complete，257.9K Decode 59.03 tok/s。

TP2 必须继续使用验证过的 legacy SourceFind image。统一 v1.2.2 image 在 TP2 arithmetic 上发生 11–12/20 的回归；v1.3.0 不以“形式统一”交换 correctness。

## 显存高水位守护

`idle_cache_flush_guard.py` 是 SGLang 外部守护器，只在 running/waiting/used_tokens 全为 0 时调用官方 `/flush_cache`。生产已验证 99% -> 94%，restart=0、OOM=false。`/v1/loads` timeout 时 fail-open，不会 flush。

## 发布资产

- [`v1.3.0/`](v1.3.0/)：精确 overlay source、launchers、guard、systemd templates 和非覆盖式 installer；
- [`results/v1.3.0/TP1_V30_RELEASE_MANIFEST_20260901.json`](results/v1.3.0/TP1_V30_RELEASE_MANIFEST_20260901.json)；
- [`results/v1.3.0/TP2_V30_RELEASE_MANIFEST_20260901.json`](results/v1.3.0/TP2_V30_RELEASE_MANIFEST_20260901.json)；
- [`results/v1.3.0/TP4_V30_CHAMPION_MANIFEST_20260901.json`](results/v1.3.0/TP4_V30_CHAMPION_MANIFEST_20260901.json)；
- [`results/qwen38_k100ai_v130_release_evidence_20260901.tar.gz`](results/qwen38_k100ai_v130_release_evidence_20260901.tar.gz)，SHA256 `a4b46213ba8f856a1676ffac5179bf376a2c0ad455dc49916c081a08162af919`。

## 已知边界

- no-thinking arithmetic 18/20 是历史冻结基线，不是 v30 新回归；
- GPU3 research slot 的 TP1 cold nondeterminism 没有在 GPU2 production 复现，GPU3 不作为正式 authority；
- TP2 128K 有局部性能低谷，但 correctness 完整；
- 偶发性能状态以后继续研究，不回滚 correctness 层。
