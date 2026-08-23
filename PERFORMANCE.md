# TP1 / TP2 / TP4 最终性能对比

> Qwen3.8-27B W8A8 · DFlash2 · Hygon K100AI

统一测试口径：**canonical corpus / output=256 / DFlash2 / cold / 每档独立 cache flush / contaminated=false**。

## 十档 TTFT / Decode

| 上下文 | TP1 TTFT | TP1 Decode | TP2 TTFT | TP2 Decode | TP4 TTFT | TP4 Decode |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 5.93s | 23.92 tok/s | 0.42s | 70.61 tok/s | 0.41s | 100.74 tok/s |
| 2K | 7.21s | 28.65 tok/s | 1.62s | 86.08 tok/s | 1.08s | 119.03 tok/s |
| 4K | 12.41s | 25.65 tok/s | 4.15s | 77.09 tok/s | 2.36s | 95.65 tok/s |
| 8K | 8.42s | 27.58 tok/s | 3.89s | 83.24 tok/s | 2.40s | 110.88 tok/s |
| 12K | 13.74s | 28.29 tok/s | 12.92s | 85.27 tok/s | 4.12s | 91.06 tok/s |
| 16K | 16.55s | 26.59 tok/s | 9.59s | 81.27 tok/s | 4.60s | 113.10 tok/s |
| 32K | 34.01s | 27.30 tok/s | 23.23s | 79.05 tok/s | 10.01s | 128.25 tok/s |
| 64K | 72.66s | 31.68 tok/s | 40.36s | 73.32 tok/s | 22.18s | 102.21 tok/s |
| 128K | 174.68s | 33.90 tok/s | 90.66s | 49.98 tok/s | 49.45s | 88.68 tok/s |
| 257.9K | 466.44s | 24.30 tok/s | 234.28s | 53.10 tok/s | 132.25s | 72.49 tok/s |

## 十档总耗时

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

## 长上下文 Scaling

| 上下文 | TP1 / TP2 TTFT | TP2 / TP4 TTFT | TP1 / TP4 TTFT |
|---:|---:|---:|---:|
| 64K | 1.80× | 1.82× | 3.28× |
| 128K | 1.93× | 1.83× | 3.53× |
| 257.9K | 1.99× | 1.77× | 3.53× |

## 正式验收状态

- **TP1**：10/10 正式十档；short semantic PASS；Arithmetic20=18/20（仅历史 case8/17）；257.9K canonical semantic PASS；257.9K P95 needle PASS；accept=true。
- **TP2**：10/10 正式十档；short semantic PASS；Arithmetic20=18/20（仅历史 case8/17）；257.9K canonical semantic PASS；257.9K P95 needle PASS；accept=true。
- **TP4**：10/10 正式十档；257.9K 长上下文质量与稳定性门禁通过；当前四卡 Champion。

## Authority 文件

- `results/tp1_acceptance_20260823.json`
- `results/tp2_acceptance_20260822.json`
- `results/tp4_10level_20260821.json`
- `results/tp1_tp2_tp4_10level_20260823.json`（统一对比）


