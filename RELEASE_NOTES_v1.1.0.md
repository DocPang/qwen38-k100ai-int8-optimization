# v1.1.0 — TP1 / TP2 / TP4 Series

发布日期：2026-08-23

这是第一次把 **TP1、TP2、TP4** 作为一个完整系列正式发布。

## 主要变化

- **TP1 更新为当前正式方案**：此前单卡 128K 方案不再作为最终性能引用；新增 BM128 long-KV 32K→253952、257.9K ceil-causal qtail、262K context 与完整 257.9K quality gate。
- **TP2 首次正式公开**：包含 cross-rank shared-scale layers32–47、TP2 row-LDSx、BM128 long-KV、257.9K qtail、raw-q8 verifier。
- **TP4 保持当前冠军配置**：与 TP1/TP2 统一进入同一系列入口。
- 统一 Docker runtime：`PROFILE=tp1|tp2|tp4`。
- 补齐 TP1/TP2/TP4 的完整 native 依赖闭包：系列共提供 **7 个**已验证用户态 gfx928 extensions，包括 TP2 row-LDSx、K5120 full5 / LDS-x 等正式运行时实际依赖，并全部附对应 HIP 源码。
- patchset 增加 TP1/TP2 完整 runtime closure、TP2 tune-cache、processor overrides。
- `run.sh` 按 Profile 只映射所需 render node：TP1=1、TP2=2、TP4=4。
- 新增统一 authority 文件与三档十档总表。

## 最终十档

统一口径：canonical corpus / output256 / DFlash2 / cold / 每档独立 cache flush / contaminated=false。

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

完整 Total / scaling / quality：[`PERFORMANCE.md`](PERFORMANCE.md)。

## TP1 旧版修正

此前 TP1 128K TTFT 为 231.13s。当前正式结果为 **174.68s**，并新增 257.9K 正式结果 **466.44s / 24.30 tok/s**。

旧研究期 TP1 文件已从公开仓库移除。

## 正式验收

- TP1：`accept=true`；short semantic PASS；Arithmetic20=18/20（仅历史 case8/17）；257.9K semantic PASS；257.9K P95 needle PASS。
- TP2：`accept=true`；同口径质量门通过。
- TP4：formal 10/10 + 长上下文 quality/stability 已完成。

## Patchset

`qwen38-k100ai-patchset.tar.gz`

```text
SHA256 861903598b683ba592d68f795ac10493cbbd5e47f9b6c2757cf542d10b2930d4
```

发布前已在 K100AI 服务器上基于固定 SourceFind image digest 完成真实 `docker build`：

```text
sha256:366525b25f452f85eb0ea5813604a64f03c648627bc824bb498b56cf5a325dde
```

构建过程中 TP1/TP2/TP4 runtime py_compile、TP2/TP4 tune-cache 数量、7 个 native `.so` 均通过 fail-closed 检查。

## 升级提醒

完整镜像已重新打包为支持 TP1 / TP2 / TP4 的统一版本。

## 完整镜像

重新打包的统一镜像支持 `PROFILE=tp1|tp2|tp4`，不包含模型权重。用户只需额外挂载 W8A8 target 与 DFlash2 draft，**不需要 BF16/FP16 Base 权重**。

- Docker tag：`qwen38-k100ai-int8:unified-20260823`
- 压缩包：`qwen38-k100ai-int8-unified-20260823.docker.tar.zst`
- 压缩大小：5.65 GiB（6,065,184,632 bytes）
- SHA256：`6d14588722b0fea0ab66a53e2810385d1f9999a9cd78c8e1d2e6640c744f2b14`
- 完整镜像下载：[夸克网盘 full_images](https://pan.quark.cn/s/e7626123faa0?pwd=M8Fr)，提取码：`M8Fr`；GitHub Release 保留 patchset、十档数据与整包 SHA256。

下载整包后按 SHA256 校验，再执行 `zstd -dc ... | docker load`。
