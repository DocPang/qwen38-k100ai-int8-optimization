# v1.3.1 Final Image Validation

最终验证对象：

```text
image: qwen38-k100ai-int8:v1.3.1
id:    sha256:534f9512a3d5217c8f65f11183ed75db19e7aa3df1ecc4da1ba993e98793973d
file:  Qwen3.8-K100AI-v1.3.1-final-image.tar.zst
sha:   4e43edd8a0cf5ee0e501aefb051170587b1da3003392fdd2d32a1d9712e11d8f
```

本文件记录的是最终单镜像 authority 验证，不是早期 overlay candidate。

## TP1

最终 image 在 GPU2 authority 卡验证：

- `PROFILE=tp1` 正常启动；
- Final-v2 -> checkpoint8192 -> Hybrid14 -> v30 common 链完整；
- Chat exact `OK`：PASS；
- named tool：PASS；
- 64K cache-resume：PASS；
- cold repeats：2；
- cold distinct SHA：1；
- cached tokens：57,344；
- replay：8,192；
- cached output == cold output；
- 请求中 HBM 约 98%，idle 后回约 94%；
- allocator trim：rank0 / 60s。

GPU3 research slot 曾复现 cold nondeterminism，因此最终 TP1 correctness authority 使用 GPU2；GPU2 final-image gate 为 deterministic PASS。

## TP2

最终 image 使用内置 legacy flash-attn 260602 sidecar：

```text
flash_attn      -> /opt/qwen38-k100ai/tp2_legacy_site/flash_attn
flash_attn_2_cuda -> /opt/qwen38-k100ai/tp2_legacy_site/flash_attn_2_cuda...
```

启动时 fail-closed 校验 sidecar 路径。

### Function / grammar / sampling

完整 function matrix：9/9 PASS。

覆盖：

- greedy exact；
- non-greedy；
- min-p；
- seeded sampling；
- JSON schema；
- regex grammar；
- `tool_choice=required`；
- named tool；
- mixed concurrency。

mixed c4：PASS。

### Arithmetic

- thinking arithmetic20：20/20；
- no-thinking arithmetic20：18/20；
- 仅历史固定 case 8 / 17 错误；
- `integrityPass=true`；
- 没有新增 correctness 回归。

### Cache-resume

16K：PASS。

```text
cached=8192
replay=8192
cold distinct SHA=1
cached==cold
```

64K：PASS。

```text
cached=57344
replay=8192
cold distinct SHA=1
cached==cold
```

### Allocator rootfix

- rank0 独立 trim：PASS；
- rank1 独立 trim：PASS；
- production interval：60s；
- idle HBM 约 95%；
- restart=0；
- OOM=false。

## TP4

最终 image 在 GPU0-3 以 `PROFILE=tp4` 启动，启动时没有挂载任何研发 runtime patch，仅挂权重、`/opt/hyhal` 和 GPU device。

镜像内关键 runtime 与当前 TP4 production 做逐 SHA 对账，包括两个 production canonical alias，最终关键目标文件 12/12 MATCH。

### Function / Agent concurrency

完整 function matrix：9/9 PASS。

mixed c8：PASS。

覆盖 tool / JSON / regex / seeded sampling 共存，无跨请求污染。

### Cache-resume

最终 release authority 使用与历史 TP4 rootfix gate 相同的参数口径：

```text
seed=938231
prime_gap=16384
out_tokens=32
cold_repeats=2
```

64K：PASS。

```text
cached=49152
replay=16384
cold distinct SHA=1
cached==cold
```

说明：一次额外的非-authority 探索测试使用 `prime_gap=1000 / out_tokens=64` 时观察到 cold trajectory 双模态，但 cached 输出仍匹配其中一个 cold。为避免混用不同 gate 口径，正式 release authority 使用历史冻结的 `prime_gap=16384 / out_tokens=32`，结果 deterministic PASS。

### Allocator rootfix

四个 scheduler rank 均持续打印独立 60s trim：

```text
rank0 PASS
rank1 PASS
rank2 PASS
rank3 PASS
```

64K 请求期间四卡 HBM 约 63%，idle 后回约 58%。

容器：restart=0 / OOM=false。

## Agent / API compatibility

Fresh checks：

- TP1 / TP2 / TP4 `max_completion_tokens`：PASS；
- TP1 / TP2 / TP4 `image_url`：正常 224x224 RGB PNG，HTTP 200；
- simple `tool_choice=auto`：三 topology PASS；
- TP4 `auto + stream` tool call：PASS；
- TP1 / TP2 在工具执行轮 `enable_thinking=false` 后长字符串 streaming tool-call PASS；
- TP2 1K / 2K chars tool argument：PASS；
- TP2 4K chars tool argument：出现重复并撞生成上限，因此不作为推荐用法。

推荐：planning turn thinking=true；tool execution turn thinking=false。

## Docker artifact round-trip

最终镜像经过完整归档回环：

1. `docker save qwen38-k100ai-int8:v1.3.1`；
2. `zstd -6`；
3. `zstd -t`：PASS；
4. 删除 image tag；
5. `zstd -dc ... | docker load`：PASS；
6. load 后 Image ID：

```text
sha256:534f9512a3d5217c8f65f11183ed75db19e7aa3df1ecc4da1ba993e98793973d
```

与原镜像一致。

Mac 下载完成后再次 SHA256：

```text
4e43edd8a0cf5ee0e501aefb051170587b1da3003392fdd2d32a1d9712e11d8f
```

与服务器归档 SHA 一致，Mac 本地 `zstd -t` PASS。

## Production restore

完成 final-image 验证后，原生产服务已经恢复：

- TP1 / 8042：HTTP 200，exact `OK`；
- TP2 / 8062：HTTP 200，exact `OK`；
- TP4 / 8068：HTTP 200，exact `OK`；
- restart=0；
- OOM=false；
- restart policy=`unless-stopped`；
- old external full-flush guards：disabled + inactive。

## Release verdict

```text
TP1: PASS
TP2: PASS
TP4: PASS
allocator rootfix: PASS
Agent/function gates: PASS
Docker archive round-trip: PASS
production restore: PASS
```
