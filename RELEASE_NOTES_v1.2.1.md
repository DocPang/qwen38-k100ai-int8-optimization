# v1.2.1 — TP4 raw-q8 performance hotfix

这是 v1.2.0 的 **TP4-only 小版本热修复**。TP1 / TP2、模型权重、SourceFind SGLang 版本和 flash-attn 版本均不变。

## 背景

v1.2.0 为修复 SourceFind flash-attn 260728 环境下 TP4 raw-q8 verifier 的输出异常，采用了正确但更保守的 `2 × q4` verifier。该方案避免乱答，但在长上下文下会重复扫描 KV Cache，使 TP4 Decode 相比旧冠军出现约 8%–20% 的回退。

进一步审计确认，问题不是 q8 算法本身，而是 260728 新增的 raw `paged_attention` layout ABI 参数适配错误。TP4 当前 paged KV 几何的正确值为 `layout=0`。修正后可以安全恢复 single raw-q8。

## 本次修复

- SourceFind flash-attn 版本仍固定为：
  `2.8.3+das.opt1.dtk2604.torch290.2607280958.gebb4be`
- TP4 exact DFlash verifier 恢复 single raw-q8；
- 260728 raw `paged_attention` 使用经过重新验证的 `layout=0`；
- raw-q8 仅允许：`q=8 / QH6 / KVH1 / D256 / page64 / BF16`；
- 所有非精确 shape fail-closed；
- BF16 KV Cache 约束不变；
- TP1 / TP2 不受影响。

## 验证结果

isolated raw-q8(layout=0) 在 16K / 64K / 128K / 257.9K 均与 v1.2.0 的 `2 × q4` reference **bitwise equal**。

full-model：

- 普通回答 PASS；
- `tool_choice=auto` PASS；
- canonical 16K / 64K / 128K / 257.9K 输出 SHA 与 v1.2.0 正确版本完全一致；
- 139265 cache-resume gate-v2：2× cold deterministic，cached_tokens=131072，replay=8193，`cached == cold`，PASS；
- 128K 三次 Decode：`87.90 / 87.33 / 87.36 tok/s`；
- 257.9K 两次 Decode：`70.49 / 73.56 tok/s`；
- 257.9K 第二轮 TTFT `132.09s`，与旧冠军约 `132.25s` 基本一致。

代表性对比：

| 上下文 | v1.2.0 q8split | v1.2.1 raw-q8 | 旧冠军参考 |
|---:|---:|---:|---:|
| 16K | 110.94 | **113.55** | 113.10 |
| 64K | ~94.41 median | **98.59 median** | 102.21 |
| 128K | ~78.06 median | **87.36 median** | 88.68 |
| 257.9K | 57.20 | **70.49 / 73.56** | 72.49 |

> 64K 仍观察到单次 Decode runtime 低点，输出 SHA 和 DFlash 接受率未改变，因此不以单次峰值或低点作为正式性能结论。

## 安装方式

GitHub Release 同时附带小补丁包：`qwen38-k100ai-v1.2.1-tp4-hotfix.tar.gz`，SHA256：`43a47bb3a12fa84ba46e49a425ec120222b401fbe4a8e72b06ba324db18385e5`。

本次不要求重新下载完整 Docker 镜像。已有 v1.2.0 RC2 镜像的用户，在仓库中执行：

```bash
cd hotfixes/v1.2.1
bash apply.sh
```

会基于本地 v1.2.0 镜像生成：

```text
qwen38-k100ai-int8:unified-20260827-v1.2.1
```

然后把原启动命令中的镜像名替换为上述 v1.2.1 镜像即可，其余参数完全不变。

详细说明、回滚方法和版本锁见 `hotfixes/v1.2.1/README.md`。

原始验证 JSON：

- `results/tp4_v121_rawq8_layout0_16k_20260827.json`
- `results/tp4_v121_rawq8_layout0_64k128k257k_20260827.json`
- `results/tp4_v121_rawq8_layout0_257900_repeat_20260827.json`
- `results/tp4_v121_cache_resume_139265_20260827.json`

## 回滚

v1.2.0 原镜像不会被修改。直接切回：

```text
qwen38-k100ai-int8:unified-20260826-fa260728-q8split-rc2
```

即可恢复 v1.2.0 的 `2 × q4` verifier。
