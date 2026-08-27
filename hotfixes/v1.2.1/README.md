# v1.2.1 TP4 raw-q8 hotfix

这是基于 **v1.2.0 unified RC2** 的小版本热修复，只修改 TP4 的 DFlash2 target verifier。TP1 / TP2 不变，模型权重不变，不需要重新下载完整 6.6GB 镜像。

## 修复内容

v1.2.0 为兼容 SourceFind `flash-attn 2.8.3+das.opt1.dtk2604.torch290.2607280958.gebb4be`，TP4 的 q=8 verifier 临时采用安全的 `2 × q4` 路径。该路径正确，但长 KV 下需要重复扫描 KV Cache，导致 64K / 128K / 257.9K Decode 有约 8%–20% 的性能损失。

进一步审计确认，260728 的 raw `paged_attention` 新增 layout ABI 后，v1.2.0 研发阶段曾把 TP4 raw-q8 的 layout 参数解释错误。当前生产 paged KV 几何对应的正确参数为 `layout=0`。修正后，single raw-q8 可以重新启用。

本 hotfix：

- 只在精确 TP4 DFlash verifier 几何 `q=8 / QH=6 / KVH=1 / D=256 / page=64 / BF16` 上恢复 single raw-q8；
- 对 260728 版本严格 version-lock；
- 其他 shape 全部 fail-closed，继承原 v1.2.0 路径；
- BF16 KV Cache 约束不变；
- 不启用通用 q>=5 vendor paged 路径；
- TP1 / TP2 payload 与启动方式不变。

## 已完成验证

### isolated correctness

raw-q8(layout=0) 与 v1.2.0 正确的 `2 × q4` reference：

| KV 长度 | 结果 |
|---:|---|
| 16K | bitwise equal |
| 64K | bitwise equal |
| 128K | bitwise equal |
| 257.9K | bitwise equal |

Q / K / V / block table 指纹在调用前后保持一致。

### full-model correctness

- 普通回答：PASS；
- `tool_choice=auto`：PASS；
- 16K / 64K / 128K / 257.9K canonical output SHA 与 v1.2.0 q8split 正确版本完全一致；
- 139265 cache-resume gate-v2：2× cold deterministic，cached_tokens=131072，replay=8193，**cached output == cold output，PASS**；
- DFlash 接受率保持一致，没有通过改变候选语义换取速度。

### TP4 Decode 参考

同一 canonical corpus、output=256：

| 上下文 | v1.2.0 q8split | v1.2.1 raw-q8 | 旧冠军参考 |
|---:|---:|---:|---:|
| 16K | 110.94 tok/s | **113.55 tok/s** | 113.10 tok/s |
| 64K | ~94.41 tok/s median | **98.59 tok/s median** | 102.21 tok/s |
| 128K | ~78.06 tok/s median | **87.36 tok/s median** | 88.68 tok/s |
| 257.9K | 57.20 tok/s | **70.49 / 73.56 tok/s** | 72.49 tok/s |

64K 仍观察到一次 78.41 tok/s 的 Decode runtime 低点；同一 prompt 的接受率和输出 SHA 没有变化，因此不把单次峰值/低点当作正式性能结论。

原始验证 JSON 已保存在仓库 `results/`：

- `results/tp4_v121_rawq8_layout0_16k_20260827.json`
- `results/tp4_v121_rawq8_layout0_64k128k257k_20260827.json`
- `results/tp4_v121_rawq8_layout0_257900_repeat_20260827.json`
- `results/tp4_v121_cache_resume_139265_20260827.json`

## 应用补丁

前提：本地已经导入 v1.2.0 完整镜像：

```text
qwen38-k100ai-int8:unified-20260826-fa260728-q8split-rc2
```

在仓库根目录执行：

```bash
cd hotfixes/v1.2.1
bash apply.sh
```

默认会生成一个新的本地镜像：

```text
qwen38-k100ai-int8:unified-20260827-v1.2.1
```

原 v1.2.0 镜像不会被修改。

随后把原来的启动命令中的镜像名替换为：

```text
qwen38-k100ai-int8:unified-20260827-v1.2.1
```

其余 `PROFILE=tp1|tp2|tp4`、权重目录、`PORT`、`MODEL_NAME`、render device 等参数全部照旧。

如果你的 v1.2.0 镜像使用了其他本地 tag，可以指定：

```bash
BASE_IMAGE=你的_v1.2.0镜像名 \
OUT_IMAGE=qwen38-k100ai-int8:unified-20260827-v1.2.1 \
bash apply.sh
```

## 回滚

本 hotfix 采用派生镜像，不会覆盖 v1.2.0。需要回滚时，把启动命令中的镜像名切回：

```text
qwen38-k100ai-int8:unified-20260826-fa260728-q8split-rc2
```

即可恢复 v1.2.0 的安全 `2 × q4` verifier。

## 适用范围

本补丁**只适用于 v1.2.0 RC2 内置的 SourceFind flash-attn 260728 精确版本**。补丁内部有版本锁；其他 flash-attn 版本会直接报错退出，不会静默套用未经验证的 raw-q8 ABI。
