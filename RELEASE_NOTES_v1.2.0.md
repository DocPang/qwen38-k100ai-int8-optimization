# v1.2.0 — Unified TP1 / TP2 / TP4 image

本次更新将 TP1 / TP2 / TP4 收敛为 **同一个完整 Docker 镜像**，并更新 SourceFind 新版 flash-attn。重点修复新版环境下可能出现的乱答、输出异常和工具调用异常，同时继续完善 TP2 / TP4 长上下文、缓存复用和 TP4 并发能力。

## 主要更新

- TP1 / TP2 / TP4 统一为一个镜像，通过 `PROFILE=tp1|tp2|tp4` 切换。
- 更新 `flash-attn 2.8.3+das.opt1.dtk2604.torch290.2607280958.gebb4be`。
- 修复新版 flash-attn 下 TP2 / TP4 输出异常；正常回答与自动工具调用已重新验证。
- TP2 继续修复部分上下文长度下速度突然下降的问题。
- TP4 继续优化长上下文、缓存复用和 Agent 场景。
- TP4 已针对 **2 / 3 / 4 并发**进行适配，最高按 4 并发优化。
- 启动时可直接修改 `PORT` 和 `MODEL_NAME`，无需修改镜像内部文件。
- 本次只提供 **完整 Docker 镜像**，不再同时维护多套部署流程。

## TP4 并发粗略参考

不同请求并发时的总吞吐约为：

| 并发 | 总吞吐 |
|---:|---:|
| 1 | ~95–97 tok/s |
| 2 | ~115–117 tok/s |
| 3 | ~141–152 tok/s |
| 4 | ~175–183 tok/s |

> 为不同 prompt 的 aggregate throughput 粗略参考；实际结果会受上下文长度、缓存命中和输出长度影响。

## 完整镜像

夸克网盘：

https://pan.quark.cn/s/156dd54a0861?pwd=2T8B

提取码：`2T8B`

文件：

```text
Qwen3.8-K100AI-Unified-20260826-RC2.tar.zst
```

SHA256：

```text
215cfb3f15254b8c8cb790a091f21c27827d77abdcd89c7e86ebbc58a4fe6770
```

## 模型权重（二选一下载）

如果已经有上一版的 Target / Draft 权重，可以直接继续使用，不需要重新下载。

推荐使用之前整理好的夸克完整权重包，一包同时包含 Target + Draft：

- `Qwen3.8-K100AI-Weights-20260823`
- https://pan.quark.cn/s/eb79a87216ba?pwd=Rcxc
- 提取码：`Rcxc`
- 文件：`qwen38-k100ai-w8a8-dflash2-weights-20260823.tar.zst`
- SHA256：`aa33b9d1ed1e31b1f5c3c6989a302299ecb957ff3f2768f233fdaab17f0073f5`

也可以按 README 中固定 revision 从 HuggingFace 分别下载 Target + Draft。**两种方式二选一，不需要 BF16 / FP16 Base 权重。**

完整导入、真实权重目录、TP1 / TP2 / TP4 启动，以及端口 / 模型名称修改方式见仓库 README。

## 历史性能数据

上一版本的正式十档测速数据、JSON 结果和测速图片继续保留在仓库中，没有删除。详细结果见 `PERFORMANCE.md`、`results/` 和 `assets/`。

## 已知限制

- 保持 BF16 KV Cache，不建议自行切换 FP8 KV Cache。
- 正常 `tool_choice=auto` / 默认工具调用可用。
- 当前 DFlash 不支持 `tool_choice=required` 对应的强制 grammar-constrained decoding。
- TP1 / TP2 极长上下文缓存复用仍在继续完善；TP4 会继续扩大随机缓存和并发回归测试。
