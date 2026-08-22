# 完整统一运行时镜像

这是已经在 K100AI 上实际验收过的预构建 Docker 运行时镜像。

**重要：单卡 TP1 和四卡 TP4 共用同一个 image，不需要分别下载两个镜像。**
通过 `PROFILE=tp1` / `PROFILE=tp4` 选择运行模式。

> 本文件是 [主 README](../README.md) 中"方式 A / 方式 B"的详细参考。
> 主 README 已经包含完整的下载、校验、导入、启动步骤；这里补充镜像内部结构和验收细节。

---

## 镜像信息

| 项目 | 值 |
|---|---|
| 下载链接 | [夸克网盘](https://pan.quark.cn/s/1e9abeef509a?pwd=duZx)（提取码 `duZx`） |
| 文件名 | `qwen38-k100ai-int8-unified-20260822.docker.tar.zst` |
| 压缩后大小 | `6,091,215,017 bytes`（约 5.67 GiB） |
| Docker 镜像大小 | `31,857,483,311 bytes`（约 31.86 GiB） |
| Docker tag | `qwen38-k100ai-int8:unified-20260822` |
| SHA256 | `e3e2874939b540a935191939fe6309e583a7bf1808f6341f07aba447740d7557` |

---

## 快速开始

```bash
# 1. 校验
echo "e3e2874939b540a935191939fe6309e583a7bf1808f6341f07aba447740d7557  qwen38-k100ai-int8-unified-20260822.docker.tar.zst" | sha256sum -c -
zstd -t qwen38-k100ai-int8-unified-20260822.docker.tar.zst

# 2. 导入
zstd -dc qwen38-k100ai-int8-unified-20260822.docker.tar.zst | docker load
docker images | grep qwen38-k100ai-int8

# 3. 启动（TP4 示例，renderD 编号按实际拓扑替换）
docker run -d \
  --name qwen38-tp4 \
  --network host --ipc host \
  --restart unless-stopped \
  --security-opt label=disable \
  --device /dev/kfd:/dev/kfd \
  --device /dev/dri/renderD128:/dev/dri/renderD128 \
  --device /dev/dri/renderD129:/dev/dri/renderD129 \
  --device /dev/dri/renderD130:/dev/dri/renderD130 \
  --device /dev/dri/renderD131:/dev/dri/renderD131 \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v /path/to/target:/models/target:ro \
  -v /path/to/draft:/models/draft:ro \
  -e PROFILE=tp4 \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -e PORT=8068 \
  qwen38-k100ai-int8:unified-20260822
```

完整启动命令（含 TP1 单卡版本）见 [主 README](../README.md#方式-a统一镜像部署推荐有网环境)。

---

## 镜像内部结构

```
/opt/qwen38-k100ai/
├── entrypoint.sh          # 统一入口，按 PROFILE 分发
├── entrypoint.tp1.sh      # TP1 启动脚本
├── entrypoint.tp4.sh      # TP4 启动脚本
└── model_overrides/       # 模型配置覆盖文件
    ├── preprocessor_config.json
    └── video_preprocessor_config.json

/data/qwen38-27b-k100ai-int8-opt/
├── native_ext/            # 预编译 gfx928 native extensions (.so)
├── scripts/               # 启动脚本
└── cache/                 # Triton/Inductor 缓存

/data/qwen38-dflash2-k100ai/
├── runtime_assets/        # chat template、launch 脚本
├── runtime_patch_*/       # runtime patches
└── work/sourcefind_sglang_overlay*/  # SGLang overlay
```

---

## 模型文件准备

镜像**不包含模型权重**。

原因：避免每次传输重复几十 GB，并方便不同用户替换模型 revision。

运行时需要挂载：

| 挂载路径 | 内容 | 来源 |
|---|---|---|
| `/models/target` | Qwen3.8-27B SmoothQuant W8A8/INT8 target | `Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8` (rev `417ede1`) |
| `/models/draft` | Qwen3.8-27B DFlash2 draft | `z-lab/Qwen3.8-27B-DFlash2` (rev `50307d4`) |

entrypoint 会检查以下文件是否存在，缺失则报错退出：

```
/models/target/config.json
/models/target/tokenizer.json
/models/draft/config.json
/models/draft/model.safetensors
```

---

## Profile 选择

| Profile | GPU | Context | 默认端口 | 用途 |
|---|---:|---:|---:|---|
| `tp1` | 1× K100AI | 147456 | 8090 | 单卡 Agent128K |
| `tp4` | 4× K100AI | 262144 | 8068 | 长上下文主服务（默认） |

统一镜像默认 `PROFILE=tp4`。

使用 TP1 时：

```bash
-e PROFILE=tp1
-e HIP_VISIBLE_DEVICES=0
-e PORT=8090
```

使用 TP4 时：

```bash
-e PROFILE=tp4
-e HIP_VISIBLE_DEVICES=0,1,2,3
-e PORT=8068
```

---

## 宿主机要求

需要已经正常工作的 K100AI 环境：

- `/dev/kfd`
- `/dev/dri/renderD*`
- `/opt/hyhal`
- Docker GPU 设备映射

本镜像**不会安装或修改宿主机驱动**。

render node 编号必须根据自己的机器拓扑填写（`ls /dev/dri/` 查看）。

---

## 离线使用

整个镜像导入过程**不需要访问互联网**：

1. 在有网络的机器上下载 `.tar.zst` 文件；
2. 校验 SHA256 和 zstd 完整性；
3. 通过 rsync / U盘 / scp 传输到目标服务器；
4. 在目标服务器上 `zstd -dc ... | docker load`；
5. 模型权重（target + draft）也需要离线传输到目标服务器。

导入完成后，`docker run` 启动服务不需要任何外网访问。

---

## 验收状态

- TP1：单卡真实启动通过，`health=200`；Chat Completion 验证通过；`restart=0 / OOM=false`。
- TP4：继承正式 longctx-v8 Champion 文件树。
- TP1/TP4 共用 unified image。
- Docker manifest、zstd 校验、Mac 端校验和 SHA256 均通过。

---

## DFlash2 sampling 注意事项

当前生产推荐仍使用确定性采样：

```text
temperature=0
```

统一镜像包含 scheduler protection：错误 sampling 请求不会直接导致 scheduler 崩溃。

该保护用于稳定性，不代表所有非 greedy speculative verification 已实现。
