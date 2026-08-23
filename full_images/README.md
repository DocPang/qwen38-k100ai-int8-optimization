# TP1 / TP2 / TP4 统一镜像入口

本目录保存 **v1.1.1 部署版所使用的 TP1 / TP2 / TP4 统一运行入口**；推理 runtime payload 与 v1.1.0 已验收版本相同：

- `entrypoint.sh`：读取 `PROFILE=tp1|tp2|tp4` 并分发；
- `entrypoint_tp1.sh`：TP1；
- `entrypoint_tp2.sh`：TP2；
- `entrypoint_tp4.sh`：TP4。

这些脚本也会被打进 `qwen38-k100ai-patchset.tar.gz`，因此通过根目录 `Dockerfile` 构建出来的镜像与这里保持同一套 Profile 逻辑。

## Profile

| PROFILE | GPU | Context | 默认端口 | Profile |
|---|---:|---:|---:|---|
| `tp1` | 1 | 262144 | 8090 | TP1 |
| `tp2` | 2 | 262144 | 8062 | TP2 |
| `tp4` | 4 | 262144 | 8068 | TP4 |

## 当前推荐

v1.1.1 的 B/C 路线推荐从固定 SourceFind 基础镜像构建：

```bash
cp .env.example .env
# 设置 PROFILE=tp1 / tp2 / tp4 和模型、renderD 路径
bash build_image.sh
bash run.sh
```

构建完成后的镜像本身就是统一镜像，一个 image 可以通过 `PROFILE` 启动三种并行度。

## 完整镜像包

当前统一镜像同时支持 `PROFILE=tp1|tp2|tp4`：

- 文件：`qwen38-k100ai-int8-unified-20260823.docker.tar.zst`
- Docker tag：`qwen38-k100ai-int8:unified-20260823`
- 压缩大小：**5.65 GiB（6,065,184,632 bytes）**
- Docker 镜像大小：**约 31.70 GB（Docker image `.Size`）**
- SHA256：`6d14588722b0fea0ab66a53e2810385d1f9999a9cd78c8e1d2e6640c744f2b14`
- 下载：**[夸克网盘：full_images](https://pan.quark.cn/s/e7626123faa0?pwd=M8Fr)** · 提取码：`M8Fr`

镜像不包含模型权重。只需额外挂载：

1. Qwen3.8-27B SmoothQuant W8A8/INT8 Target；
2. Qwen3.8-27B DFlash2 Draft。

模型下载来源**二选一，不要重复下载**：

- **夸克整合包**：https://pan.quark.cn/s/eb79a87216ba?pwd=Rcxc，提取码 `Rcxc`；一个包已经同时包含 Target + Draft。
- **HuggingFace**：按主 README 中固定 revision 下载两份模型；国内网络可设置 `HF_ENDPOINT=https://hf-mirror.com`、`HF_HUB_DISABLE_XET=1`。

**不需要下载 BF16/FP16 Base 权重**，processor metadata 已内置。完整命令见 [主 README](../README.md)。

## 离线使用

v1.1.1 可以完全离线构建：

1. 有网机器准备仓库、Target/Draft 权重、固定 SourceFind base image；
2. `docker save` 基础镜像；
3. 拷贝到离线 K100AI 服务器；
4. `docker load` 基础镜像；
5. `.env` 中设置 `BASE_IMAGE=<本地 tag>`；
6. `bash build_image.sh`；
7. `PROFILE=tp1|tp2|tp4` 启动。

详细说明见 [主 README](../README.md)。
