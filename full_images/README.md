# 完整统一运行时镜像

这是已经在 K100AI 上实际验收过的预构建 Docker 运行时镜像，单卡 TP1 与四卡 TP4 共用同一个 image tag，通过 `PROFILE` 切换。

## 下载

夸克网盘：

- 链接：https://pan.quark.cn/s/1e9abeef509a?pwd=duZx
- 提取码：`duZx`
- 文件：`qwen38-k100ai-int8-unified-20260822.docker.tar.zst`
- 压缩后大小：`6,091,215,017 bytes`（约 5.67 GiB）
- Docker image size：`31,857,483,311 bytes`
- Docker tag：`qwen38-k100ai-int8:unified-20260822`
- SHA256：`e3e2874939b540a935191939fe6309e583a7bf1808f6341f07aba447740d7557`

下载后建议先校验：

```bash
sha256sum qwen38-k100ai-int8-unified-20260822.docker.tar.zst
zstd -t qwen38-k100ai-int8-unified-20260822.docker.tar.zst
```

导入：

```bash
zstd -dc qwen38-k100ai-int8-unified-20260822.docker.tar.zst | docker load
```

## Profile

| Profile | GPU | 默认端口 | Context | 说明 |
|---|---:|---:|---:|---|
| `PROFILE=tp1` | 1× K100AI | 8090 | 147456 | TP1 Agent128K |
| `PROFILE=tp4` | 4× K100AI | 8068 | 262144 | TP4 longctx-v8 Champion，镜像默认 |

模型权重没有重复塞进 Docker image。运行时仍需挂载：

- `/models/target` → Qwen3.8-27B SmoothQuant W8A8/INT8 target
- `/models/draft` → Qwen3.8-27B DFlash2 draft

宿主机仍需提供已经正常工作的 `/dev/kfd`、对应的 `/dev/dri/renderD*` 和 `/opt/hyhal`。render node 编号按机器真实拓扑填写，不要照抄别人的设备号。

TP1 启动时设置 `PROFILE=tp1` 并只映射 1 张 K100AI；TP4 设置 `PROFILE=tp4` 并映射 4 张 K100AI。

## 验收状态

- TP1：空闲 K100AI 真冷启动通过，`health=200`；真实 Chat Completion 精确返回 `UNIFIED_TP1_OK`；`restart=0 / OOM=false`。
- TP4：统一镜像直接继承正式验收的 longctx-v8 Champion 文件树；TP4 entrypoint 与 Champion 镜像 SHA256 完全一致。
- TP4 DFlash2 worker 与当前 hardened 运行版 SHA256 完全一致。
- TP1 generic DFlash2 worker 同样为 hardened 版本。
- 统一镜像内同时包含 TP1 generic overlay 与 TP4 overlay。
- Docker save manifest、服务端 zstd 校验、Mac 端 zstd 校验与双端 SHA256 均已通过。

## DFlash2 sampling 注意事项

当前 K100AI DFlash2 后端的正式生产路径仍建议使用确定性采样，例如 `temperature=0`。统一镜像已包含 worker 级可用性保护：如果误发非 greedy sampling，请求不会再因为旧的 fatal gate 直接杀死 scheduler，而会退化到 greedy argmax verification。

这项保护的目标是避免服务崩溃，不代表当前后端已经实现完整的非 greedy speculative verification 语义。

## 这些脚本是什么

本目录中的：

- `entrypoint.sh`
- `entrypoint_tp1.sh`
- `entrypoint_tp4.sh`

对应统一镜像中的 profile dispatch 与两套已验收启动参数，便于审计。根目录的 `Dockerfile / build_image.sh / run.sh` 仍表示源码构建/部署路线；预构建完整镜像是额外的快速分发路线，两者不要混为一谈。
