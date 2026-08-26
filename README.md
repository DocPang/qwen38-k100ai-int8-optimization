# Qwen3.8-27B W8A8 on K100AI

面向 **Hygon K100AI / gfx928** 的 Qwen3.8-27B W8A8 SGLang 优化方案。

当前统一支持：

- **TP1：1 张 K100AI**
- **TP2：2 张 K100AI**
- **TP4：4 张 K100AI**

本次发布只提供 **完整 Docker 镜像**。不需要自己编译 SGLang、flash-attn 或本项目补丁。

> ⚠️ 本项目是社区研究成果，不是海光、SourceFind、Qwen、SGLang 或 DFlash2 官方发行版。请确保宿主机 K100AI 驱动、`/dev/kfd`、`/opt/hyhal` 和 Docker 本身工作正常。

## v1.2.0 更新

- TP1 / TP2 / TP4 合并为 **同一个完整镜像**，通过 `PROFILE=tp1|tp2|tp4` 切换。
- 更新 SourceFind 新版 `flash-attn 2.8.3+...2607280958.gebb4be`。
- 修复新版 flash-attn 下 TP2 / TP4 可能出现的 **乱答、输出异常、工具调用异常**。
- TP2 继续优化短、中、长上下文，修复部分长度下速度突然下降的问题。
- TP4 继续强化长上下文、缓存复用和 Agent 场景，并完善 **2 / 3 / 4 并发**。
- 正常 OpenAI Compatible API、Reasoning、`tool_choice=auto` / 默认工具调用已验证可用。
- 启动时可直接修改 **端口** 和 **对外模型名称**。

## 性能参考

以下为前一轮正式十档结果中的代表点，本次主要更新正确性、兼容性和统一镜像，数值用于选 Profile：

| Profile | GPU | 64K Decode | 128K Decode | 257.9K Decode | 建议用途 |
|---|---:|---:|---:|---:|---|
| **TP1** | 1 | ~31.7 tok/s | ~33.9 tok/s | ~24.3 tok/s | GPU 最省、单用户 |
| **TP2** | 2 | ~73.3 tok/s | ~50.0 tok/s | ~53.1 tok/s | 性能 / GPU 成本平衡 |
| **TP4** | 4 | ~102.2 tok/s | ~88.7 tok/s | ~72.5 tok/s | 长上下文、Agent、并发 |

TP4 并发总吞吐粗略参考（不同请求并发，总吞吐，不是单请求速度）：

| 并发数 | 总吞吐 |
|---:|---:|
| 1 | ~95–97 tok/s |
| 2 | ~115–117 tok/s |
| 3 | ~141–152 tok/s |
| 4 | ~175–183 tok/s |

> 并发结果会受 prompt 长度、缓存命中、输出长度等影响，仅作为实际部署容量参考。

完整性能表见 [PERFORMANCE.md](PERFORMANCE.md)。

---

## 1. 下载完整镜像

夸克网盘：

**[Qwen3.8-K100AI-Unified-20260826-RC2](https://pan.quark.cn/s/156dd54a0861?pwd=2T8B)**

提取码：`2T8B`

镜像文件：

```text
Qwen3.8-K100AI-Unified-20260826-RC2.tar.zst
```

SHA256：

```text
215cfb3f15254b8c8cb790a091f21c27827d77abdcd89c7e86ebbc58a4fe6770
```

压缩包约 **6.6 GB**，Docker 镜像约 **33.9 GB**。

校验并导入：

```bash
IMAGE_ARCHIVE=Qwen3.8-K100AI-Unified-20260826-RC2.tar.zst

echo "215cfb3f15254b8c8cb790a091f21c27827d77abdcd89c7e86ebbc58a4fe6770  $IMAGE_ARCHIVE" | sha256sum -c -
zstd -t "$IMAGE_ARCHIVE"
zstd -dc "$IMAGE_ARCHIVE" | docker load
```

导入后的镜像：

```text
qwen38-k100ai-int8:unified-20260826-fa260728-q8split-rc2
```

---

## 2. 准备模型

本次镜像更新 **不包含模型权重**。如果你已经部署过上一版，Target / Draft 权重可以直接继续使用，不需要重新下载。

需要两份模型：

- Target：`Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8`
- Draft：`z-lab/Qwen3.8-27B-DFlash2`

设置实际路径：

```bash
export TARGET_MODEL=/path/to/Qwen3.8-27B-SmoothQuant-W8A8-INT8
export DRAFT_MODEL=/path/to/Qwen3.8-27B-DFlash2
```

不需要额外准备 BF16 / FP16 Base 权重。

---

## 3. 启动

先确认准备使用的 GPU 对应哪个 `renderD*`：

```bash
hy-smi
ls -l /dev/dri/renderD*
```

下面使用 `renderD128` 开始举例，**请替换成你机器上的实际设备号**。

统一镜像：

```bash
export IMAGE=qwen38-k100ai-int8:unified-20260826-fa260728-q8split-rc2
```

### TP1：1 张卡

```bash
export R0=/dev/dri/renderD128

docker run -d \
  --name qwen38-tp1 \
  --restart unless-stopped \
  --network host --ipc host \
  --security-opt label=disable \
  --device /dev/kfd:/dev/kfd \
  --device "$R0:$R0" \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$TARGET_MODEL:/models/target:ro" \
  -v "$DRAFT_MODEL:/models/draft:ro" \
  -e PROFILE=tp1 \
  -e HIP_VISIBLE_DEVICES=0 \
  -e PORT=8090 \
  -e MODEL_NAME=Qwen3.8-27B-W8A8-TP1 \
  "$IMAGE"
```

### TP2：2 张卡

```bash
export R0=/dev/dri/renderD128
export R1=/dev/dri/renderD129

docker run -d \
  --name qwen38-tp2 \
  --restart unless-stopped \
  --network host --ipc host \
  --security-opt label=disable \
  --device /dev/kfd:/dev/kfd \
  --device "$R0:$R0" \
  --device "$R1:$R1" \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$TARGET_MODEL:/models/target:ro" \
  -v "$DRAFT_MODEL:/models/draft:ro" \
  -e PROFILE=tp2 \
  -e HIP_VISIBLE_DEVICES=0,1 \
  -e PORT=8062 \
  -e MODEL_NAME=Qwen3.8-27B-W8A8-TP2 \
  "$IMAGE"
```

### TP4：4 张卡

```bash
export R0=/dev/dri/renderD128
export R1=/dev/dri/renderD129
export R2=/dev/dri/renderD130
export R3=/dev/dri/renderD131

docker run -d \
  --name qwen38-tp4 \
  --restart unless-stopped \
  --network host --ipc host \
  --security-opt label=disable \
  --device /dev/kfd:/dev/kfd \
  --device "$R0:$R0" \
  --device "$R1:$R1" \
  --device "$R2:$R2" \
  --device "$R3:$R3" \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$TARGET_MODEL:/models/target:ro" \
  -v "$DRAFT_MODEL:/models/draft:ro" \
  -e PROFILE=tp4 \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -e PORT=8068 \
  -e MODEL_NAME=Qwen3.8-27B-W8A8-TP4 \
  "$IMAGE"
```

---

## 4. 修改端口和模型名称

不需要修改镜像内部文件，直接改启动命令中的环境变量：

```bash
-e PORT=9000
-e MODEL_NAME=my-qwen38
```

例如：

```text
PORT=9000
MODEL_NAME=my-qwen38
```

API 地址就是：

```text
http://服务器IP:9000/v1
```

`/v1/models` 返回的模型 ID 就会是：

```text
my-qwen38
```

也兼容使用：

```bash
-e SERVED_MODEL_NAME=my-qwen38
```

---

## 5. 验证和管理

查看模型：

```bash
curl http://127.0.0.1:8068/v1/models
```

查看日志：

```bash
docker logs -f qwen38-tp4
```

停止 / 启动 / 重启：

```bash
docker stop qwen38-tp4
docker start qwen38-tp4
docker restart qwen38-tp4
```

---

## 已知限制

- 当前统一使用 **BF16 KV Cache**；不要自行切换 FP8 KV Cache。
- 正常 `tool_choice=auto` / 默认工具调用已验证可用。
- 当前 SGLang DFlash 不支持 `tool_choice=required` 对应的强制 grammar-constrained decoding，该模式会返回 400。
- TP1 / TP2 的极长上下文缓存复用仍在继续完善；TP4 当前完成度更高，但仍会继续扩大随机缓存与并发回归测试。

技术细节与完整数据： [TP1](TP1.md) · [TP2](TP2.md) · [TP4](TP4.md) · [PERFORMANCE](PERFORMANCE.md)

## License

项目自身代码见 [LICENSE](LICENSE)，第三方来源与许可说明见 [NOTICE.md](NOTICE.md)。
