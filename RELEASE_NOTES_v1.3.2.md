# v1.3.2 Release Notes — TP4 DFlash/Mamba Boundary Hotfix

发布日期：2026-09-08

v1.3.2 是一个 **小型运行时补丁版本**。本次**不重新制作或上传 Docker 镜像**，继续使用 v1.3.1 最终镜像：

```text
qwen38-k100ai-int8:v1.3.1
```

Image ID 保持：

```text
sha256:534f9512a3d5217c8f65f11183ed75db19e7aa3df1ecc4da1ba993e98793973d
```

小补丁包：[`Qwen3.8-K100AI-v1.3.2-hotfix.tar.gz`](Qwen3.8-K100AI-v1.3.2-hotfix.tar.gz)（约 13 KiB）

SHA256：

```text
1da6cb64abbef1a637d10d83c139a70bc74b10476d31ee4ab624c322456be896
```

## 1. 修复 TP4 长 Agent / OpenCode cache-resume 卡死

在真实 OpenCode 多轮工具调用中复现到 TP4 服务卡死：

- prefill 正常完成；
- `/v1/chat/completions` streaming 已建立；
- 之后不再进入正常 decode；
- TP0 scheduler 主线程停在 CPU `futex_wait`；
- TP1 / TP2 / TP3 停在 `kfd_wait_on_events`；
- HCU 利用率呈现约 `0% / 6.7% / 6.7% / 6.7%`；
- 300 秒后四 rank scheduler watchdog，服务重启。

本地自然复现请求为：

```text
cached prefix = 49,152
fresh replay  = 1,600
```

之前群友日志中的故障点还包括：

```text
114,688 + 2,496
131,072 + 128
180,224 + 1,344
212,992 + 576
```

这些 cached prefix 都与 16,384-token Mamba tracking boundary 高度相关。

## 2. 根因

当前 v1.3.1 所使用的 DFlash worker 在 verify 后判断是否跨过 Mamba tracking boundary 时，使用：

```python
seq_lens_pre_verify // interval != batch.seq_lens // interval
```

但此时 `batch.seq_lens` 尚未推进本轮 accepted verify tokens。

因此某轮 DFlash 恰好跨过 tracking boundary 时，scheduler bookkeeping 可以继续推进，而对应 boundary 的 Mamba recurrent state 没有 materialize 到 tracking buffer。后续 Radix cache-resume 可能得到：

```text
full-attention KV = 新 checkpoint
Mamba / linear state = 旧 checkpoint
```

最终可导致 TP rank 执行失步和 watchdog deadlock。

该问题与 SGLang upstream issue `sgl-project/sglang#37817` 描述一致。

## 3. 修复方式

v1.3.2 使用真正的 post-verify committed length：

```python
seq_lens_post_verify = seq_lens_pre_verify + commit_lens
```

再计算：

- 是否跨 tracking boundary；
- boundary 位于本轮第几个 accepted step；
- 应保存哪一个 Mamba intermediate state。

补丁只替换：

```text
DFlashWorker._update_target_mamba_state_after_verify
```

不修改：

- attention kernel；
- q8 / q16 / short-Q / long-KV 动态路由；
- sampling；
- DFlash verifier；
- allocator；
- Radix cache policy；
- 模型权重；
- TP4 正式启动参数。

## 4. 与“动态算法切换”的关系

本次问题**不是**项目中按 `q_len / kv_len / batch_size` 做 short-Q、q8、q16、long-KV 路由的动态 attention 选择导致。

直接触发因素是另一类有状态动态行为：

```text
DFlash 每轮动态接受 draft tokens
-> 某轮 accepted range 跨 Mamba tracking boundary
-> 旧实现漏保存 boundary recurrent state
-> 下一轮 cached-resume 恢复到不一致状态
```

因此 v1.3.2 不撤销现有性能动态路由。

## 5. 验证结果

CPU boundary regression：

```text
PASS tp4-mamba-verify-boundary-fix-v1 regression vectors
```

K100AI TP4 正式环境实机复测：

| cached prefix | fresh replay | 结果 | cached E2E |
|---:|---:|---|---:|
| 49,152 | 1,600 | PASS，cold/cached 输出逐 token 一致 | 7.40s |
| 114,688 | 2,496 | PASS，cold/cached 输出逐 token 一致 | 5.63s |
| 131,072 | 128 | PASS，cold/cached 输出逐 token 一致 | 2.79s |

最终运行态：

```text
restart=0
OOM=false
health=200
```

OpenCode 实际链路也重新验证通过。

## 6. 安装方式

不需要新镜像。

继续保留 / 导入 v1.3.1 镜像，然后：

```bash
cd v1.3.2
python3 test_boundary_fix.py

export TARGET_MODEL=/path/to/target
export DRAFT_MODEL=/path/to/draft

HOST_GPU_IDS="0 1 2 3" \
RENDER_DEVICES="/dev/dri/renderD128 /dev/dri/renderD129 /dev/dri/renderD130 /dev/dri/renderD131" \
./scripts/launch.sh tp4
```

v1.3.2 launcher 会自动把 hotfix 只读挂进 v1.3.1 容器。

启动日志必须出现：

```text
[K100 TP4 DFlash Mamba verify-boundary-fix-v1] installed
[K100 TP4 v1.3.2] installed
```

详细说明见 [`v1.3.2/README.md`](v1.3.2/README.md)。

## 7. TP1 / TP2

本次公开补丁只对 TP4 启用。

TP1 / TP2 继续使用 v1.3.1 原有 runtime，不做行为改动。

## 8. 回滚

移除 v1.3.2 hotfix volume mount，重新使用 v1.3.1 TP4 launcher 即可。

由于没有修改 Docker image layer，所以不需要重新导入镜像。
