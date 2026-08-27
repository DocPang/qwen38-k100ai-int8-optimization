# v1.2.2 Release Notes

日期：2026-08-27

## 定位

v1.2.2 是当前推荐正式版本。对外优先提供 **完整 Docker 镜像**；它在 v1.2.1 基础上加入 TP4 DFlash2 non-greedy 功能与稳定性修复，不修改模型权重，不升级 SGLang / Torch / flash-attn，也不改变 TP1 / TP2 payload。

本次主要解决一个用户可直接触发的严重问题：旧 K100AI DFlash2 backport 仅允许 greedy verification，当客户端使用默认采样参数（模型 generation config 为 `temperature=1.0 / top_k=20 / top_p=0.95`）或显式设置 `temperature>0` 时，会进入 non-greedy 路径并触发 hard raise，进而导致 TP4 scheduler 退出。

## 修复内容

v1.2.2 将 DFlash2 selector sampling 与 verifier rejection sampling 在 K100AI/gfx928 上补齐：

1. selector 按请求真实 `temperature` 进行 proposal sampling；
2. 保存 selector sparse proposal distribution `q_rows`；
3. target verify 按正常采样语义构造 `p`：`temperature -> top-k -> top-p -> min-p -> renormalize`；
4. draft token 以 `min(1, p/q)` 进行 rejection sampling；
5. reject 时从 `relu(p-q)` 归一化分布中采样 bonus token；
6. TP>1 下 proposal/acceptance RNG 由 rank0 统一产生并广播，最终 accept length / bonus 同样由 rank0 作为 authority，避免不同 TP rank 分叉；
7. `temperature=0` / greedy 请求仍走原有已验证路径，不引入新的 sampling collective。

这不是把 non-greedy 请求偷偷退回 greedy，也不是简单删除异常；采样语义已经做独立 Monte Carlo 验证。

## 实机验证

### Full-model / API

K100AI TP4，SourceFind SGLang 0.5.12，DFlash2 block=8：

- greedy candidate vs v1.2.1 production：完整 message SHA 一致；
- `temperature=1.0 / top_k=20 / top_p=0.95`：PASS；
- 完全不传 sampling 参数、继承模型 generation config：PASS；
- mixed c4：greedy + `temperature=0.7/top_p=0.8/top_k=20` + `temperature=1/top_p=.95/top_k=20` + default sampling：4/4 PASS；
- TP4 verify consistency audit：76 verify groups × 4 ranks，0 divergence；
- 12K prompt non-greedy：PASS；
- scheduler crash / SIGQUIT / VMFault：0。

### Rejection sampling correctness

独立 50,000 样本 Monte Carlo：

- 普通 p/q rejection：empirical TV distance = `0.00392`，最大单 token 偏差 `0.00262`；
- `temperature=0.7 / top_k=20 / top_p=0.8 / min_p=0.05`：TV distance = `0.00644`，最大单 token 偏差 `0.00295`；
- 过滤后理论 support = 9，实测 support = 9。

该误差量级符合有限 Monte Carlo 采样噪声，没有发现系统性分布偏置。

### 正式生产部署 gate

GPU4-7 正式服务使用：

- image：`qwen38-k100ai-int8:unified-20260827-v1.2.2`
- container：`qwen38-k100ai-tp4-production-v1.2.2`
- served model：`Qwen3.8-27B-W8A8-DFlash2-TP4`
- port：`8068`
- `MEM_FRACTION_STATIC=0.95`

warm gate：

- greedy：200 OK，约 0.95 s；
- default sampling：200 OK，约 6.0 s；
- explicit `temperature=1/top_k=20/top_p=.95`：200 OK，约 2.97 s；
- mixed c4：4/4 200 OK，约 2.55-2.77 s；
- container restart=0；
- error log：0 Traceback / RuntimeError / SIGQUIT / VMFault / Scheduler hit exception。

## 已知边界

以下内容 **不要理解为 v1.2.2 已完整支持**：

1. **deterministic `sampling_seed`**：当前 DFlash2 proposal/accept RNG 尚未完整接入每请求 deterministic seed，重复使用同一 seed 不保证输出完全一致；
2. **grammar / JSON schema / `tool_choice=required` 的 DFlash-native speculative path**：仍属于后续功能项；默认工具调用 / `tool_choice=auto` 不受本次改动影响；
3. **non-greedy cache-resume 全矩阵**：重复请求和长 prompt 已验证稳定，但本轮测试返回的 `cached_tokens=0`，因此没有把 non-greedy cache-resume 宣称为完整 release authority；
4. 首次冷启动后第一条请求可能触发额外 JIT/编译，延迟显著高于 warm steady-state；这属于当前 SourceFind/K100AI 栈的 cold-start 行为，性能比较应使用 warm authority。

## 安装

### 推荐：完整镜像

夸克网盘：[Qwen3.8-K100AI-Unified-v1.2.2](https://pan.quark.cn/s/653e165c2fc7?pwd=DBia)，提取码：`DBia`。

文件：`Qwen3.8-K100AI-Unified-v1.2.2.tar.zst`

SHA256：`91818fcc5ae0fc1cfcec6b6b9cc2950ee991f293fd16221ccd80e91fa069850d`

导入后使用：

```bash
export IMAGE=qwen38-k100ai-int8:unified-20260827-v1.2.2
```

之后继续使用仓库 README 中的 TP1 / TP2 / TP4 `docker run` 命令即可。

### 可选：旧用户增量升级

已经部署旧版本、且不想重新导入完整镜像的用户，可以使用 `hotfixes/v1.2.2/apply.sh`。这只是可选升级路径，不是新用户必需步骤。

启动方式、`PROFILE=tp1|tp2|tp4`、`PORT`、`MODEL_NAME` 与之前保持一致。

## 回滚

v1.2.2 的直接父镜像为：

```text
qwen38-k100ai-int8:unified-20260827-v1.2.1
```

如需回滚，只需停止 v1.2.2 容器并重新使用 v1.2.1 镜像启动；模型权重不需要重新下载。
