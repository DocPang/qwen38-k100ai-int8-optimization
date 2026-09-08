# Qwen3.8-K100AI v1.3.2 — TP4 DFlash/Mamba Boundary Hotfix

发布日期：2026-09-08

v1.3.2 是基于 **v1.3.1 最终镜像** 的小型运行时补丁，**不重新发布 Docker 镜像**。

本次只修复 TP4 + DFlash2 + Mamba/Radix cache-resume 在 tracking boundary 上可能出现的 scheduler deadlock。TP1 / TP2 沿用 v1.3.1 已验证运行时，不做行为改动。

## 需要什么

继续使用 v1.3.1 镜像：

```text
qwen38-k100ai-int8:v1.3.1
```

验证过的 Image ID：

```text
sha256:534f9512a3d5217c8f65f11183ed75db19e7aa3df1ecc4da1ba993e98793973d
```

不需要重新下载模型权重，也不需要重新下载 6.3 GiB 镜像。

## 修复的问题

在 DFlash verify 中，如果本轮 accepted tokens 跨过 Mamba tracking boundary，旧代码使用：

```python
seq_lens_pre_verify // interval != batch.seq_lens // interval
```

判断是否跨界。但此时 `batch.seq_lens` 尚未推进本轮真正接受的 `commit_lens`，因此边界可能被漏掉。

v1.3.2 改为按真正的 post-verify 长度判断：

```python
seq_lens_post_verify = seq_lens_pre_verify + commit_lens
```

然后保存对应 accepted step 的 Mamba recurrent checkpoint。

这与 SGLang upstream issue `sgl-project/sglang#37817` 描述的问题一致：full-attention KV / scheduler bookkeeping 已推进，但 Mamba tracking buffer 可能仍是旧状态，后续 Radix cached-resume 会恢复到不一致状态。

在 K100AI TP4 上，实际故障表现为：

```text
prefill 正常完成
-> HTTP streaming 已返回 200
-> 没有进入正常 Decode
-> TP0 卡在 CPU futex
-> TP1/TP2/TP3 卡在 kfd_wait_on_events
-> HCU 利用率约 0% / 6.7% / 6.7% / 6.7%
-> 300 秒 scheduler watchdog
-> 容器重启
```

OpenCode / coding Agent 的连续 tool-call 会话特别容易触发，因为它会频繁形成大 prefix reuse 和固定 16K Mamba checkpoint resume。

## 安装 / 启动

先停止旧 TP4 容器，然后直接使用 v1.3.2 launcher：

```bash
cd v1.3.2

python3 test_boundary_fix.py

export TARGET_MODEL=/path/to/Qwen3.8-27B-SmoothQuant-W8A8-INT8
export DRAFT_MODEL=/path/to/Qwen3.8-27B-DFlash2

HOST_GPU_IDS="0 1 2 3" \
RENDER_DEVICES="/dev/dri/renderD128 /dev/dri/renderD129 /dev/dri/renderD130 /dev/dri/renderD131" \
./scripts/launch.sh tp4
```

启动脚本仍然验证并启动：

```text
qwen38-k100ai-int8:v1.3.1
```

但会额外只读挂载三个很小的 hotfix 目录：

```text
runtime_patch_dflash_tp4_v30_base_v131
runtime_patch_dflash_tp4_mamba_verify_boundary_fix_v1
runtime_patch_dflash_tp4_v30_mamba_boundaryfix_v1
```

其中：

- `runtime_patch_dflash_tp4_v30_base_v131` 是 v1.3.1 原始 TP4 v30 `sitecustomize.py` 的冻结副本；
- boundary fix 只替换 `DFlashWorker._update_target_mamba_state_after_verify`；
- wrapper 先加载完整 v1.3.1 v30，再加载 hotfix；
- attention / sampling / allocator / Radix / verifier / short-Q / q8 / q16 / long-KV 路由全部继承 v1.3.1。

## 启动后确认补丁已加载

日志中应看到：

```text
[K100 TP4 DFlash Mamba verify-boundary-fix-v1] installed: post_verify=pre_verify+commit_lens; all non-Mamba/DFlash paths unchanged
[K100 TP4 v1.3.2] installed: v1.3.1 v30 base + DFlash/Mamba verify-boundary correctness hotfix
```

如果没有这两个 marker，不要把服务当作 v1.3.2。

## 实机验证

修复在 K100AI TP4 正式服务上完成后，复测了本次自然复现点和群友原始故障形状：

| cached prefix | fresh replay | 结果 | cached E2E |
|---:|---:|---|---:|
| 49,152 | 1,600 | PASS，cold/cached 输出逐 token 一致 | 7.40s |
| 114,688 | 2,496 | PASS，cold/cached 输出逐 token 一致 | 5.63s |
| 131,072 | 128 | PASS，cold/cached 输出逐 token 一致 | 2.79s |

验证期间：

```text
restart=0
OOM=false
```

原来 `49,152 + 1,600` 会在 prefill 后进入 TP rank 失步并最终触发 300 秒 watchdog；修复后同一 cache-resume 几何正常进入 decode 并结束。

## TP1 / TP2

v1.3.2 的公开 hotfix 当前只对 TP4 启用。

```bash
./scripts/launch.sh tp1
./scripts/launch.sh tp2
```

仍使用 v1.3.1 镜像里的原有运行时，不注入本次 TP4 boundary patch。

## 自定义 launcher 的用户

如果不用本仓库的 `scripts/launch_tp4.sh`，需要给 v1.3.1 容器增加以下三个只读挂载：

```bash
-v "$PWD/hotfix/runtime_patch_dflash_tp4_v30_base_v131:/data/qwen38-dflash2-k100ai/runtime_patch_dflash_tp4_v30_base_v131:ro" \
-v "$PWD/hotfix/runtime_patch_dflash_tp4_mamba_verify_boundary_fix_v1:/data/qwen38-dflash2-k100ai/runtime_patch_dflash_tp4_mamba_verify_boundary_fix_v1:ro" \
-v "$PWD/hotfix/runtime_patch_dflash_tp4_v30_mamba_boundaryfix_v1:/data/qwen38-dflash2-k100ai/runtime_patch_dflash_tp4_v30:ro"
```

第三个挂载会覆盖镜像中原 `runtime_patch_dflash_tp4_v30` 目录；wrapper 会从第一个挂载重新加载冻结的 v1.3.1 base，因此不会递归，也不会丢失原 v30 组合。

## 回滚

删除 v1.3.2 的三个 hotfix volume mount，重新使用 v1.3.1 launcher 即可：

```bash
../v1.3.1/scripts/launch.sh tp4
```

补丁不会修改镜像层，因此回滚不需要重新 `docker load`。

## 文件

- `hotfix/runtime_patch_dflash_tp4_v30_base_v131/sitecustomize.py`
- `hotfix/runtime_patch_dflash_tp4_mamba_verify_boundary_fix_v1/sitecustomize.py`
- `hotfix/runtime_patch_dflash_tp4_v30_mamba_boundaryfix_v1/sitecustomize.py`
- `scripts/launch.sh`
- `scripts/launch_tp4.sh`
- `test_boundary_fix.py`
- `SHA256SUMS`
- `RELEASE_MANIFEST.json`

完整项目说明仍见仓库根目录 `README.md`。
