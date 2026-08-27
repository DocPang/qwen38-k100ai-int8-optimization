# v1.2.2 DFlash2 non-greedy hotfix

这是一个基于 v1.2.1 的小型派生镜像补丁，主要修复 TP4 DFlash2 在 `temperature>0` 或使用模型默认 sampling 参数时可能触发 scheduler hard-fail 的问题。

## 使用

```bash
cd hotfixes/v1.2.2
bash apply.sh
cd ../..
export IMAGE=qwen38-k100ai-int8:unified-20260827-v1.2.2
```

如果本地已经有 v1.2.1 镜像，会直接在其上叠加本补丁；如果只有 v1.2.0 基础镜像，脚本会尝试先调用相邻的 `v1.2.1/apply.sh` 构建 prerequisite。

启动命令无需改变，仍然通过：

```text
PROFILE=tp1|tp2|tp4
PORT=<your port>
MODEL_NAME=<served model name>
```

TP1 / TP2 payload 不变；本补丁只替换 TP4 DFlash2 runtime 中与 selector proposal / verify sampling 有关的三个 Python 文件。

## 支持范围

已验证：

- greedy (`temperature=0`)
- default model sampling
- `temperature>0`
- `top_k`
- `top_p`
- `min_p`
- mixed greedy + sampling batch，最高 c4

暂未声明完整支持：

- deterministic `sampling_seed`
- DFlash-native grammar / JSON schema / `tool_choice=required`
- non-greedy cache-resume 全矩阵

详见仓库根目录 `RELEASE_NOTES_v1.2.2.md`。

## 回滚

直接切回：

```text
qwen38-k100ai-int8:unified-20260827-v1.2.1
```

无需修改模型权重。
