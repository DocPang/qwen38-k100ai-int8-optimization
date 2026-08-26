# v1.1.1 — Deployment / Reproducibility Fix

发布日期：2026-08-23

这是 **v1.1.0 的部署与复现修复版**。推理 runtime、模型权重、性能结果以及完整镜像 `qwen38-k100ai-int8:unified-20260823` 均未改变，不需要重新下载或重新打包完整镜像。

## 修复内容

- 重写主 README 的 A / B / C 从零部署流程，明确三种部署方式三选一；
- 明确模型权重只有一套，夸克整合包 / HuggingFace 二选一；
- 写明夸克权重包真实解压目录与 Target / Draft 的准确挂载路径；
- 明确完整镜像内部启动链：`/opt/qwen38-k100ai/start.sh` → `entrypoint.tp1.sh / tp2.sh / tp4.sh`；
- 明确 B / C 的宿主机启动入口是仓库根目录 `run.sh`，不再错误复用 A 的完整镜像 tag；
- 补全 B 路线固定 SourceFind 基础镜像的 `docker pull`；
- 修正离线教程中的旧仓库目录、旧容器名和“4 个扩展”旧描述；
- 修复 C 路线从 `.env` 解析本地 `BASE_IMAGE` 后向 `build_native.sh` 传递的问题；
- 将原生 `sglang serve` 参数移到高级排障说明，不再作为首次部署必做步骤；
- TP1 / TP2 / TP4 专页统一回链主 README，避免维护多套互相漂移的部署教程。

## 不变内容

- 完整镜像：`qwen38-k100ai-int8-unified-20260823.docker.tar.zst`
- Docker tag：`qwen38-k100ai-int8:unified-20260823`
- 完整镜像 SHA256：`6d14588722b0fea0ab66a53e2810385d1f9999a9cd78c8e1d2e6640c744f2b14`
- W8A8 Target revision：`417ede1e4524c8fdbb586ebdabc9cfc5d0760b3e`
- DFlash2 Draft revision：`50307d4c4cde6860d4eee73e2547cd786fe8e8a4`
- TP1 / TP2 / TP4 runtime 参数、十档结果与质量验收结果均不变。
