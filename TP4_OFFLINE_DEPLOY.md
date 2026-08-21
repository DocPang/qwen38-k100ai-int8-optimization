# 离线算力服务器部署教程

[← 返回项目总览](README.md)

> ⚠️ **免责声明 / 风险提示**
>
> 本项目是社区研究成果，不是海光、SourceFind、Qwen、SGLang 或 DFlash2 官方发行版。部署会直接访问 GPU，并依赖宿主机的 K100AI 驱动、DTK、hyhal、PCIe/P2P 和 `renderD*` 设备映射。
>
> **不要为了照抄本文而直接覆盖驱动、修改 GRUB/IOMMU/ACS、执行未知 `setpci` 命令或在生产服务器上盲目试验。** 配置错误可能导致 GPU 不可用、现有业务中断，极端情况下可能需要重启服务器。先备份现有环境；如果 `hy-smi` 本来就不正常，请先停止部署。

这篇只讲最简单的离线流程：

**联网机器下载 → 搬到算力服务器 → `docker load` → 直接构建薄优化镜像 → `docker run`。**

不要求离线服务器安装 Git、Hugging Face CLI 或 Docker Compose。

---

## 1. 在联网机器准备 4 样东西

### A. SourceFind 基础镜像

```bash
BASE='harbor.sourcefind.cn:5443/dcu/admin/base/custom@sha256:366525b25f452f85eb0ea5813604a64f03c648627bc824bb498b56cf5a325dde'

docker pull "$BASE"
docker tag "$BASE" qwen38-sourcefind-base:20260620
docker save -o sourcefind-sglang0512-k100ai-20260620.tar qwen38-sourcefind-base:20260620
```

约 31.7GB。本机已有对应镜像时 Docker 会复用，不会重新下载全部 layer。

### B. W8A8 target 权重

```bash
hf download Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8 \
  --revision 417ede1e4524c8fdbb586ebdabc9cfc5d0760b3e \
  --local-dir Qwen3.8-27B-SmoothQuant-W8A8-INT8
```

约 30GB。

### C. DFlash2 draft 权重

```bash
hf download z-lab/Qwen3.8-27B-DFlash2 \
  --revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4 \
  --local-dir Qwen3.8-27B-DFlash2
```

约 3.6GB。

### D. 本项目

```bash
git clone https://github.com/DocPang/qwen38-k100ai-int8-optimization.git
```

本项目不到 1MB。

把下面 4 项通过内网、移动硬盘、NAS、`scp` 或 `rsync` 搬到离线服务器：

```text
sourcefind-sglang0512-k100ai-20260620.tar
Qwen3.8-27B-SmoothQuant-W8A8-INT8/
Qwen3.8-27B-DFlash2/
qwen38-k100ai-int8-optimization/
```

---

## 2. 离线服务器先检查环境

先执行：

```bash
uname -a
hy-smi
ls -l /dev/kfd
ls -l /dev/dri/
ls -ld /opt/hyhal
docker version
```

至少确认：

- `hy-smi` 能正常看到 K100AI；
- `/dev/kfd` 存在；
- 你知道准备给 TP4 使用的 4 张卡对应哪 4 个 `renderD*`；
- `/opt/hyhal` 存在；
- Docker 可用。

**不满足就先停，不要为了跑这个项目去自动覆盖驱动。**

建议顺手留一份现场：

```bash
mkdir -p ~/qwen38-host-backup
uname -a > ~/qwen38-host-backup/uname.txt
hy-smi > ~/qwen38-host-backup/hy-smi.txt 2>&1 || true
lspci -tv > ~/qwen38-host-backup/lspci-tv.txt
ls -l /dev/dri/ > ~/qwen38-host-backup/dri.txt
modinfo amdgpu > ~/qwen38-host-backup/amdgpu-modinfo.txt 2>&1 || true
```

---

## 3. 加载官方镜像，再生成我们的优化镜像

假设所有文件放在 `/data/qwen38-offline/`：

```bash
cd /data/qwen38-offline

docker load -i sourcefind-sglang0512-k100ai-20260620.tar
```

确认：

```bash
docker image inspect qwen38-sourcefind-base:20260620 >/dev/null && echo OK
```

然后直接使用仓库内已经验证过的 4 个预编译用户态 HIP 扩展构建薄优化镜像：

```bash
cd /data/qwen38-offline/qwen38-w8a8-k100ai-dflash2-tp4

BASE_IMAGE=qwen38-sourcefind-base:20260620 bash build_image.sh
```

默认 `build_image.sh` 不会启动编译容器，只会校验仓库内 `native_ext/prebuilt/` 的 4 个已验证 `.so`，随后使用本地 SourceFind 底座构建薄优化镜像。

如果你使用的不是本文锁定的 SourceFind/DTK 组合，或者明确希望从源码重编，可选执行：

```bash
BASE_IMAGE=qwen38-sourcefind-base:20260620 REBUILD_NATIVE=1 bash build_image.sh
```

这个备用编译容器无网络、不映射 `/dev/kfd` 或 `renderD*`、不使用 privileged，只读挂载 `/opt/hyhal`。它编译的是用户态 `.so`，不会安装、卸载、重载或替换宿主机 `amdgpu` 驱动。

这个流程使用刚才 `docker load` 的**本地基础镜像**，离线服务器不需要访问 SourceFind Harbor。我们已经在 K100AI 验证机的 Docker 18.09 环境实际完成了“四个源码编译 → Docker 镜像构建 → 无 GPU PyBind 动态加载”验证。

---

## 4. 启动

复制配置模板：

```bash
cd /data/qwen38-offline/qwen38-w8a8-k100ai-dflash2-tp4
cp .env.example .env
```

编辑 `.env`，至少改成你自己的模型路径和 4 个 renderD：

```text
TARGET_MODEL=/data/qwen38-offline/Qwen3.8-27B-SmoothQuant-W8A8-INT8
DRAFT_MODEL=/data/qwen38-offline/Qwen3.8-27B-DFlash2
RENDER0=/dev/dri/renderDxxx
RENDER1=/dev/dri/renderDxxx
RENDER2=/dev/dri/renderDxxx
RENDER3=/dev/dri/renderDxxx
PORT=8068
```

`renderDxxx` 必须换成自己服务器的实际设备号，不能照抄验证机。

然后只需要：

```bash
bash run.sh
```

`run.sh` 会先检查模型目录、`/dev/kfd`、4 个 renderD、`/opt/hyhal` 和本地优化镜像。检查不通过就直接停止；如果同名容器已经存在，也会拒绝自动删除或覆盖。

看日志：

```bash
docker logs -f --tail=100 qwen38-w8a8-k100ai-dflash2-tp4
```

默认端口是 `8068`。长期运行的只有这一个模型服务容器。

---

## 以后更新

通常不需要再搬 31.7GB 基础镜像，也不需要重新搬两份模型。

只把新版这个小仓库拷到离线服务器，再重新执行：

```bash
BASE_IMAGE=qwen38-sourcefind-base:20260620 bash build_image.sh
```

更新现有服务时，请先人工确认当前容器是否可以停止，再手动 `docker stop` / `docker rm` 旧容器，最后重新执行 `bash run.sh`。脚本不会替你自动删除正在使用的容器。

---

## 一句话版

```text
联网机：拉官方镜像 + 两份权重 + GitHub 仓库 → 搬到离线服务器
离线机：docker load → bash build_image.sh → bash run.sh
```

**不需要服务器联网，不需要 Docker Compose，也不需要群友自己手工打 20 层补丁。**
