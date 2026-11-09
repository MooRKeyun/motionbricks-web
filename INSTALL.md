# 从本仓库安装

本仓库只包含**我们自己写的那一层**（Web 前端、生成服务、姿态库、测试、文档）。
官方 [NVlabs/MotionBricks](https://github.com/NVlabs/motionbricks) 以 **git submodule** 原样引用，
指向 commit `54fc563`，不做任何修改。

模型与动作资产（ONNX 约 2.1 GB、BVH 108 MB）**不在本仓库里**：GitHub 单文件上限 100 MB，
免费 LFS 配额 1 GB，放不下。它们由官方仓库的 LFS 提供，克隆后 `git lfs pull` 即可。

## 1. 克隆（必须带 submodule）

```powershell
git clone --recurse-submodules https://github.com/MooRKeyun/motionbricks-web.git
cd motionbricks-web
```

已经克隆过但目录是空的？补一句：

```powershell
git submodule update --init --recursive
```

## 2. 拉取模型与动作资产（LFS）

```powershell
cd motionbricks
git lfs install --local
git lfs pull
cd ..
```

校验：`motionbricks/motionbricks/assets/onnx/soma/mmm_v6_soma_fp32.onnx` 应该是 700 MB 左右。
如果只有 134 字节，说明拿到的是 LFS 指针，重跑 `git lfs pull`。

## 3. 创建 conda 环境

```powershell
conda create -y -n motionbricks -c conda-forge --override-channels python=3.12
conda activate motionbricks
pip install -e motionbricks
pip install fastapi "uvicorn[standard]"
```

> 依赖由 `motionbricks/pyproject.toml` 钉死（warp-lang、tensorrt-cu12 10.8、newton 1.6、usd-core 26.3 等）。
> 需要 NVIDIA GPU；TensorRT 首次运行会构建 plan，约 1 分钟。

## 4. 生成角色网格并启动

```powershell
python -m webapp.glb_export          # SOMA USD -> webapp/static/soma.glb（1.3 MB，不入库）
```

然后双击 `start.bat`，或：

```powershell
python -m uvicorn webapp.server:app --host 127.0.0.1 --port 8000
```

> `start.bat` 里的 `PYTHON=` 写死的是本机环境路径
> （`C:\Users\13119\anaconda3\envs\motionbricks\python.exe`）。
> 换机器时改这一行，或直接用上面的 `python -m uvicorn` 命令启动。

浏览器打开 <http://127.0.0.1:8000>。

## 5. 浏览器端到端测试（可选）

```powershell
pip install playwright
python -m playwright install chromium   # 或者用系统已装的 Edge
python -m tests.e2e
```

`tests/e2e.py` 默认用 `channel="msedge"` 启动浏览器；没有 Edge 时把 `msedge` 换成 `chromium`。

## 目录分工

| 路径 | 来源 | 说明 |
| --- | --- | --- |
| `motionbricks/` | submodule → NVlabs/MotionBricks | 官方库、ONNX 模型、BVH 动作、SOMA USD |
| `webapp/` | 本仓库 | 姿态库、生成服务、GLB 导出、FastAPI、three.js 前端 |
| `tests/` | 本仓库 | Warp 冒烟、动作质量检查、浏览器端到端 |
| `docs/images/` | 本仓库 | 文档截图 |

## 常见问题

**`motionbricks/` 目录是空的**
没带 `--recurse-submodules`，执行 `git submodule update --init --recursive`。

**页面一直停在「构建引擎」**
首次运行在构建 TensorRT plan（约 1 分钟）。看 `tests/server.err.log` 或控制台输出。

**`state: error`，TensorRT 报错**
确认显卡是 NVIDIA 且驱动足够新。本机验证环境：RTX 5060 Ti（sm_120）+ 驱动 610.88 + TensorRT 10.8.0.43。

**中文路径下 LFS / TensorRT 出错**
把仓库放到纯 ASCII 路径（例如 `C:\dev\motionbricks-web`）。本项目开发时就用 ASCII 路径规避了这个问题。
