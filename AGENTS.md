# AGENTS.md — drunet_deepinv_eval

端到端验证：**deepinv 的 DRUNet（加载 KAIR `drunet_color.pth` 权重）在 Set5 彩色去噪上的精度不劣于 KAIR 原版 benchmark**。
整套走 deepinv（model / data / metrics），仅加噪保留 KAIR 的数值约定。

`utils_image.py`（KAIR 的图像工具百宝箱，~1000 行，含 cv2/matplotlib 依赖）已 **vendored 进本项目**（整文件复制，非 git 子模块）。当前复用了其中三个函数：`imread_uint`（cv2 读图、BGR→RGB，返回 uint8 HWC）、`uint2single`（`/255` 归一化，已验证在 Set5 上 PIL→np.asarray→/255 **逐位相同**——Set5 为 PNG 无损，两解码器像素一致）与 `single2tensor4`（HWC numpy → `(1,3,H,W)` 张量）。数据布局因此采用 **HWC**（与 KAIR 原版 benchmark 同口径）；transform 内 `imread_uint(path)` + `uint2single` 直接产出 HWC float32，**不再有自定义读取 helper**。其余函数（`calculate_psnr/ssim`、`imresize` 等）暂未使用。导入它会连带 import cv2/matplotlib——`run_set5_deepinv.py` 顶部已 `from utils_image import uint2single, single2tensor4, imread_uint`。

## 运行

用 `run.bat` 启动（它会设置好所有路径环境变量，再调用 KAIR 的 venv 执行脚本）：

```bash
run.bat
```

所有路径**写在 `run.bat` 里**，不在 `run_set5_deepinv.py`。`run.bat` 通过 `%~dp0` 相对定位到 KAIR 根目录与 shim，再注入 `SHIM_ROOT` / `DRUNET_WEIGHTS` / `TESTSET_DIR` 三个环境变量；`run_set5_deepinv.py` 只 `os.environ[...]` 读这三个量，没有 hard code 默认值。路径若搬迁，只改 `run.bat` 顶部那段即可。

`deepinv` 通过**最小 shim**（`SHIM_ROOT`）解析，复用 KAIR venv 里的 cu128 torch。**不要 pip install 完整 deepinv**——shim 只拷了需要的源文件。

## 核心心智模型：评估是目的，对比是验证手段

| 部分 | 角色 | 是否含 ref |
|------|------|-----------|
| `eval_workflow()` | **目的**：加载模型 + 构建 per-sigma 数据集 + `evaluate()` + `print_scores()` | 否（零 ref） |
| `COMPARISON` (`run_comparison` + `build_benchmark`) | **验证手段**（非目的）：唯一知道参考值/容差的地方，消费 `eval_workflow` 的 scores | 是（仅此处） |
| `evaluate(model, dataset, metrics, sigma, device)` | 纯打分函数：`dataset` 与 `metrics` 外部注入，不构造数据、不知 ref | 否 |

`eval_workflow()` 单独调用即可完成「跑 deepinv DRUNet 出分数」这件事；删掉 `main` 里那三行对比调用，它照样能独立运行。

## 关键约定

- **加噪约定（KAIR benchmark）**：`clean/255 + np.random.seed(0) + AWGN(sigma/255)`，不裁剪。
  ⚠️ `np.random.normal(size=shape)` 的 realization **依赖数组内存布局**——CHW 与 HWC 抽出的噪声图不同。本项目**刻意采用 HWC**（与 KAIR 原版 benchmark 同口径），故噪声 realization 与 KAIR 一致、结果可直接对照；若改回 CHW 会换一份噪声场、数值随之变化（但仍 PASS）。
- **ref 隔离**：所有基准参考值 & 容差只在 `build_benchmark()`。评估路径（`eval_workflow` / `scoring_metrics` / `evaluate` / `print_scores`）不得引用 ref。
- **指标可扩展**：新增指标只改两处——`scoring_metrics()`（追加 `Metric`）和 `build_benchmark()`（追加对应 ref/tol）；`evaluate` 一行不用动。
- **数据集外部注入 `evaluate`**：`evaluate` 不构造 dataset；`eval_workflow` 内用 deepinv `ImageFolder`（`loader=lambda p: p` 把路径喂入 transform）+ `_make_pair_transform(sigma)`，transform 经 vendored `imread_uint` + `uint2single` 产出 `(clean, noisy)` numpy HWC float32 `[0,1]`，再经 `single2tensor4` 转 `(1,3,H,W)` 张量喂模型（无自定义读取 helper）。

## 判据

`ΔdB ≥ -0.05` 且 `ΔSSIM ≥ -0.001` → PASS。

## 当前结果（Set5，相对 KAIR drunet_color 基准）

| sigma | PSNR | SSIM | ΔPSNR | ΔSSIM | status |
|------:|-----:|-----:|------:|------:|--------|
| 15 | 34.9284 | 0.9269 | +0.0301 | +0.0006 | PASS |
| 25 | 32.7537 | 0.8965 | +0.0290 | +0.0006 | PASS |
| 50 | 29.8981 | 0.8451 | +0.0315 | +0.0000 | PASS |

## 环境

- KAIR venv（uv 管理，勿直接 pip）；已装 `torchmetrics`。
- `deepinv` 来自 minimal shim，非 pip 安装。
- 路径不写在 `.py` 里：由 `run.bat` 设置 `SHIM_ROOT` / `DRUNET_WEIGHTS` / `TESTSET_DIR` 三个环境变量（无默认值）。
