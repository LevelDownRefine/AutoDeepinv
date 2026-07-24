# AGENTS.md — AutoDeepInv（自动化模型调优框架）

本项目是一个**自动化模型调优框架**：同一任务下遍历不同配置（数据集处理方式 / 网络 / 超参数——后两者本身也是超参数），产出同一套指标并横向对比。训练与评估流程相对固定，变的是「配置」，不变的是「流程」。

> **历史注记（可行性探针，已结案）**：最初用 deepinv 的 DRUNet（加载 KAIR `drunet_color.pth`）在 Set5 彩色去噪端到端验证「deepinv 是否可用」。结果 PASS（见文末结果表），**可行性已确认**。DRUNet 本身此后**不再是项目重点**，仅作为第一个 `DenoisingTask.evaluate` 的参考实现（`run_set5_deepinv.py`）。本文件不再以它为叙述中心。

---

## 回归 / 验收方法论（核心，优先于一切实现细节）

**有无回归，靠端到端对比「之前跑出来的数据」，不靠数据集约定。**

- **不**依赖「匹配 KAIR 的加噪 realization / HWC 布局 / 数值约定」来判定没回归——那是可行性探针阶段的临时手段，不是本框架的验收机制。
- 正道：把一次运行（配置 + 端到端产出的指标 / 产物）存为 **基线快照（golden master）**；后续改动后跑**同样配置端到端**，把新产出与快照**逐指标 diff**，超容差即视为回归。
- **可复现是前提，不是附加项**：同一 `RunConfig` 必须能稳定复现上次的端到端产出（固定随机种子 / 确定性数据构建 / 无隐藏全局状态）。复现性不成立，端到端对比就无意义。
- 落到架构：`GridSearch.run` 产出 `RunResult` 列表 → 基线 = 上一次 `RunResult` 集合的序列化；回归 = 同 `SweepSpec` 重跑后 diff。容差与「哪些指标参与对比」由**自动化层**持有，领域 `Task` 不碰。

> 可行性探针里的 `COMPARISON`/`build_benchmark`（把 scores 对照 KAIR 参考值判 PASS）**仅用于当初证明 deepinv≈KAIR**，不是回归机制；后续应从框架中剥离，或仅作一次性对照保留。

---

## 运行与环境

- 用 `run.bat` 启动：路径通过环境变量注入（`SHIM_ROOT` / `DRUNET_WEIGHTS` / `TESTSET_DIR`，均无 hard code 默认值），不写死在 `.py` 里。
- `deepinv` 经**最小 shim**（`SHIM_ROOT`）解析，复用 KAIR venv 的 cu128 torch；**勿 `pip install` 完整 deepinv**。
- 依赖走 **uv**：`pyproject.toml` + `.python-version`(3.12) + `uv.lock`（CUDA-128 torch 照搬 KAIR）；CI 用 `uv sync --frozen` + `uv run pytest tests/ -q`。
- `utils_image.py`（KAIR 图像工具）已 vendored 进 `utils/`，目前复用 `imread_uint` / `uint2single` / `single2tensor4`。

---

## 可行性探针结果（历史，仅证明 deepinv 可用，非回归门禁）

判据（当时）：`ΔdB ≥ -0.05` 且 `ΔSSIM ≥ -0.001` → PASS。

| sigma | PSNR | SSIM | ΔPSNR | ΔSSIM | status |
|------:|-----:|-----:|------:|------:|--------|
| 15 | 34.9284 | 0.9269 | +0.0301 | +0.0006 | PASS |
| 25 | 32.7537 | 0.8965 | +0.0290 | +0.0006 | PASS |
| 50 | 29.8981 | 0.8451 | +0.0315 | +0.0000 | PASS |

---

## 项目愿景与设计原则

### 1. 目标：自动化调优，而非一次性跑通

框架的终态是：**同一个任务**下，自动遍历不同配置并横向对比指标。

- **同一任务**可对应多种**数据集处理方式**（加噪约定、预处理、transform 等）；
- 可对应多种**网络** backbone；
- 可对应多种**超参数**。
- **网络与数据集本身也是超参数**：它们和 lr / batch size 一样，是「配置空间」的可枚举维度，不该写死在代码里。
- 每次运行都产出**同一套指标**（PSNR / SSIM 等），并支持在配置间**对比**。
- **训练与评估流程相对固定**：变的是「配置」，不变的是「流程」。把固定流程抽象成稳定的 pipeline，配置只是 pipeline 的输入。

### 2. 技术选型：先 deepinv，后 KAIR

- **优先看 deepinv 是否支持**：[deepinv](https://github.com/deepinv/deepinv) 是首选基础设施（model / data / metrics / training 有统一抽象），能覆盖就直接用。
- **不支持时参考 `../KAIR`**：KAIR 是后备参考实现（尤其图像工具、benchmark 数值约定）。本项目已 vendored `utils_image.py` 进 `utils/`（见上文）。
- 只引入需要的，不无脑照搬；能复用 deepinv 抽象就复用。

### 3. 稳定性 > 一切（最高优先级）

高于便利、高于开发速度。

- **参数严格显式传入**：从调用方显式传，不从全局 / 隐式状态偷读（路径通过 `run.bat` 注入 env 且无默认值，属显式）。
- **不设默认值，除非有明确理由**：每个默认值的存在都要能回答「为什么这里可以有个默认」。没理由就别默认——逼调用方想清楚，避免「静默的错误配置」。
- **该 assert 的 assert**：对不变量、前提条件、配置合法性，在入口处用 `assert` 严格校验，尽早 fail loud。`assert` 是契约，不是调试摆设。

### 4. 灵活扩展：用 kwargs，但严格检查 kwargs

自动化任务的配置空间开放且会增长，所以**大量输入 / 输出走 `kwargs`**——否则每加一个维度（新网络 / 新数据集 / 新超参）就要改函数签名，框架会僵硬到无法支撑自动化。

但 `kwargs` ≠ 「随便塞、随便漏」：

- **该存在的必须存在**：进入函数后对每个必填 kwarg 做存在性检查（`assert key in kwargs` 或显式 `kwargs.pop("required")`），缺了立刻报错，别让 `KeyError` 在深层随机冒出。
- **不该存在的不能存在**：函数消费完它认识的 kwargs 后，必须校验**无残留未知 key**（如 `assert not kwargs` 或 `assert set(kwargs) <= allowed`）。这能捕捉「调用方拼错参数名」或「配置 schema 漂移」这类沉默错误。
- 二者缺一不可：**kwargs 提供签名级灵活性，strict-check 提供契约级安全性**。只灵活不检查会积累配置 bug，只检查不灵活就退回硬编码。

### 5. 配置 schema 草案 v2（见 `config_schema.py`）

已重构成三层（stdlib-only，可独立运行验证）。终态架构：

```
SweepSpec ──► GridSearch(spec) ──► Task(领域单元: DenoisingTask)
             展开 RunConfig         ├─ train(run, **kwargs)
             编排 train/eval        └─ evaluate(run, **kwargs)
```

- `SweepSpec`：自动化输入 = `models`(多模型) + `hyperparameters`(min/max/step) + `metrics`(同一套对比指标) + 固定 `pipeline`。（旧版 `Task` 已改名，避免与领域 `Task` 撞名。）
- `GridSearch`：**最顶层（网格搜索）**。持有 `SweepSpec`，`build_runs()` 笛卡尔积展开成 `RunConfig` 列表，`run(task, registries)` 逐配置调用 `task.train/evaluate` 并产出 `RunResult` 列表（含同一套指标），`summarize()` 渲染横向对比表。用户确认：网格即最终策略，不抽 `SearchStrategy` 基类（其他策略不够极限，特殊问题再说）。
- `Task`（领域单元，如 `DenoisingTask`）：核心两子方法 `train` / `evaluate`，只吃给定超参做训练/评估，**不处理自动化**；`run()` 默认组合 train+eval（benchmark-only 子类可让 train 退化为加载权重）。`GridSearch` 以「组合」方式持有 Task（has-a），不继承。
- `Model`（deepinv）/`DatasetProcessing`（参考 KAIR）/`Hyperparameter`(带 scope) 不变；`RunConfig`（展开后的单次运行）/ `RunResult`（配置 + 指标）。

**核心关系**：`Model.dataset_processing` 是到 `DatasetProcessing` 的 FK，many-to-one——同一份数据处理被多模型公平对比。

**纪律落地**：无默认值（空固定参数 `default_factory=dict` 除外）；`__post_init__` 严格 assert；dict 装载（`_strict_consume`）与 `Task` 的 `**kwargs` 均严格检查（必填存在、无残留未知 key）。

> 草案骨架，尚未接入 deepinv/KAIR 真实训练；`DenoisingTask.train/evaluate` 为占位（`raise NotImplementedError`），真实实现接 deepinv+KAIR。回存放 `RunResult` 集合、做端到端 diff 的回归机制尚未落地（见上文「回归 / 验收方法论」）。

### 6. vendored KAIR 数据层（见 `AutoDeepInv/data/`）

从 KAIR `data/` 搬入本项目源码包（`AutoDeepInv/data/`）并**按本项目纪律严格审查改写**。共 1 基类 + 6 去噪相关数据集：

- `base_dataset.py` — 共享基类（见下）。
- `dataset_dncnn.py` / `dataset_fdncnn.py` / `dataset_ffdnet.py` — **纯去噪**：固定/可变 sigma 的 AWGN，分别输出 `(L,H)` / `(L,H,M噪声水平图)` / `(L,H,C噪声水平标量)`。
- `dataset_dnpatch.py` — DnCNN 思路的 **patch 预抽取**版（构造时把 H patch 抽进 buffer 再退化）。
- `dataset_plain.py` / `dataset_plainpatch.py` — **通用 image-to-image 配对**数据集（KAIR 也用于去噪训练；load 双方 L/H）。

测试在 `tests/data/`：
- `conftest.py` 提供 `write_rgb_png` / `make_image_dir` 两个合成图夹具（无 scipy、不提交二进制）。
- `test_base_dataset.py` 覆盖基类契约；`test_dataset_{dncnn,fdncnn,ffdnet,dnpatch,plain,plainpatch}.py` 覆盖各子类（KAIR 原测试，保留）。
- 运行：`uv run pytest tests/data/ -v -p no:cacheprovider`（与 CI 同 venv）。当前 `tests/data/` 共 **50 passed**（含 6 个「缺必填超参即 fail loud」回归测试）。
- 导入与项目一致（靠 `pythonpath=["AutoDeepInv"]`）：源码 `import utils.utils_image as util`，测试 `from data.dataset_x import DatasetX`。

**严格审查要点（本项目纪律 §3）**：

1. **`n_channels` 必填**：基类 `assert "n_channels" in opt`，不再有 `n_channels_default=3` 静默默认。
2. **`paths_H` 改为「可选 + `len` 失败 loud」**（重要修正）：原 KAIR 基类 `paths_H = opt.get('paths_H') or get_image_paths(opt.get('dataroot_H'))` 在两者皆缺时静默置 `None`，会造出「空数据集却不报错」。初版我们把它改成「二选一否则 `assert`」，但 **TDD 发现这过严**——`DnCNN/FDnCNN/FFDNet` 在「合成/测试模式」下本就不需要磁盘路径（测试直接构造合成图调 `_make_sample`），KAIR 原测试正是这么用。修正为：`paths_H` 可选（都没有则为 `None`），但 `__len__` 在 `paths_H is None` 时 **`assert` 失败 loud**（绝不让训练在 0 张图上静默跑）；磁盘型子类（`DnPatch`/`Plain`/`PlainPatch`）在 `__init__` 里再 `assert paths_H` 必填。意图「绝不静默空训练」不变，只是把检查点从构造挪到长度查询。
3. **子类超参「无默认 + `_demand` 失败 loud」**（2026-07-25 二次收紧）：初版我们保留 `opt.get('X', D)` 并写理由注释，但那仍属「静默默认」——既违反本项目 §3「不设默认值除非有明确理由」，更关键的冲突是：sigma/patch_size/num_patches 正是 `GridSearch` 要扫的超参，把它们写死成类内默认值，等于把「该由上层注入的超参」藏进了类内部，形成埋雷（实例化漏传就静默拿到 64/25/40/3000，与 sweep 意图悄悄脱节）。现改为：所有超参 **必填**，经基类实例方法 `BaseDataset._demand(opt, key)` 取值（缺失即抛出**自动取自 `type(self).__name__` 的 owner-tagged** `AssertionError`，绝不静默回退；调用处只写 `self._demand(opt, 'H_size')`，不用传类名）。**唯一保留的默认是「派生型默认」**：`DatasetDnCNN`/`DatasetDnPatch` 的 `sigma_test = opt.get('sigma_test', self.sigma)`——测试噪声等级默认等于训练 sigma（单 sigma 训练/测试时），属推导而非惯例值。而 `FDnCNN`/`FFDNet` 的 `sigma` 是 `[min,max]` 训练区间、`sigma_test` 是独立标量，**无法从 sigma 推导**，故 `sigma_test` 也改为必填（去掉原魔数 25 默认）。`tests/data/` 新增 6 个 `test_*_init_missing_required_raises`，锁死「漏传超参即 fail loud」，防止日后重构偷偷加回默认。
4. **去掉噪声 `print`**：`dataset_dncnn` 与 `dataset_plain/plainpatch` 构造 / `update_data` 里的 `print(...)` 删除——库代码在自动化循环里不应有 stdout 噪声。
5. **结构不变量保留并强化**：`paths_H`/`paths_L` 配对长度一致、`paths_L` 缺则报错等断言保留；基类 `_load_img_*` 仍含非 `None` + 越界断言。

**TDD 验证（非空过）**：
- 基类：严格测试先跑 KAIR 原版 → `test_base_requires_n_channels` 失败（证原版不合规），换严格版 → 通过；`paths_H` 策略经「二选一 assert → TDD 暴露过严 → 改可选+len 守卫」迭代，最终 `test_base_paths_H_optional_but_len_fails_loud` 证明合成模式可构造、但 `len()` 在无路径时 fail loud。
- 6 子类：先原样拷入（此时仍继承我们初版「paths_H 必填」基类）→ 14 个测试**失败**（dncnn/fdncnn/ffdnet 在合成模式构造无路径被 assert 拦下）；把基类 `paths_H` 放宽为可选+len 守卫后 → **44 passed**。

**有意留白**：`opt`（含子类 opt）**不做**未知 key 穷举——基类面向子类扩展，子类加自己的 key（scale/sigma 等）。采取「基类消费 key 严格必填 + 子类 key 放行」。若日后要收紧再议。
