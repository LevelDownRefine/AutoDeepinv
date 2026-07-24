"""自动化调优框架 —— 配置 schema 骨架（草案 v2）。

架构（详见 AGENTS.md《项目愿景与设计原则》）：

    SweepSpec                    自动化输入：多模型 + 被扫超参(min/max/step) + 同一套 metrics
        │
    GridSearch(spec)             最顶层（网格搜索）：展开 RunConfig + 编排 task.train/evaluate + 横向对比
        │  build_runs() 产出 RunConfig(model + 其数据处理 + 解析超参)
        ▼
    Task (领域单元, 如 DenoisingTask)
        ├─ train(run, **kwargs)      训练/调优（给定超参，优化权重）
        └─ evaluate(run, **kwargs)   评估（产出同一套指标）

设计要点：
- **模型确定其数据集处理方式**：`Model.dataset_processing` 是到 `DatasetProcessing`
  注册表的 FK；一个 DatasetProcessing 可被多个 Model 引用（many-to-one），
  使「同一份数据处理」能被多模型公平对比。
- 数据集处理方式参考 KAIR（`DatasetProcessing.impl`）；模型用 deepinv（`Model.impl`）。
- **Task 自身不处理自动化**：GridSearch 以「组合」方式持有 Task（has-a）并驱动它，不继承。
- 用户确认：网格即最终策略，不抽 `SearchStrategy` 基类（其他策略不够极限，特殊问题再说）。

纪律（AGENTS.md 第 3、4 点）：无默认值（空固定参数 field(default_factory=dict) 除外）、
`__post_init__` 严格 assert、dict 装载与 Task 的 `**kwargs` 均严格检查
（必填必须存在、消费完不得残留未知 key）。

本文件为草案骨架，仅依赖标准库，未接入 deepinv/KAIR 真实训练。
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class HyperparamScope(str, Enum):
    DATASET = "dataset"
    MODEL = "model"
    TRAINING = "training"


@dataclass(frozen=True)
class SweepRange:
    min: float
    max: float
    step: float

    def __post_init__(self) -> None:
        assert self.step > 0, f"SweepRange.step 必须 > 0，得到 {self.step}"
        assert self.min <= self.max, f"SweepRange.min({self.min}) 必须 <= max({self.max})"

    def discrete(self) -> list[float]:
        """闭区间 [min, max] 按 step 离散；浮点容差 1e-9（语义同 arange，max 须可整除）。"""
        n = int(round((self.max - self.min) / self.step))
        values: list[float] = []
        for i in range(n + 1):
            v = self.min + i * self.step
            if abs(v - self.max) <= 1e-9:
                v = self.max
            values.append(round(v, 9))
        return values


@dataclass(frozen=True)
class Hyperparameter:
    name: str
    scope: HyperparamScope
    sweep: SweepRange
    # scope==MODEL 时必须指定 target（模型名）；其余不应指定
    target: str | None = None

    def __post_init__(self) -> None:
        assert self.name, "Hyperparameter.name 不能为空"
        if self.scope is HyperparamScope.MODEL:
            assert self.target, "scope=MODEL 的 hyperparameter 必须指定 target（模型名）"
        else:
            assert self.target is None, "scope≠MODEL 时不应指定 target"


@dataclass(frozen=True)
class DatasetProcessing:
    name: str
    impl: str  # 参考 KAIR 的处理实现标识，如 "awgn_color"
    fixed_params: Mapping[str, Any] = field(default_factory=dict)  # 非扫固定值（允许为空）

    def __post_init__(self) -> None:
        assert self.name and self.impl, "DatasetProcessing.name/impl 不能为空"


@dataclass(frozen=True)
class Model:
    name: str
    impl: str  # deepinv 模型标识，如 "drunet" / "scunet"
    dataset_processing: str  # FK -> DatasetProcessing.name（模型确定数据处理）
    fixed_params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert self.name and self.impl and self.dataset_processing, (
            "Model.name/impl/dataset_processing 不能为空"
        )


@dataclass(frozen=True)
class SweepSpec:
    """自动化输入：一个任务 = 多模型 + 被扫超参 + 同一套对比指标 + 固定 pipeline。"""

    name: str
    pipeline: str
    models: Sequence[str]
    hyperparameters: Sequence[Hyperparameter]
    metrics: Sequence[str]

    def __post_init__(self) -> None:
        assert self.name and self.pipeline
        assert self.models, "SweepSpec 必须给出 >=1 个模型"
        assert self.metrics, "SweepSpec 必须给出 >=1 个对比指标"

    @staticmethod
    def from_dict(d: dict) -> "SweepSpec":
        d = _strict_consume(
            d, required=["name", "pipeline", "models", "hyperparameters", "metrics"], optional=[]
        )
        return SweepSpec(
            name=d["name"],
            pipeline=d["pipeline"],
            models=list(d["models"]),
            hyperparameters=[_build_hyperparameter(h) for h in d["hyperparameters"]],
            metrics=list(d["metrics"]),
        )


@dataclass(frozen=True)
class RunConfig:
    """一次展开后的具体运行：模型 + 其数据处理 + 解析后的超参。"""

    task: str
    model: str
    dataset_processing: str
    resolved: Mapping[str, Any]


@dataclass(frozen=True)
class RunResult:
    """一次运行的结果：配置 + 同一套指标。"""

    task: str
    model: str
    dataset_processing: str
    resolved: Mapping[str, Any]
    metrics: Mapping[str, float]


class Task(ABC):
    """领域单元（如 DenoisingTask）。核心两个子方法：train / evaluate。

    Task 只吃给定超参做训练/评估，不处理自动化调优；自动化层（GridSearch）
    以组合方式持有 Task 并驱动它。run() 默认组合 train+eval；
    benchmark-only 的子类可让 train 退化为「加载权重」而不训练。
    """

    @abstractmethod
    def train(self, run: RunConfig, **kwargs: Any) -> Any:
        """给定 RunConfig（含模型/数据处理/超参），训练或加载模型，返回模型对象。"""
        ...

    @abstractmethod
    def evaluate(self, run: RunConfig, **kwargs: Any) -> Mapping[str, float]:
        """给定 RunConfig，产出同一套指标（PSNR/SSIM 等）。"""
        ...

    def run(self, run: RunConfig, **kwargs: Any) -> Mapping[str, float]:
        self.train(run, **kwargs)
        return self.evaluate(run, **kwargs)


class DenoisingTask(Task):
    """去噪任务的领域单元（schema 占位，未接线）。

    真实实现：train 接 deepinv（训练或加载 drunet_color.pth 等权重），
    evaluate 接 KAIR 约定（加噪/HWC/PSNR/SSIM）。本骨架只定义形状。
    """

    def train(self, run: RunConfig, **kwargs: Any) -> Any:
        raise NotImplementedError("DenoisingTask.train：接 deepinv（训练或加载权重）")

    def evaluate(self, run: RunConfig, **kwargs: Any) -> Mapping[str, float]:
        raise NotImplementedError("DenoisingTask.evaluate：接 KAIR 约定（PSNR/SSIM）")


@dataclass
class GridSearch:
    """最顶层（网格搜索）：把 SweepSpec 展开成 RunConfig，并编排 Task 训练/评估 + 横向对比。

    用户确认：网格即最终策略，不抽 SearchStrategy 基类（其他策略不够极限，特殊问题再说）。
    """

    spec: SweepSpec

    def build_runs(
        self,
        model_registry: Mapping[str, Model],
        ds_registry: Mapping[str, DatasetProcessing],
    ) -> list[RunConfig]:
        hp_values = [(hp, hp.sweep.discrete()) for hp in self.spec.hyperparameters]
        runs: list[RunConfig] = []
        for model_name in self.spec.models:
            assert model_name in model_registry, f"SweepSpec 引用了未知模型 {model_name!r}"
            model = model_registry[model_name]
            assert model.dataset_processing in ds_registry, (
                f"模型 {model_name!r} 指向未知 dataset_processing {model.dataset_processing!r}"
            )
            ds_name = model.dataset_processing
            # 该模型实际参与的超参维度
            active: list[tuple[Hyperparameter, list[float]]] = []
            for hp, vals in hp_values:
                if hp.scope is HyperparamScope.MODEL:
                    if hp.target == model_name:
                        active.append((hp, vals))
                    # 否则该超参不适用此模型，跳过
                else:
                    active.append((hp, vals))
            if not active:
                runs.append(RunConfig(self.spec.name, model_name, ds_name, {}))
                continue
            keys = [hp.name for hp, _ in active]
            value_lists = [vals for _, vals in active]
            for combo in itertools.product(*value_lists):
                resolved = dict(zip(keys, combo))
                runs.append(RunConfig(self.spec.name, model_name, ds_name, resolved))
        return runs

    def run(
        self,
        task: Task,
        model_registry: Mapping[str, Model],
        ds_registry: Mapping[str, DatasetProcessing],
    ) -> list[RunResult]:
        results: list[RunResult] = []
        for run in self.build_runs(model_registry, ds_registry):
            metrics = task.run(run)
            results.append(
                RunResult(run.task, run.model, run.dataset_processing, run.resolved, metrics)
            )
        return results


def summarize(results: Sequence[RunResult]) -> str:
    """把 RunResult 列表渲染成横向对比表。"""
    if not results:
        return "(no results)"
    metric_keys = list(results[0].metrics.keys())
    resolved_keys = _resolved_keys(results)
    header = ["model", "dataset_processing"] + [f"hp:{k}" for k in resolved_keys] + metric_keys
    lines = ["\t".join(header)]
    for r in results:
        row = (
            [r.model, r.dataset_processing]
            + [str(r.resolved.get(k, "")) for k in resolved_keys]
            + [f"{r.metrics[k]:.4f}" for k in metric_keys]
        )
        lines.append("\t".join(row))
    return "\n".join(lines)


def _resolved_keys(results: Sequence[RunResult]) -> list[str]:
    keys: list[str] = []
    for r in results:
        for k in r.resolved:
            if k not in keys:
                keys.append(k)
    return keys


# ---------------------------------------------------------------------------
# 严格 dict 装载 / kwargs 检查（AGENTS.md 第 4 点：必填存在、无残留未知 key）
# ---------------------------------------------------------------------------

def _strict_consume(d: dict, required: Sequence[str], optional: Sequence[str]) -> dict:
    if not isinstance(d, dict):
        raise AssertionError(f"期望 dict，得到 {type(d).__name__}")
    out: dict = {}
    for k in required:
        assert k in d, f"缺少必填字段 {k!r}"
        out[k] = d.pop(k)
    for k in optional:
        if k in d:
            out[k] = d.pop(k)
    assert not d, f"存在未预期的字段：{sorted(d)!r}"
    return out


def _build_hyperparameter(d: dict) -> Hyperparameter:
    d = _strict_consume(d, required=["name", "scope", "sweep"], optional=["target"])
    sweep = d["sweep"]
    assert isinstance(sweep, dict), f"hyperparameter {d['name']!r} 的 sweep 必须是 dict"
    sweep = _strict_consume(sweep, required=["min", "max", "step"], optional=[])
    return Hyperparameter(
        name=d["name"],
        scope=HyperparamScope(d["scope"]),
        sweep=SweepRange(**sweep),
        target=d.get("target"),
    )


if __name__ == "__main__":
    ds_registry = {"kair_color_awgn": DatasetProcessing("kair_color_awgn", "awgn_color")}
    model_registry = {
        "drunet": Model("drunet", "drunet", "kair_color_awgn"),
        "scunet": Model("scunet", "scunet", "kair_color_awgn"),  # many-to-one
    }
    spec = SweepSpec(
        name="set5_color_denoise",
        pipeline="fixed_train_eval_v1",
        models=["drunet", "scunet"],
        hyperparameters=[
            Hyperparameter("sigma", HyperparamScope.DATASET, SweepRange(15, 45, 15)),
            Hyperparameter("lr", HyperparamScope.TRAINING, SweepRange(1e-4, 2e-4, 1e-4)),
            # 仅对 scunet 生效的模型级超参
            Hyperparameter("width", HyperparamScope.MODEL, SweepRange(32, 64, 32), target="scunet"),
        ],
        metrics=["PSNR", "SSIM"],
    )

    gs = GridSearch(spec)
    runs = gs.build_runs(model_registry, ds_registry)
    print(f"build_runs -> {len(runs)} runs")

    # 仅用于跑通编排的假实现：指标为占位值，真实值来自 deepinv+KAIR
    class DemoDenoisingTask(DenoisingTask):
        def train(self, run, **kwargs):
            _strict_consume(kwargs, [], ["device"])  # 演示 kwargs 严格检查
            return None

        def evaluate(self, run, **kwargs):
            _strict_consume(kwargs, [], ["device"])
            sigma = run.resolved.get("sigma", 0.0)
            return {"PSNR": 40.0 - sigma / 10.0, "SSIM": 0.9}

    results = gs.run(DemoDenoisingTask(), model_registry, ds_registry)
    print(f"run -> {len(results)} results")
    print(summarize(results))
