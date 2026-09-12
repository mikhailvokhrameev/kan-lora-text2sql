"""Конфигурация одного прогона.

Заморожена целиком и попадает в карточку результата дословно: воспроизводимость
держится на том, что по карточке прогон восстанавливается без обращения
к истории команд.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import yaml

from kanlora.adapters.base import AdapterConfig
from kanlora.train.loop import TrainConfig
from kanlora.train.optimizers import OptimizerConfig

__all__ = ["ExperimentConfig", "apply_overrides", "config_to_dict", "load_config"]


@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str
    data_root: str
    method: str
    dataset: str
    seed: int
    train_subset: int
    subsample_seed: int
    max_length: int
    eval_limit: int | None
    eval_batch_size: int
    max_new_tokens: int
    adapter: AdapterConfig
    train: TrainConfig
    optimizer: OptimizerConfig

    @property
    def run_id(self) -> str:
        """Имя прогона, различающее все ячейки матрицы экспериментов."""
        name = f"{self.method}-{self.dataset}-{self.optimizer.name}-seed{self.seed}"
        if self.method == "kan_lora" and not self.adapter.learn_input_scale:
            name += "-no-input-scale"
        return name


def load_config(path: Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ExperimentConfig(
        model_name=raw["model_name"],
        data_root=raw["data_root"],
        method=raw["method"],
        dataset=raw["dataset"],
        seed=raw["seed"],
        train_subset=raw["train_subset"],
        subsample_seed=raw["subsample_seed"],
        max_length=raw["max_length"],
        eval_limit=raw["eval_limit"],
        eval_batch_size=raw["eval_batch_size"],
        max_new_tokens=raw["max_new_tokens"],
        adapter=AdapterConfig(**raw["adapter"]),
        train=TrainConfig(seed=raw["seed"], **raw["train"]),
        optimizer=OptimizerConfig(**raw["optimizer"]),
    )


def apply_overrides(
    config: ExperimentConfig,
    *,
    method: str | None = None,
    dataset: str | None = None,
    seed: int | None = None,
    optimizer_name: str | None = None,
    learn_input_scale: bool | None = None,
    train_subset: int | None = None,
    eval_limit: int | None = None,
) -> ExperimentConfig:
    """Накладывает ключи командной строки. None означает «не трогать»."""
    updated = config
    if method is not None:
        updated = replace(updated, method=method)
    if dataset is not None:
        updated = replace(updated, dataset=dataset)
    if seed is not None:
        updated = replace(updated, seed=seed, train=replace(updated.train, seed=seed))
    if optimizer_name is not None:
        updated = replace(updated, optimizer=replace(updated.optimizer, name=optimizer_name))
    if learn_input_scale is not None:
        updated = replace(
            updated, adapter=replace(updated.adapter, learn_input_scale=learn_input_scale)
        )
    if train_subset is not None:
        updated = replace(updated, train_subset=train_subset)
    if eval_limit is not None:
        updated = replace(updated, eval_limit=eval_limit)
    return updated


def config_to_dict(config: ExperimentConfig) -> dict:
    data = asdict(config)
    data["train"].pop("seed", None)  # зерно живёт на верхнем уровне
    return data
