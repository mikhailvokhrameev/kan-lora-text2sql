"""AdamW и Muon за единым интерфейсом.

Muon ортогонализует направление обновления линейного отображения и
осмыслен только для параметров, которые вычисляются как W @ x — то есть
для настоящих матриц (`AdapterLinear.matrix_parameters()`), а не для любого
тензора, которому случайно досталась двумерная форма. У KAN-LoRA ровно две
таких матрицы (`lora_a`, `lora_b`) — всё остальное, включая двумерные
`spline_scale`/`base_weight` слоя `KANLayer` (они входят в вычисление
поэлементно, а не матричным умножением), идёт в запасной AdamW. Прогон
KAN-LoRA с Muon поэтому всегда гибридный — это самостоятельный результат
работы, а не техническая деталь.

Оба оптимизатора видят одну скорость обучения — как и все три метода.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.optim import Optimizer

from kanlora.adapters.inject import adapter_modules

__all__ = ["OptimizerBundle", "OptimizerConfig", "build_optimizer", "split_by_matrix_role"]


@dataclass(frozen=True)
class OptimizerConfig:
    name: Literal["adamw", "muon"] = "adamw"
    learning_rate: float = 2.0e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0


def split_by_matrix_role(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Делит обучаемые параметры модели на настоящие матрицы отображения и всё прочее.

    «Настоящая матрица» — то, что каждый внедрённый адаптер сам называет через
    `matrix_parameters()`, а не любой параметр с `dim() == 2`: у KAN-LoRA
    `spline_scale` и `base_weight` тоже двумерны по форме, но участвуют в
    вычислении поэлементно, а не как матрица `y = W @ x`, и поэтому не
    должны считаться пригодными для ортогонализации Ньютона — Шульца.
    """
    matrix_ids = {
        id(parameter)
        for _, adapter in adapter_modules(model)
        for parameter in adapter.matrix_parameters()
    }
    trainable = [p for p in model.parameters() if p.requires_grad]
    return (
        [p for p in trainable if id(p) in matrix_ids],
        [p for p in trainable if id(p) not in matrix_ids],
    )


class OptimizerBundle:
    """Один или два оптимизатора, ведущие себя как один."""

    def __init__(self, optimizers: list[Optimizer], groups: dict[str, list[nn.Parameter]]) -> None:
        self.optimizers = optimizers
        self._groups = groups
        self.parameters = [p for group in groups.values() for p in group]

    def zero_grad(self) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def clip_grad_norm_(self, max_norm: float) -> torch.Tensor:
        return torch.nn.utils.clip_grad_norm_(self.parameters, max_norm)

    def coverage(self) -> dict[str, int]:
        """Сколько параметров досталось каждому оптимизатору. Идёт в карточку результата."""
        return {name: sum(p.numel() for p in group) for name, group in self._groups.items()}


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> OptimizerBundle:
    builders = {"adamw": _build_adamw, "muon": _build_muon}
    return builders[config.name](model, config)


def _build_adamw(model: nn.Module, config: OptimizerConfig) -> OptimizerBundle:
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    return OptimizerBundle([optimizer], {"adamw": trainable, "muon": []})


def _build_muon(model: nn.Module, config: OptimizerConfig) -> OptimizerBundle:
    matrices, rest = split_by_matrix_role(model)

    optimizers: list[Optimizer] = []
    if matrices:
        optimizers.append(
            torch.optim.Muon(
                matrices, lr=config.learning_rate, weight_decay=config.weight_decay
            )
        )
    if rest:
        optimizers.append(
            torch.optim.AdamW(rest, lr=config.learning_rate, weight_decay=config.weight_decay)
        )

    return OptimizerBundle(optimizers, {"muon": matrices, "adamw": rest})
