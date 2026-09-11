"""AdamW и Muon за единым интерфейсом.

Muon ортогонализует обновления и применим только к двумерным матрицам —
всё остальное он отвергает жёсткой ошибкой. Поэтому три сравниваемых метода
покрываются им по-разному: LoRA целиком (обе матрицы двумерны), DoRA частично
(вектор модуля одномерен), KAN-LoRA частично (коэффициенты сплайнов
трёхмерны). Непокрытые параметры идут в запасной AdamW, и прогон с Muon для
KAN-LoRA всегда гибридный. Это самостоятельный результат работы: индуктивное
смещение современного оптимизатора не распространяется на нелинейную часть
адаптера.

Оба оптимизатора видят одну скорость обучения — как и все три метода.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.optim import Optimizer

__all__ = ["OptimizerBundle", "OptimizerConfig", "build_optimizer", "split_by_dimension"]


@dataclass(frozen=True)
class OptimizerConfig:
    name: Literal["adamw", "muon"] = "adamw"
    learning_rate: float = 2.0e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0


def split_by_dimension(
    parameters: Iterable[nn.Parameter],
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Делит обучаемые параметры на двумерные и все прочие."""
    trainable = [p for p in parameters if p.requires_grad]
    return (
        [p for p in trainable if p.dim() == 2],
        [p for p in trainable if p.dim() != 2],
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


def build_optimizer(
    parameters: Iterable[nn.Parameter], config: OptimizerConfig
) -> OptimizerBundle:
    builders = {"adamw": _build_adamw, "muon": _build_muon}
    return builders[config.name](list(parameters), config)


def _build_adamw(parameters: list[nn.Parameter], config: OptimizerConfig) -> OptimizerBundle:
    trainable = [p for p in parameters if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    return OptimizerBundle([optimizer], {"adamw": trainable, "muon": []})


def _build_muon(parameters: list[nn.Parameter], config: OptimizerConfig) -> OptimizerBundle:
    matrices, rest = split_by_dimension(parameters)

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
