"""Общий интерфейс адаптеров.

Все три сравниваемых метода подставляются вместо nn.Linear и обязаны быть
взаимозаменяемы для цикла обучения, генерации и замера памяти. Различия
методов не должны просачиваться в код вокруг: иначе сравнение начнёт мерить
качество реализаций, а не свойства методов.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn

__all__ = ["AdapterConfig", "AdapterLinear", "freeze"]


@dataclass(frozen=True)
class AdapterConfig:
    """Гиперпараметры адаптера, общие для всех трёх методов.

    Заморожен намеренно: гиперпараметры фиксируются один раз и не подбираются,
    поэтому случайная правка поля в середине матрицы прогонов невозможна.
    Поля grid_size, spline_order и learn_input_scale используются только
    KAN-LoRA; для LoRA и DoRA они игнорируются и хранятся ради того, чтобы
    карточка результата содержала одинаковый набор полей для всех прогонов.
    """

    rank: int = 8
    alpha: float = 16.0
    grid_size: int = 5
    spline_order: int = 3
    learn_input_scale: bool = True


def freeze(module: nn.Module) -> None:
    """Снимает requires_grad со всех параметров модуля."""
    for parameter in module.parameters():
        parameter.requires_grad_(False)


class AdapterLinear(nn.Module, ABC):
    """Замороженный nn.Linear плюс обучаемая поправка."""

    def __init__(self, base: nn.Linear, config: AdapterConfig) -> None:
        super().__init__()
        self.base = base
        self.config = config
        self.scaling = config.alpha / config.rank
        freeze(self.base)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def analytic_parameter_count(self) -> int:
        """Число обучаемых параметров по формуле, независимо от реализации.

        Существует ради теста: расхождение с фактическим счётчиком означает,
        что утверждение о равном бюджете параметров в таблицах неверно.
        """

    @property
    @abstractmethod
    def can_merge(self) -> bool:
        """Сливается ли поправка с весами основы после обучения."""

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def merge(self) -> nn.Linear:
        """Возвращает обычный nn.Linear с поглощённой поправкой."""
        raise NotImplementedError(
            f"{type(self).__name__} нелинеен и не сливается с весами: "
            "поправка зависит от входа, а не только от весов"
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.base.in_features}, out_features={self.base.out_features}, "
            f"rank={self.config.rank}, alpha={self.config.alpha}"
        )
