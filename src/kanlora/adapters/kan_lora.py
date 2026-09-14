"""Нелинейная низкоранговая адаптация.

    delta_W * x = (alpha / r) * B * phi(A x),

где phi — слой Колмогорова — Арнольда: обучаемые одномерные функции на рёбрах,
заданные B-сплайнами. Отличие от LoRA ровно одно — между двумя низкоранговыми
матрицами появляется нелинейность.

Слой phi при инициализации тождественен, а B обнулена, поэтому KAN-LoRA
стартует численно совпадающим с LoRA. Это и делает сравнение осмысленным:
всё последующее расхождение приписывается выученной нелинейности.

Сливаться с весами адаптер не может в принципе: поправка зависит от входа
нелинейно, поэтому задержку на инференсе метод платит всегда. Это его цена,
и она замеряется отдельно.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from kanlora.adapters.base import AdapterConfig, AdapterLinear
from kanlora.adapters.kan_layer import KANLayer

__all__ = ["KANLoRALinear"]


class KANLoRALinear(AdapterLinear):
    def __init__(self, base: nn.Linear, config: AdapterConfig) -> None:
        super().__init__(base, config)
        factory = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_a = nn.Parameter(torch.empty(config.rank, base.in_features, **factory))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, config.rank, **factory))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

        self.kan = KANLayer(
            config.rank,
            config.rank,
            grid_size=config.grid_size,
            spline_order=config.spline_order,
            **factory,
        )
        if not config.learn_input_scale:
            self.kan.input_scale.requires_grad_(False)

        self._fraction_inside_grid: torch.Tensor | None = None
        self.register_buffer("observed_lo", torch.tensor(float("inf")))
        self.register_buffer("observed_hi", torch.tensor(float("-inf")))

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        projected = torch.nn.functional.linear(x, self.lora_a)
        with torch.no_grad():
            self._fraction_inside_grid = self.kan.fraction_inside_grid(projected)
            self.observed_lo = torch.minimum(self.observed_lo, projected.min())
            self.observed_hi = torch.maximum(self.observed_hi, projected.max())
        return self.scaling * torch.nn.functional.linear(self.kan(projected), self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.delta(x)

    def last_fraction_inside_grid(self) -> float | None:
        """Доля входов сплайна, попавших в сетку на последнем проходе.

        Если она мала, базис обнуляется и адаптер вырождается в LoRA с
        неработающими параметрами — самая частая скрытая ошибка в реализациях
        KAN-адаптеров. Поэтому величина снимается на каждом проходе.
        """
        return self._fraction_inside_grid.item() if self._fraction_inside_grid is not None else None

    def observed_range(self) -> tuple[float, float]:
        """Фактический размах входа слоя KAN, накопленный за все проходы.

        `(inf, -inf)` до первого прохода — намеренно немой диапазон, а не
        значение по умолчанию сетки: пустой размах обязан быть отличим от
        размаха, реально попавшего внутрь сетки.
        """
        return float(self.observed_lo), float(self.observed_hi)

    def analytic_parameter_count(self) -> int:
        rank = self.config.rank
        basis = self.config.grid_size + self.config.spline_order
        kan_parameters = rank * rank * (basis + 2) + (
            rank if self.config.learn_input_scale else 0
        )
        return rank * (self.base.in_features + self.base.out_features) + kan_parameters

    @property
    def can_merge(self) -> bool:
        return False

    def matrix_parameters(self) -> tuple[nn.Parameter, ...]:
        return (self.lora_a, self.lora_b)
