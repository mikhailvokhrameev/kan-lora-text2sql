"""Нелинейная низкоранговая адаптация.

    delta_W * x = (alpha / r) * B * phi(A x),

где phi — слой Колмогорова — Арнольда: обучаемые одномерные функции на рёбрах,
заданные B-сплайнами. Отличие от LoRA ровно одно — между двумя низкоранговыми
матрицами появляется нелинейность.

Слой phi при инициализации тождественен, а B обнулена, поэтому KAN-LoRA
стартует численно совпадающим с LoRA. Это и делает сравнение осмысленным:
всё последующее расхождение приписывается выученной нелинейности.

Поправка зависит от входа нелинейно, поэтому она не сводится к одной матрице.
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

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        projected = torch.nn.functional.linear(x, self.lora_a)
        return self.scaling * torch.nn.functional.linear(self.kan(projected), self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.delta(x)

    def analytic_parameter_count(self) -> int:
        rank = self.config.rank
        basis = self.config.grid_size + self.config.spline_order
        kan_parameters = rank * rank * (basis + 2) + (
            rank if self.config.learn_input_scale else 0
        )
        return rank * (self.base.in_features + self.base.out_features) + kan_parameters
