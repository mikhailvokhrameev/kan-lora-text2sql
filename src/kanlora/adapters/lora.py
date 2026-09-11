# src/kanlora/adapters/lora.py
"""Классическая низкоранговая адаптация.

Поправка к весу раскладывается в произведение двух матриц ранга r:
    delta_W * x = (alpha / r) * B (A x),   A: (r, in),   B: (out, r).
Матрица B инициализируется нулём, поэтому в начале обучения модель не изменена.
Обнуляется ровно одна из матриц: при A = B = 0 градиент по обеим тождественно
равен нулю и адаптер не сдвинулся бы с места.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from kanlora.adapters.base import AdapterConfig, AdapterLinear

__all__ = ["LoRALinear"]


class LoRALinear(AdapterLinear):
    def __init__(self, base: nn.Linear, config: AdapterConfig) -> None:
        super().__init__(base, config)
        factory = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_a = nn.Parameter(torch.empty(config.rank, base.in_features, **factory))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, config.rank, **factory))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        projected = torch.nn.functional.linear(x, self.lora_a)
        return self.scaling * torch.nn.functional.linear(projected, self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.delta(x)

    def analytic_parameter_count(self) -> int:
        return self.config.rank * (self.base.in_features + self.base.out_features)

    @property
    def can_merge(self) -> bool:
        return True

    def merge(self) -> nn.Linear:
        merged = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        with torch.no_grad():
            merged.weight.copy_(self.base.weight + self.scaling * (self.lora_b @ self.lora_a))
            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias)
        return merged
