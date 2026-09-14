"""Адаптация с разложением веса на модуль и направление (DoRA).

Вес раскладывается как W = m * V / ||V||, где норма берётся по каждому
выходному каналу отдельно. Направление V получает ту же низкоранговую поправку,
что и в LoRA, а модуль m обучается отдельным вектором. Замысел метода: в ходе
полной тонкой настройки модуль и направление меняются по-разному, а LoRA
вынуждена менять их совместно.

При инициализации m равен норме исходного веса, а B = 0, поэтому
m * W / ||W|| в точности равно W и модель стартует немодифицированной —
как и два других сравниваемых метода.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from kanlora.adapters.base import AdapterConfig, AdapterLinear, _build_merged_linear

__all__ = ["DoRALinear"]


class DoRALinear(AdapterLinear):
    def __init__(self, base: nn.Linear, config: AdapterConfig) -> None:
        super().__init__(base, config)
        factory = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_a = nn.Parameter(torch.empty(config.rank, base.in_features, **factory))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, config.rank, **factory))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        with torch.no_grad():
            self.magnitude = nn.Parameter(base.weight.norm(dim=1).clone())

    def effective_weight(self) -> torch.Tensor:
        direction = self.base.weight + self.scaling * (self.lora_b @ self.lora_a)
        norm = direction.norm(dim=1, keepdim=True)
        return self.magnitude.unsqueeze(1) * direction / norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.effective_weight(), self.base.bias)

    def analytic_parameter_count(self) -> int:
        return (
            self.config.rank * (self.base.in_features + self.base.out_features)
            + self.base.out_features
        )

    @property
    def can_merge(self) -> bool:
        return True

    def merge(self) -> nn.Linear:
        return _build_merged_linear(self.base, self.effective_weight())

    def matrix_parameters(self) -> tuple[nn.Parameter, ...]:
        return (self.lora_a, self.lora_b)
