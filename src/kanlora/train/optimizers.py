"""AdamW для всех обучаемых параметров модели."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 2.0e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
