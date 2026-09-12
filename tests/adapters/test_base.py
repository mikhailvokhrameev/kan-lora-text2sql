"""Проверки общего интерфейса адаптеров.

Интерфейс существует ради одного: обучение, оценка и замер памяти для LoRA,
DoRA и KAN-LoRA обязаны идти по одному и тому же коду. Тесты закрепляют то,
на что этот код опирается: замороженную основу, счётчик обучаемых параметров
и явный ответ на вопрос, сливается ли адаптер с весами.
"""

import pytest
import torch
from torch import nn

from kanlora.adapters.base import AdapterConfig, AdapterLinear, freeze

IN_FEATURES, OUT_FEATURES = 6, 4


class ConstantAdapter(AdapterLinear):
    """Простейшая реализация интерфейса — только чтобы проверить сам интерфейс."""

    def __init__(self, base: nn.Linear, config: AdapterConfig) -> None:
        super().__init__(base, config)
        self.shift = nn.Parameter(torch.zeros(base.out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.shift

    def analytic_parameter_count(self) -> int:
        return self.base.out_features

    @property
    def can_merge(self) -> bool:
        return False


@pytest.fixture
def adapter() -> ConstantAdapter:
    torch.manual_seed(0)
    return ConstantAdapter(nn.Linear(IN_FEATURES, OUT_FEATURES), AdapterConfig())


def test_scaling_is_alpha_over_rank() -> None:
    base = nn.Linear(IN_FEATURES, OUT_FEATURES)
    adapter = ConstantAdapter(base, AdapterConfig(rank=8, alpha=16.0))
    assert adapter.scaling == pytest.approx(2.0)


def test_base_is_frozen(adapter: ConstantAdapter) -> None:
    assert adapter.base.weight.requires_grad is False
    assert adapter.base.bias.requires_grad is False


def test_trainable_count_excludes_frozen_base(adapter: ConstantAdapter) -> None:
    assert adapter.trainable_parameter_count() == OUT_FEATURES
    assert adapter.trainable_parameter_count() == adapter.analytic_parameter_count()


def test_freeze_marks_every_parameter(adapter: ConstantAdapter) -> None:
    freeze(adapter)
    assert all(not p.requires_grad for p in adapter.parameters())


def test_non_mergeable_adapter_refuses_to_merge(adapter: ConstantAdapter) -> None:
    assert adapter.can_merge is False
    with pytest.raises(NotImplementedError):
        adapter.merge()


def test_interface_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        AdapterLinear(nn.Linear(IN_FEATURES, OUT_FEATURES), AdapterConfig())


def test_default_last_fraction_inside_grid_is_none(adapter: ConstantAdapter) -> None:
    """Линейные адаптеры не ведут статистику сетки — цикл обучения не обязан о них знать."""
    assert adapter.last_fraction_inside_grid() is None
