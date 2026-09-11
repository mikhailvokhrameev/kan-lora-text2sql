"""Проверки DoRA.

DoRA раскладывает вес на модуль и направление: W = m * V / ||V||, где норма
берётся построчно, по каждому выходному каналу. Тесты закрепляют два факта,
без которых сравнение теряет смысл: старт из немодифицированной модели и
сводимость к LoRA, когда модуль равен норме поправленного веса.
"""

import pytest
import torch
from torch import nn

from kanlora.adapters.base import AdapterConfig
from kanlora.adapters.dora import DoRALinear
from kanlora.adapters.lora import LoRALinear

IN_FEATURES, OUT_FEATURES, RANK = 12, 7, 4


@pytest.fixture
def base() -> nn.Linear:
    torch.manual_seed(0)
    return nn.Linear(IN_FEATURES, OUT_FEATURES, dtype=torch.float64)


@pytest.fixture
def config() -> AdapterConfig:
    return AdapterConfig(rank=RANK, alpha=2.0 * RANK)


@pytest.fixture
def adapter(base: nn.Linear, config: AdapterConfig) -> DoRALinear:
    return DoRALinear(base, config)


def test_output_shape(adapter: DoRALinear) -> None:
    x = torch.randn(3, 5, IN_FEATURES, dtype=torch.float64)
    assert adapter(x).shape == (3, 5, OUT_FEATURES)


def test_magnitude_is_initialized_to_row_norms(base: nn.Linear, adapter: DoRALinear) -> None:
    expected = base.weight.norm(dim=1)
    assert adapter.magnitude.shape == (OUT_FEATURES,)
    assert torch.allclose(adapter.magnitude, expected, atol=1e-12)


def test_zero_correction_at_initialization(base: nn.Linear, adapter: DoRALinear) -> None:
    """m равен норме W, а B = 0, поэтому m * W / ||W|| в точности равно W."""
    x = torch.randn(16, IN_FEATURES, dtype=torch.float64)
    assert torch.allclose(adapter(x), base(x), atol=1e-10)


def test_reduces_to_lora_when_magnitude_matches_direction_norm(
    base: nn.Linear, config: AdapterConfig
) -> None:
    """При m = ||W + s*BA|| нормировка сокращается и DoRA совпадает с LoRA.

    Это и есть смысл утверждения «DoRA обобщает LoRA»: без отдельного модуля
    остаётся ровно LoRA.
    """
    torch.manual_seed(1)
    dora = DoRALinear(base, config)
    lora = LoRALinear(base, config)
    with torch.no_grad():
        lora.lora_a.copy_(dora.lora_a)
        dora.lora_b.normal_()
        lora.lora_b.copy_(dora.lora_b)
        direction = base.weight + dora.scaling * (dora.lora_b @ dora.lora_a)
        dora.magnitude.copy_(direction.norm(dim=1))

    x = torch.randn(16, IN_FEATURES, dtype=torch.float64)
    assert torch.allclose(dora(x), lora(x), atol=1e-10)


def test_parameter_count_matches_formula(adapter: DoRALinear) -> None:
    expected = RANK * (IN_FEATURES + OUT_FEATURES) + OUT_FEATURES
    assert adapter.analytic_parameter_count() == expected
    assert adapter.trainable_parameter_count() == expected


def test_merged_linear_reproduces_adapter_output(adapter: DoRALinear) -> None:
    with torch.no_grad():
        adapter.lora_b.normal_()
        adapter.magnitude.mul_(1.3)

    assert adapter.can_merge is True
    merged = adapter.merge()
    x = torch.randn(16, IN_FEATURES, dtype=torch.float64)
    assert torch.allclose(merged(x), adapter(x), atol=1e-10)


def test_gradients_reach_magnitude_and_both_matrices(adapter: DoRALinear) -> None:
    with torch.no_grad():
        adapter.lora_b.normal_()

    adapter(torch.randn(4, IN_FEATURES, dtype=torch.float64)).sum().backward()
    for name in ("lora_a", "lora_b", "magnitude"):
        parameter = getattr(adapter, name)
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert adapter.base.weight.grad is None
