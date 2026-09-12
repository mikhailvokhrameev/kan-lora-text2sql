# tests/adapters/test_lora.py
"""Проверки классической LoRA.

Главное здесь — нулевая поправка при инициализации. Все три метода обязаны
стартовать из одной точки, иначе разницу в качестве нельзя приписать методу.
"""

import pytest
import torch
from torch import nn

from kanlora.adapters.base import AdapterConfig
from kanlora.adapters.lora import LoRALinear

IN_FEATURES, OUT_FEATURES, RANK = 12, 7, 4


@pytest.fixture
def base() -> nn.Linear:
    torch.manual_seed(0)
    return nn.Linear(IN_FEATURES, OUT_FEATURES, dtype=torch.float64)


@pytest.fixture
def adapter(base: nn.Linear) -> LoRALinear:
    return LoRALinear(base, AdapterConfig(rank=RANK, alpha=2.0 * RANK))


def test_output_shape(adapter: LoRALinear) -> None:
    x = torch.randn(3, 5, IN_FEATURES, dtype=torch.float64)
    assert adapter(x).shape == (3, 5, OUT_FEATURES)


def test_zero_correction_at_initialization(base: nn.Linear, adapter: LoRALinear) -> None:
    """B = 0, поэтому модель с адаптером в точности равна модели без него."""
    x = torch.randn(16, IN_FEATURES, dtype=torch.float64)
    assert torch.allclose(adapter(x), base(x), atol=1e-12)


def test_matrix_a_is_not_zero(adapter: LoRALinear) -> None:
    """Обнулять надо ровно одну матрицу: при A = B = 0 градиент тождественно нулевой."""
    assert adapter.lora_a.abs().sum().item() > 0
    assert adapter.lora_b.abs().sum().item() == pytest.approx(0.0)


def test_delta_equals_scaled_low_rank_product(adapter: LoRALinear) -> None:
    with torch.no_grad():
        adapter.lora_b.normal_()

    x = torch.randn(9, IN_FEATURES, dtype=torch.float64)
    expected = adapter.scaling * (x @ adapter.lora_a.T @ adapter.lora_b.T)
    assert torch.allclose(adapter.delta(x), expected, atol=1e-12)


def test_parameter_count_matches_formula(adapter: LoRALinear) -> None:
    expected = RANK * (IN_FEATURES + OUT_FEATURES)
    assert adapter.analytic_parameter_count() == expected
    assert adapter.trainable_parameter_count() == expected


def test_merged_linear_reproduces_adapter_output(base: nn.Linear, adapter: LoRALinear) -> None:
    """LoRA сливается с весами и на инференсе не стоит ничего — в этом её отличие."""
    with torch.no_grad():
        adapter.lora_b.normal_()

    assert adapter.can_merge is True
    merged = adapter.merge()
    x = torch.randn(16, IN_FEATURES, dtype=torch.float64)
    assert torch.allclose(merged(x), adapter(x), atol=1e-10)


def test_merge_matches_base_bias_dtype_and_device(base: nn.Linear, adapter: LoRALinear) -> None:
    merged = adapter.merge()
    assert isinstance(merged, nn.Linear)
    assert (merged.bias is not None) == (base.bias is not None)
    assert merged.weight.dtype == base.weight.dtype
    assert merged.weight.device == base.weight.device


def test_merge_omits_bias_when_base_has_none() -> None:
    base_without_bias = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False, dtype=torch.float64)
    adapter_without_bias = LoRALinear(base_without_bias, AdapterConfig(rank=RANK, alpha=2.0 * RANK))
    merged = adapter_without_bias.merge()
    assert merged.bias is None


def test_merge_does_not_touch_the_original_base(base: nn.Linear, adapter: LoRALinear) -> None:
    original = base.weight.detach().clone()
    with torch.no_grad():
        adapter.lora_b.normal_()
    adapter.merge()
    assert torch.equal(base.weight, original)


def test_gradients_reach_both_matrices(adapter: LoRALinear) -> None:
    with torch.no_grad():
        adapter.lora_b.normal_()

    adapter(torch.randn(4, IN_FEATURES, dtype=torch.float64)).sum().backward()
    assert adapter.lora_a.grad is not None
    assert adapter.lora_b.grad is not None
    assert adapter.base.weight.grad is None
