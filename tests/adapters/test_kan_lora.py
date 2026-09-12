"""Проверки нелинейного адаптера B * phi(A x).

Замысел всей работы держится на одном свойстве: при тождественных сплайнах
KAN-LoRA численно совпадает с LoRA. Тогда любое расхождение в качестве после
обучения объясняется выученной нелинейностью, а не другой точкой старта или
другим масштабом градиентов. Это свойство проверяется здесь напрямую.
"""

import pytest
import torch
from torch import nn

from kanlora.adapters.base import AdapterConfig
from kanlora.adapters.kan_lora import KANLoRALinear
from kanlora.adapters.lora import LoRALinear

IN_FEATURES, OUT_FEATURES, RANK = 12, 7, 4
GRID_SIZE, SPLINE_ORDER = 5, 3


@pytest.fixture
def base() -> nn.Linear:
    torch.manual_seed(0)
    return nn.Linear(IN_FEATURES, OUT_FEATURES, dtype=torch.float64)


@pytest.fixture
def config() -> AdapterConfig:
    return AdapterConfig(
        rank=RANK, alpha=2.0 * RANK, grid_size=GRID_SIZE, spline_order=SPLINE_ORDER
    )


@pytest.fixture
def adapter(base: nn.Linear, config: AdapterConfig) -> KANLoRALinear:
    return KANLoRALinear(base, config)


def small_input(rows: int = 16) -> torch.Tensor:
    """Вход, при котором A x гарантированно попадает внутрь сетки [-1, 1].

    Масштаб подобран так, чтобы проверялось совпадение с LoRA, а не поведение
    сплайна за границами сетки, — за границы отвечает отдельный тест.
    """
    torch.manual_seed(7)
    return 0.02 * torch.randn(rows, IN_FEATURES, dtype=torch.float64)


def test_output_shape(adapter: KANLoRALinear) -> None:
    x = torch.randn(3, 5, IN_FEATURES, dtype=torch.float64)
    assert adapter(x).shape == (3, 5, OUT_FEATURES)


def test_zero_correction_at_initialization(base: nn.Linear, adapter: KANLoRALinear) -> None:
    x = torch.randn(16, IN_FEATURES, dtype=torch.float64)
    assert torch.allclose(adapter(x), base(x), atol=1e-12)


def test_matches_lora_with_identity_splines(base: nn.Linear, config: AdapterConfig) -> None:
    """ГЛАВНЫЙ ТЕСТ. При phi = тождество KAN-LoRA поэлементно совпадает с LoRA.

    Совпадение точное, а не приближённое: тождественность обеспечена свойством
    линейной точности B-сплайнов, а не подгонкой коэффициентов.
    """
    torch.manual_seed(1)
    kan_lora = KANLoRALinear(base, config)
    lora = LoRALinear(base, config)
    with torch.no_grad():
        lora.lora_a.copy_(kan_lora.lora_a)
        kan_lora.lora_b.normal_()
        lora.lora_b.copy_(kan_lora.lora_b)

    x = small_input()
    assert kan_lora.kan.fraction_inside_grid(x @ kan_lora.lora_a.T).item() == pytest.approx(1.0)
    assert torch.allclose(kan_lora(x), lora(x), atol=1e-10)


def test_learned_nonlinearity_moves_output_away_from_lora(
    base: nn.Linear, config: AdapterConfig
) -> None:
    """Обратная сторона главного теста: со сдвинутыми сплайнами выходы расходятся.

    Без неё главный тест проходил бы и на реализации, где сплайны ни на что
    не влияют, — то есть на скрытой LoRA.
    """
    torch.manual_seed(1)
    kan_lora = KANLoRALinear(base, config)
    lora = LoRALinear(base, config)
    with torch.no_grad():
        lora.lora_a.copy_(kan_lora.lora_a)
        kan_lora.lora_b.normal_()
        lora.lora_b.copy_(kan_lora.lora_b)
        kan_lora.kan.spline_coefficients.add_(0.3 * torch.randn_like(kan_lora.kan.spline_coefficients))

    x = small_input()
    assert not torch.allclose(kan_lora(x), lora(x), atol=1e-4)


def test_parameter_count_matches_formula(adapter: KANLoRALinear) -> None:
    """Формула: LoRA + слой KAN размера r x r.

    Слой KAN добавляет r^2 * (grid + order) коэффициентов сплайна,
    r^2 масштабов сплайна, r^2 весов базовой ветви и r масштабов входа.
    """
    basis = GRID_SIZE + SPLINE_ORDER
    expected = (
        RANK * (IN_FEATURES + OUT_FEATURES) + RANK * RANK * (basis + 2) + RANK
    )
    assert adapter.analytic_parameter_count() == expected
    assert adapter.trainable_parameter_count() == expected


def test_refuses_to_merge(adapter: KANLoRALinear) -> None:
    """Нелинейный адаптер не сливается с весами и платит задержкой всегда."""
    assert adapter.can_merge is False
    with pytest.raises(NotImplementedError):
        adapter.merge()


def test_reports_fraction_inside_grid_after_forward(adapter: KANLoRALinear) -> None:
    """Диагностика вырождения снимается на каждом проходе, а не отдельным прогоном."""
    assert adapter.last_fraction_inside_grid() is None
    adapter(small_input())
    assert adapter.last_fraction_inside_grid() == pytest.approx(1.0)

    adapter(1e4 * torch.randn(16, IN_FEATURES, dtype=torch.float64))
    assert adapter.last_fraction_inside_grid() < 0.5


def test_gradients_reach_spline_coefficients(adapter: KANLoRALinear) -> None:
    with torch.no_grad():
        adapter.lora_b.normal_()

    adapter(small_input(4)).sum().backward()
    for name in ("lora_a", "lora_b"):
        assert getattr(adapter, name).grad is not None, name
    assert adapter.kan.spline_coefficients.grad is not None
    assert torch.isfinite(adapter.kan.spline_coefficients.grad).all()
    assert adapter.base.weight.grad is None


def test_frozen_input_scale_is_excluded_from_training(base: nn.Linear) -> None:
    """Режим абляции: сетка не растягивается, масштаб входа заморожен на 1.0."""
    config = AdapterConfig(
        rank=RANK, alpha=2.0 * RANK, grid_size=GRID_SIZE,
        spline_order=SPLINE_ORDER, learn_input_scale=False,
    )
    adapter = KANLoRALinear(base, config)

    assert adapter.kan.input_scale.requires_grad is False
    assert torch.allclose(adapter.kan.input_scale, torch.ones_like(adapter.kan.input_scale))
    assert adapter.analytic_parameter_count() == adapter.trainable_parameter_count()


def test_observed_range_expands_after_forward_pass(adapter: KANLoRALinear) -> None:
    """Мера нелинейности обязана мерить там, куда реально попадают активации."""
    assert adapter.observed_range() == (float("inf"), float("-inf"))

    adapter(small_input())
    first_lo, first_hi = adapter.observed_range()
    assert first_lo < first_hi

    adapter(10.0 * small_input())
    second_lo, second_hi = adapter.observed_range()
    assert second_lo <= first_lo
    assert second_hi >= first_hi
