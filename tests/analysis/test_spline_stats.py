"""Проверки меры выученной нелинейности.

Мера отвечает на главный вопрос работы: используется ли нелинейность на самом
деле. Ошибка здесь не роняет ничего — она просто выдаёт неверное число,
на котором строится вывод. Поэтому мера проверяется на функциях с заранее
известным ответом.
"""

import math

import pytest
import torch

from kanlora.adapters.base import AdapterConfig
from kanlora.adapters.inject import adapter_modules, inject_adapters
from kanlora.adapters.kan_layer import KANLayer
from kanlora.adapters.spline import greville_abscissae
from kanlora.analysis.spline_stats import (
    _GRID_MARGIN,
    layer_nonlinearity,
    model_nonlinearity,
    nonlinearity_index,
    pooled_nonlinearity,
)

GRID = torch.linspace(-1.0, 1.0, 401, dtype=torch.float64)


def test_straight_line_has_zero_nonlinearity() -> None:
    assert nonlinearity_index(GRID, 3.0 * GRID + 1.5) == pytest.approx(0.0, abs=1e-12)


def test_identity_has_zero_nonlinearity() -> None:
    assert nonlinearity_index(GRID, GRID) == pytest.approx(0.0, abs=1e-12)


def test_symmetric_parabola_is_fully_unexplained_by_a_line() -> None:
    """У x^2 на симметричном отрезке лучшая прямая — константа: прямая не объясняет ничего."""
    assert nonlinearity_index(GRID, GRID**2) == pytest.approx(1.0, abs=1e-6)


def test_line_plus_bump_is_between_zero_and_one() -> None:
    outputs = 2.0 * GRID + 0.1 * torch.sin(4.0 * math.pi * GRID)
    index = nonlinearity_index(GRID, outputs)
    assert 0.0 < index < 1.0


def test_constant_function_is_reported_as_linear() -> None:
    """Вырожденный случай: постоянная функция — частный случай прямой, а не деление на ноль."""
    assert nonlinearity_index(GRID, torch.zeros_like(GRID)) == pytest.approx(0.0)


def test_identity_layer_shows_no_nonlinearity() -> None:
    """Слой при инициализации тождественен, значит мера обязана дать ноль на всех рёбрах."""
    layer = KANLayer(3, 3, grid_size=5, spline_order=3, dtype=torch.float64)
    stats = layer_nonlinearity(layer, lo=-0.99, hi=0.99)

    assert len(stats) == 9
    assert max(item.nonlinearity for item in stats) == pytest.approx(0.0, abs=1e-8)


def test_perturbed_layer_shows_nonlinearity() -> None:
    layer = KANLayer(3, 3, grid_size=5, spline_order=3, dtype=torch.float64)
    torch.manual_seed(0)
    with torch.no_grad():
        layer.spline_coefficients.add_(0.5 * torch.randn_like(layer.spline_coefficients))

    stats = layer_nonlinearity(layer, lo=-0.99, hi=0.99)
    assert max(item.nonlinearity for item in stats) > 0.01


def test_edges_are_identified_by_index() -> None:
    layer = KANLayer(2, 3, grid_size=5, spline_order=3, dtype=torch.float64)
    stats = layer_nonlinearity(layer, lo=-0.9, hi=0.9)
    assert {(item.source, item.target) for item in stats} == {
        (source, target) for source in range(2) for target in range(3)
    }


def test_model_summary_covers_every_kan_adapter(tiny_causal_lm) -> None:
    inject_adapters(tiny_causal_lm, "kan_lora", AdapterConfig(rank=4))
    summary = model_nonlinearity(tiny_causal_lm)

    assert len(summary) == 14  # 2 слоя * 7 проекций
    for values in summary.values():
        assert set(values) == {"index", "fraction_inside_grid"}
        assert values["index"] == pytest.approx(0.0, abs=1e-8)


def test_model_summary_is_empty_for_linear_methods(tiny_causal_lm) -> None:
    inject_adapters(tiny_causal_lm, "lora", AdapterConfig(rank=4))
    assert model_nonlinearity(tiny_causal_lm) == {}


def test_measurement_interval_is_taken_from_the_caller() -> None:
    """Мерить надо там, куда попадают активации, а не по всей области определения."""
    layer = KANLayer(2, 2, grid_size=5, spline_order=3, dtype=torch.float64)
    torch.manual_seed(0)
    with torch.no_grad():
        layer.spline_coefficients.add_(0.5 * torch.randn_like(layer.spline_coefficients))

    wide = max(item.nonlinearity for item in layer_nonlinearity(layer, -0.99, 0.99))
    narrow = max(item.nonlinearity for item in layer_nonlinearity(layer, -0.05, 0.05))
    assert narrow < wide, "на узком отрезке любая гладкая функция ближе к прямой"


def test_model_summary_uses_supplied_ranges(tiny_causal_lm) -> None:
    inject_adapters(tiny_causal_lm, "kan_lora", AdapterConfig(rank=4))
    for _, module in adapter_modules(tiny_causal_lm):
        with torch.no_grad():
            module.kan.spline_coefficients.add_(
                0.5 * torch.randn_like(module.kan.spline_coefficients)
            )

    names = [name for name, _ in adapter_modules(tiny_causal_lm)]
    wide = model_nonlinearity(tiny_causal_lm, {name: (-0.99, 0.99) for name in names})
    narrow = model_nonlinearity(tiny_causal_lm, {name: (-0.05, 0.05) for name in names})

    assert narrow[names[0]]["index"] < wide[names[0]]["index"]


def test_identity_layer_is_linear_even_if_observed_range_exceeds_the_grid() -> None:
    """За сеткой сплайн равен нулю: мерить там нельзя, иначе тождество даст «нелинейность»."""
    layer = KANLayer(3, 3, grid_size=5, spline_order=3, dtype=torch.float64)
    stats = layer_nonlinearity(layer, lo=-3.0, hi=3.0)

    assert len(stats) == 9
    assert max(item.nonlinearity for item in stats) == pytest.approx(0.0, abs=1e-8)


def test_measurement_domain_follows_input_scale() -> None:
    """Сетка в координатах входа — это [-s, s]: отрезок обрезается по ней, а не по [-1, 1]."""
    layer = KANLayer(2, 2, grid_size=5, spline_order=3, dtype=torch.float64)
    with torch.no_grad():
        layer.input_scale.fill_(2.0)
        layer.spline_coefficients.add_(0.5 * torch.randn_like(layer.spline_coefficients))

    clipped = layer_nonlinearity(layer, lo=-5.0, hi=5.0)
    inside = layer_nonlinearity(layer, lo=-_GRID_MARGIN * 2.0, hi=_GRID_MARGIN * 2.0)

    assert [item.nonlinearity for item in clipped] == pytest.approx(
        [item.nonlinearity for item in inside]
    )


def test_measurement_domain_is_per_input_feature() -> None:
    layer = KANLayer(2, 1, grid_size=5, spline_order=3, dtype=torch.float64)
    with torch.no_grad():
        layer.input_scale.copy_(torch.tensor([1.0, 4.0], dtype=torch.float64))
        layer.spline_coefficients.add_(0.5 * torch.randn_like(layer.spline_coefficients))

    wide = layer_nonlinearity(layer, lo=-10.0, hi=10.0)
    exact = [
        layer_nonlinearity(layer, lo=-_GRID_MARGIN * s, hi=_GRID_MARGIN * s)[source].nonlinearity
        for source, s in enumerate((1.0, 4.0))
    ]
    assert [item.nonlinearity for item in wide] == pytest.approx(exact)


def test_range_entirely_outside_the_grid_yields_no_measurement() -> None:
    """Пустое пересечение с областью сетки: мерить нечего, рёбра не попадают в результат."""
    layer = KANLayer(2, 2, grid_size=5, spline_order=3, dtype=torch.float64)
    assert layer_nonlinearity(layer, lo=5.0, hi=6.0) == []


def test_model_summary_ignores_the_part_of_observed_range_outside_the_grid(
    tiny_causal_lm,
) -> None:
    inject_adapters(tiny_causal_lm, "kan_lora", AdapterConfig(rank=4))
    for _, module in adapter_modules(tiny_causal_lm):
        module.observed_lo.fill_(-3.0)
        module.observed_hi.fill_(3.0)

    summary = model_nonlinearity(tiny_causal_lm)
    for values in summary.values():
        assert values["index"] == pytest.approx(0.0, abs=1e-6)


def test_model_summary_is_nan_when_observed_range_is_outside_the_grid(tiny_causal_lm) -> None:
    inject_adapters(tiny_causal_lm, "kan_lora", AdapterConfig(rank=4))
    for _, module in adapter_modules(tiny_causal_lm):
        module.observed_lo.fill_(5.0)
        module.observed_hi.fill_(6.0)

    for values in model_nonlinearity(tiny_causal_lm).values():
        assert math.isnan(values["index"])


def test_pooled_index_ignores_near_zero_edges() -> None:
    """Ребро с ничтожной амплитудой не должно решать за слой.

    Так выглядит KAN-LoRA после нескольких шагов: недиагональные рёбра стартуют с
    нуля и обрастают крошечной кривизной. Индекс отдельного ребра от амплитуды не
    зависит и для него близок к 1, индекс слоя — нет.
    """
    layer = KANLayer(2, 2, grid_size=5, spline_order=3, dtype=torch.float64)
    greville = greville_abscissae(layer.knots, layer.spline_order).to(torch.float64)
    with torch.no_grad():
        layer.spline_coefficients[1, 0] = 1e-4 * greville**2  # ребро source=0 -> target=1

    stats = layer_nonlinearity(layer, lo=-0.99, hi=0.99)
    edge = next(item for item in stats if (item.source, item.target) == (0, 1))
    assert edge.nonlinearity > 0.5
    assert pooled_nonlinearity(stats) < 1e-6


def test_pooled_index_of_one_edge_is_its_own_index() -> None:
    layer = KANLayer(1, 1, grid_size=5, spline_order=3, dtype=torch.float64)
    torch.manual_seed(0)
    with torch.no_grad():
        layer.spline_coefficients.add_(0.5 * torch.randn_like(layer.spline_coefficients))

    stats = layer_nonlinearity(layer, lo=-0.99, hi=0.99)
    assert pooled_nonlinearity(stats) == pytest.approx(stats[0].nonlinearity)


def test_pooled_index_does_not_depend_on_the_overall_scale() -> None:
    layer = KANLayer(2, 2, grid_size=5, spline_order=3, dtype=torch.float64)
    torch.manual_seed(0)
    with torch.no_grad():
        layer.spline_coefficients.add_(0.5 * torch.randn_like(layer.spline_coefficients))
    before = pooled_nonlinearity(layer_nonlinearity(layer, lo=-0.99, hi=0.99))
    with torch.no_grad():
        layer.spline_coefficients.mul_(10.0)
    after = pooled_nonlinearity(layer_nonlinearity(layer, lo=-0.99, hi=0.99))
    assert after == pytest.approx(before)


def test_pooled_index_of_constant_edges_is_zero() -> None:
    """Все рёбра постоянны (ΣTSS = 0): константа — частный случай прямой, не NaN."""
    layer = KANLayer(2, 2, grid_size=5, spline_order=3, dtype=torch.float64)
    with torch.no_grad():
        layer.spline_coefficients.zero_()
    assert pooled_nonlinearity(layer_nonlinearity(layer, lo=-0.99, hi=0.99)) == 0.0


def test_pooled_index_without_edges_is_nan() -> None:
    assert math.isnan(pooled_nonlinearity([]))
