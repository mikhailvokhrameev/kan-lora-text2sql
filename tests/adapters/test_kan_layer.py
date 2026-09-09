"""Проверки слоя Колмогорова — Арнольда.

Слой обязан стартовать тождественным отображением: на этом стоит вся схема
сравнения — KAN-LoRA в начале обучения совпадает с обычной LoRA, поэтому любое
последующее расхождение объясняется выученной нелинейностью, а не инициализацией.
"""

import pytest
import torch

from kanlora.adapters.kan_layer import KANLayer

RANK = 8
GRID_SIZE = 5
SPLINE_ORDER = 3


@pytest.fixture
def layer() -> KANLayer:
    torch.manual_seed(0)
    return KANLayer(
        RANK, RANK, grid_size=GRID_SIZE, spline_order=SPLINE_ORDER, dtype=torch.float64
    )


def inside_grid(n: int = 64) -> torch.Tensor:
    return torch.linspace(-0.99, 0.99, n, dtype=torch.float64).unsqueeze(1).repeat(1, RANK)


def test_output_shape(layer: KANLayer) -> None:
    x = torch.randn(3, 5, RANK, dtype=torch.float64)
    assert layer(x).shape == (3, 5, RANK)


def test_starts_as_identity_inside_grid(layer: KANLayer) -> None:
    """Внутри сетки слой при инициализации — тождество, с точностью до счёта."""
    x = inside_grid()
    assert torch.allclose(layer(x), x, atol=1e-10)


def test_off_diagonal_edges_start_silent(layer: KANLayer) -> None:
    """Недиагональные рёбра обнулены: каналы в начале не перемешиваются."""
    x = torch.zeros(1, RANK, dtype=torch.float64)
    x[0, 0] = 0.5

    out = layer(x)
    assert out[0, 0].item() == pytest.approx(0.5, abs=1e-10)
    assert torch.allclose(out[0, 1:], torch.zeros(RANK - 1, dtype=torch.float64), atol=1e-10)


def test_parameter_count_matches_formula(layer: KANLayer) -> None:
    basis = GRID_SIZE + SPLINE_ORDER
    expected = RANK * RANK * basis + RANK * RANK + RANK * RANK + RANK
    assert sum(p.numel() for p in layer.parameters()) == expected
    assert layer.parameter_count() == expected


def test_all_parameters_receive_gradients(layer: KANLayer) -> None:
    layer(inside_grid()).sum().backward()

    for name, parameter in layer.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_reports_fraction_of_activations_inside_grid(layer: KANLayer) -> None:
    """Диагностика вырождения: доля входов, попавших в область определения сетки."""
    x = torch.tensor([[0.0] * RANK, [5.0] * RANK], dtype=torch.float64)
    assert layer.fraction_inside_grid(x).item() == pytest.approx(0.5)


def test_degenerates_outside_grid(layer: KANLayer) -> None:
    """Физика процесса: вне сетки базис обнуляется и слой теряет сигнал.

    Именно поэтому размах активаций приходится контролировать, а не надеяться,
    что он сам окажется внутри [-1, 1].
    """
    far = torch.full((1, RANK), 50.0, dtype=torch.float64)
    assert torch.allclose(layer(far), torch.zeros(1, RANK, dtype=torch.float64), atol=1e-10)


def test_learnable_scale_stretches_grid(layer: KANLayer) -> None:
    """Растяжение сетки возвращает тождество на более широком отрезке."""
    with torch.no_grad():
        layer.input_scale.fill_(10.0)

    x = torch.full((1, RANK), 5.0, dtype=torch.float64)
    assert torch.allclose(layer(x), x, atol=1e-9)
