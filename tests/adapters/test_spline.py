"""Проверки B-сплайнового базиса.

Базис — математическое основание KAN-LoRA. Ошибка здесь не роняет обучение,
а тихо подменяет измеряемую нелинейность артефактом реализации, поэтому
свойства базиса проверяются напрямую, а не через поведение модели.
"""

import pytest
import torch

from kanlora.adapters.spline import bspline_basis, greville_abscissae, make_knots

GRID_SIZE = 5
SPLINE_ORDER = 3
LO, HI = -1.0, 1.0


@pytest.fixture
def knots() -> torch.Tensor:
    return make_knots(GRID_SIZE, SPLINE_ORDER, LO, HI, dtype=torch.float64)


def sample_domain(n: int = 257) -> torch.Tensor:
    return torch.linspace(LO, HI, n, dtype=torch.float64)


def test_knot_vector_is_uniform_and_extended(knots: torch.Tensor) -> None:
    assert knots.shape == (GRID_SIZE + 2 * SPLINE_ORDER + 1,)

    spacing = knots[1:] - knots[:-1]
    assert torch.allclose(spacing, spacing[0].expand_as(spacing))

    # Границы отрезка обязаны попасть в узлы ровно после k расширяющих узлов.
    assert knots[SPLINE_ORDER].item() == pytest.approx(LO)
    assert knots[SPLINE_ORDER + GRID_SIZE].item() == pytest.approx(HI)


def test_basis_count_is_grid_plus_order(knots: torch.Tensor) -> None:
    values = bspline_basis(sample_domain(), knots, SPLINE_ORDER)
    assert values.shape == (257, GRID_SIZE + SPLINE_ORDER)


def test_basis_is_non_negative(knots: torch.Tensor) -> None:
    values = bspline_basis(sample_domain(), knots, SPLINE_ORDER)
    assert (values >= 0).all()


def test_partition_of_unity(knots: torch.Tensor) -> None:
    """Сумма базисных функций равна единице на всём рабочем отрезке."""
    values = bspline_basis(sample_domain(), knots, SPLINE_ORDER)
    total = values.sum(dim=-1)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-12)


def test_local_support(knots: torch.Tensor) -> None:
    """B_i отлична от нуля только на (knots[i], knots[i + k + 1])."""
    x = sample_domain(1024)
    values = bspline_basis(x, knots, SPLINE_ORDER)

    for i in range(values.shape[-1]):
        outside = (x <= knots[i]) | (x >= knots[i + SPLINE_ORDER + 1])
        assert torch.allclose(
            values[outside, i], torch.zeros(int(outside.sum()), dtype=values.dtype)
        )


def test_greville_coefficients_reproduce_identity(knots: torch.Tensor) -> None:
    """Ключевое свойство: с абсциссами Гревилля сплайн ТОЧНО равен x.

    На этом стоит вся конструкция KAN-LoRA — адаптер стартует как LoRA не
    приближённо, а с точностью до численной ошибки.
    """
    x = sample_domain()
    values = bspline_basis(x, knots, SPLINE_ORDER)
    coefficients = greville_abscissae(knots, SPLINE_ORDER)

    assert coefficients.shape == (GRID_SIZE + SPLINE_ORDER,)
    assert torch.allclose(values @ coefficients, x, atol=1e-12)


def test_linear_order_matches_hat_functions() -> None:
    """Независимая проверка: при k = 1 базис — это шляпные функции.

    Замкнутая форма выписана вручную, без обращения к реализации, поэтому
    ошибка в рекурсии Кокса — де Бура здесь всплывёт.
    """
    order, grid = 1, 4
    knots = make_knots(grid, order, LO, HI, dtype=torch.float64)
    x = sample_domain(101)
    values = bspline_basis(x, knots, order)

    for i in range(values.shape[-1]):
        left, peak, right = knots[i], knots[i + 1], knots[i + 2]
        rising = (x - left) / (peak - left)
        falling = (right - x) / (right - peak)
        expected = torch.where(
            (x >= left) & (x < peak),
            rising,
            torch.where((x >= peak) & (x < right), falling, torch.zeros_like(x)),
        )
        assert torch.allclose(values[:, i], expected, atol=1e-12)


def test_basis_is_differentiable_in_x(knots: torch.Tensor) -> None:
    x = torch.tensor([-0.7, 0.0, 0.31], dtype=torch.float64, requires_grad=True)
    bspline_basis(x, knots, SPLINE_ORDER).sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_supports_batched_input(knots: torch.Tensor) -> None:
    x = torch.rand(3, 4, 5, dtype=torch.float64) * (HI - LO) + LO
    values = bspline_basis(x, knots, SPLINE_ORDER)
    assert values.shape == (3, 4, 5, GRID_SIZE + SPLINE_ORDER)
