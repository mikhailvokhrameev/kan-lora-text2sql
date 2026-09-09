"""B-сплайновый базис на равномерной сетке.

Чистая математика без обучаемых параметров: узловой вектор, вычисление базиса
по рекурсии Кокса — де Бура и абсциссы Гревилля.
"""

from __future__ import annotations

import torch

__all__ = ["make_knots", "bspline_basis", "greville_abscissae", "num_basis_functions"]


def num_basis_functions(grid_size: int, spline_order: int) -> int:
    return grid_size + spline_order


def make_knots(
    grid_size: int,
    spline_order: int,
    lo: float = -1.0,
    hi: float = 1.0,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Равномерный узловой вектор, продлённый на `spline_order` узлов в обе стороны.

    Продление нужно, чтобы базис был полным на всём отрезке [lo, hi]: без него
    у краёв сумма базисных функций перестаёт равняться единице.
    """
    step = (hi - lo) / grid_size
    indices = torch.arange(
        -spline_order, grid_size + spline_order + 1, dtype=dtype, device=device
    )
    return lo + step * indices


def bspline_basis(x: torch.Tensor, knots: torch.Tensor, spline_order: int) -> torch.Tensor:
    """Значения базисных функций в точках `x`.

    Возвращает тензор формы (*x.shape, len(knots) - 1 - spline_order).
    """
    x = x.unsqueeze(-1)
    knots = knots.to(x.dtype)

    bases = ((x >= knots[:-1]) & (x < knots[1:])).to(x.dtype)

    for order in range(1, spline_order + 1):
        left = (x - knots[: -(order + 1)]) / (knots[order:-1] - knots[: -(order + 1)])
        right = (knots[order + 1 :] - x) / (knots[order + 1 :] - knots[1:-order])
        bases = left * bases[..., :-1] + right * bases[..., 1:]

    return bases


def greville_abscissae(knots: torch.Tensor, spline_order: int) -> torch.Tensor:
    """Абсциссы Гревилля — коэффициенты, при которых сплайн равен тождеству.

    Свойство линейной точности B-сплайнов: сумма greville_i * B_i(x) равна x
    в точности. Именно поэтому KAN-LoRA удаётся инициализировать так, чтобы он
    совпадал с обычной LoRA, а не просто приближал её.
    """
    count = knots.numel() - 1 - spline_order
    offsets = torch.arange(1, spline_order + 1, device=knots.device)
    indices = torch.arange(count, device=knots.device).unsqueeze(1) + offsets.unsqueeze(0)
    return knots[indices].mean(dim=1)
