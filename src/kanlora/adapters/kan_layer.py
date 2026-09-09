"""Слой Колмогорова — Арнольда: обучаемые одномерные функции на рёбрах.

Формулировка сохранена в том виде, в каком она задана в исходной работе о KAN:
каждое ребро (i -> j) несёт функцию

    phi_ji(u) = w_base_ji * silu(u) + w_spline_ji * spline_ji(u),

а выход слоя равен сумме по входным каналам. Добавлен ровно один элемент —
масштаб входа s_i, растягивающий сетку под фактический размах активаций:

    out_j = sum_i s_i * phi_ji(x_i / s_i).

Масштаб введён потому, что B-сплайновый базис определён на конечной сетке, и при
выходе за неё все базисные функции обращаются в ноль: слой молча вырождается,
превращаясь в LoRA с неработающими параметрами. Масштаб входит и выходит
симметрично, поэтому тождественная инициализация от него не страдает.
"""

from __future__ import annotations

import torch
from torch import nn

from kanlora.adapters.spline import bspline_basis, greville_abscissae, make_knots


class KANLayer(nn.Module):
    """Отображение R^in -> R^out, при инициализации тождественное.

    Тождественность обеспечивается свойством линейной точности B-сплайнов:
    с абсциссами Гревилля в качестве коэффициентов сплайн в точности равен
    своему аргументу. Диагональные рёбра получают эти коэффициенты,
    недиагональные — нули, вес базовой ветви обнуляется.

    Тождество действует внутри сетки. За её пределами слой отдаёт ноль, и
    доля таких входов отслеживается методом `fraction_inside_grid`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range

        # Узлы держатся в двойной точности независимо от точности слоя: они
        # задают абсциссы Гревилля, а через них — точность тождественной
        # инициализации. Округление до одинарной точности здесь стоило бы
        # примерно двух разрядов в проверке совпадения KAN-LoRA с LoRA.
        knots = make_knots(grid_size, spline_order, *grid_range, dtype=torch.float64)
        self.register_buffer("knots", knots)
        num_basis = grid_size + spline_order

        self.spline_coefficients = nn.Parameter(
            torch.zeros(out_features, in_features, num_basis, **factory)
        )
        self.spline_scale = nn.Parameter(torch.ones(out_features, in_features, **factory))
        self.base_weight = nn.Parameter(torch.zeros(out_features, in_features, **factory))
        self.input_scale = nn.Parameter(torch.ones(in_features, **factory))

        self.reset_to_identity()

    @torch.no_grad()
    def reset_to_identity(self) -> None:
        self.spline_coefficients.zero_()
        identity = greville_abscissae(self.knots, self.spline_order)
        diagonal = min(self.in_features, self.out_features)
        index = torch.arange(diagonal)
        self.spline_coefficients[index, index] = identity.to(self.spline_coefficients.dtype)

        self.spline_scale.fill_(1.0)
        self.base_weight.zero_()
        self.input_scale.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled = x / self.input_scale
        basis = bspline_basis(scaled, self.knots, self.spline_order)

        spline = torch.einsum("...ib,oib->...oi", basis, self.spline_coefficients)
        base = torch.nn.functional.silu(scaled).unsqueeze(-2) * self.base_weight

        edges = base + self.spline_scale * spline
        return (edges * self.input_scale).sum(dim=-1)

    def fraction_inside_grid(self, x: torch.Tensor) -> torch.Tensor:
        """Доля входов, попавших в область определения сетки.

        Показатель вырождения: чем он ниже, тем большая часть адаптера не
        участвует в вычислении.
        """
        lo, hi = self.grid_range
        scaled = x / self.input_scale
        return ((scaled >= lo) & (scaled <= hi)).to(x.dtype).mean()

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"grid_size={self.grid_size}, spline_order={self.spline_order}"
        )
