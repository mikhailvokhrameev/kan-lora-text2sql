"""Мера выученной нелинейности.

Отвечает на главный вопрос работы: используется ли нелинейность в адаптере
на самом деле. К каждой выученной одномерной функции подгоняется прямая
методом наименьших квадратов на том отрезке, куда реально попадают активации
и где при этом определён сплайн, и берётся доля дисперсии, которую прямая не
объясняет:

    index = RSS / TSS,   RSS — остаток после подгонки, TSS — полная дисперсия.

Ноль означает, что функция — прямая, единица — что прямая не объясняет
ничего. Если индекс близок к нулю по всем слоям, нелинейность не используется
и KAN-LoRA сводится к LoRA с накладными расходами. Это доказанный
отрицательный результат, а не неудача, и на защите он весит больше, чем
прибавка в доли процента к Exact Match.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from kanlora.adapters.inject import adapter_modules
from kanlora.adapters.kan_layer import KANLayer
from kanlora.adapters.kan_lora import KANLoRALinear

__all__ = [
    "EdgeStats",
    "layer_nonlinearity",
    "measurement_interval",
    "model_nonlinearity",
    "nonlinearity_index",
]


@dataclass(frozen=True)
class EdgeStats:
    source: int
    target: int
    nonlinearity: float


def nonlinearity_index(inputs: torch.Tensor, outputs: torch.Tensor) -> float:
    """Доля дисперсии функции, не объяснённая наилучшей прямой."""
    x = inputs.to(torch.float64).flatten()
    y = outputs.to(torch.float64).flatten()

    total = ((y - y.mean()) ** 2).sum()
    if total.item() <= 1e-18:
        # Постоянная функция — частный случай прямой, а не деление на ноль.
        return 0.0

    design = torch.stack([x, torch.ones_like(x)], dim=1)
    coefficients = torch.linalg.lstsq(design, y.unsqueeze(1)).solution
    residual = ((y - (design @ coefficients).squeeze(1)) ** 2).sum()
    return float((residual / total).clamp(min=0.0, max=1.0))


# Запас от правого края сетки: базис полуоткрыт, `x == hi` уже даёт ноль.
_GRID_MARGIN = 0.99


def measurement_interval(
    layer: KANLayer, source: int, lo: float, hi: float
) -> tuple[float, float] | None:
    """Отрезок измерения для входа `source`: [lo, hi] внутри области определения сетки.

    Слой считает `out = s * phi(x / s)`, сетка лежит в координате `u = x / s`,
    то есть в координатах входа это отрезок `grid_range * s` (для отрицательного
    `s` границы меняются местами). За ним сплайн равен нулю, и прямая,
    подогнанная по такому участку, оценивает не выученную функцию, а границу
    области определения. Поэтому отрезок обрезается по сетке. Возвращает `None`,
    если пересечение пусто или вырождено в точку: мерить нечего.
    """
    grid_lo, grid_hi = layer.grid_range
    scale = float(layer._safe_input_scale_divisor()[source])
    edges = sorted((_GRID_MARGIN * grid_lo * scale, _GRID_MARGIN * grid_hi * scale))
    start, stop = max(lo, edges[0]), min(hi, edges[1])
    return (start, stop) if start < stop else None


def layer_nonlinearity(
    layer: KANLayer, lo: float, hi: float, samples: int = 257
) -> list[EdgeStats]:
    """Индекс нелинейности каждого ребра слоя на отрезке [lo, hi] внутри сетки.

    Отрезок задаётся снаружи: мерить надо там, куда попадают активации, а не на
    всей области определения сетки. Но и за пределами сетки мерить нельзя: там
    сплайн равен нулю, и даже тождественный слой получил бы высокий индекс.
    Поэтому для каждого входа `source` отрезок обрезается по области определения
    сетки в координатах входа (`measurement_interval`; масштаб `input_scale`
    у каждого входа свой). Допускаются `lo = -inf`, `hi = inf` — тогда мерится
    вся область определения. Рёбра входа, для которого пересечение пусто,
    в результат не попадают.
    """
    device = layer.spline_coefficients.device
    dtype = layer.spline_coefficients.dtype

    stats: list[EdgeStats] = []
    with torch.no_grad():
        for source in range(layer.in_features):
            interval = measurement_interval(layer, source, lo, hi)
            if interval is None:
                continue

            grid = torch.linspace(*interval, samples, dtype=torch.float64, device=device)
            probe = torch.zeros(samples, layer.in_features, dtype=dtype, device=device)
            probe[:, source] = grid.to(dtype)
            response = layer(probe)
            for target in range(layer.out_features):
                stats.append(
                    EdgeStats(source, target, nonlinearity_index(grid, response[:, target]))
                )
    return stats


def model_nonlinearity(
    model: torch.nn.Module, ranges: dict[str, tuple[float, float]] | None = None
) -> dict[str, dict[str, float]]:
    """Сводка по всем KAN-адаптерам модели. Для LoRA и DoRA возвращает пустой словарь.

    Отрезок измерения для каждого адаптера берётся из `ranges`, если он задан;
    иначе — фактический размах, накопленный адаптером за прошедшие проходы
    (`observed_range`), а если проходов ещё не было — вся область определения
    сетки. В любом случае отрезок обрезается по сетке (`layer_nonlinearity`):
    за ней сплайн равен нулю и измерять там нечего. Если для адаптера пересечение
    пусто, `mean` и `max` равны `NaN`; долю входов вне сетки показывает
    `fraction_inside_grid`.
    """
    summary: dict[str, dict[str, float]] = {}

    for name, module in adapter_modules(model):
        if not isinstance(module, KANLoRALinear):
            continue

        if ranges is not None and name in ranges:
            lo, hi = ranges[name]
        else:
            lo, hi = module.observed_range()
            if lo > hi:
                lo, hi = float("-inf"), float("inf")

        values = [item.nonlinearity for item in layer_nonlinearity(module.kan, lo, hi)]
        fraction = module.mean_fraction_inside_grid()
        summary[name] = {
            "mean": sum(values) / len(values) if values else float("nan"),
            "max": max(values) if values else float("nan"),
            "fraction_inside_grid": float("nan") if fraction is None else fraction,
        }
    return summary
