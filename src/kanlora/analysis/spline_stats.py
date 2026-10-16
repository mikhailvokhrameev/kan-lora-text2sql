"""Мера выученной нелинейности.

Отвечает на главный вопрос работы: используется ли нелинейность в адаптере
на самом деле. К каждой выученной одномерной функции подгоняется прямая
методом наименьших квадратов на том отрезке, куда реально попадают активации
и где при этом определён сплайн, и берётся доля дисперсии, которую прямая не
объясняет:

    index = RSS / TSS,   RSS — остаток после подгонки, TSS — полная дисперсия.

Это индекс ребра. Индекс слоя берёт те же суммы по всем его рёбрам сразу:
ΣRSS / ΣTSS — доля суммарной дисперсии рёбер, которую не объясняют прямые.
Рёбра с ничтожной амплитудой в нём почти ничего не весят.

Ноль означает, что функция — прямая, единица — что прямая не объясняет
ничего. Если индекс близок к нулю по всем слоям, нелинейность не используется
и KAN-LoRA сводится к LoRA с накладными расходами. Это доказанный
отрицательный результат, а не неудача, и на защите он весит больше, чем
прибавка в доли процента к Exact Match.
"""

from __future__ import annotations

from collections.abc import Sequence
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
    "pooled_nonlinearity",
]


@dataclass(frozen=True)
class EdgeStats:
    source: int
    target: int
    nonlinearity: float
    # Суммы квадратов на сетке замера: остаток после подгонки прямой (RSS) и
    # отклонение от среднего (TSS). Нужны индексу слоя (pooled_nonlinearity).
    residual: float
    total: float


def _line_fit_sums(inputs: torch.Tensor, outputs: torch.Tensor) -> tuple[float, float]:
    """(RSS, TSS) наилучшей прямой; у постоянной функции RSS = 0."""
    x = inputs.to(torch.float64).flatten()
    y = outputs.to(torch.float64).flatten()

    total = ((y - y.mean()) ** 2).sum()
    if total.item() <= 1e-18:
        # Постоянная функция — частный случай прямой, а не деление на ноль.
        return 0.0, float(total)

    design = torch.stack([x, torch.ones_like(x)], dim=1)
    coefficients = torch.linalg.lstsq(design, y.unsqueeze(1)).solution
    residual = ((y - (design @ coefficients).squeeze(1)) ** 2).sum()
    return float(residual.clamp(min=0.0, max=float(total))), float(total)


def nonlinearity_index(inputs: torch.Tensor, outputs: torch.Tensor) -> float:
    """Доля дисперсии функции, не объяснённая наилучшей прямой."""
    residual, total = _line_fit_sums(inputs, outputs)
    return 0.0 if total <= 1e-18 else residual / total


def pooled_nonlinearity(stats: Sequence[EdgeStats]) -> float:
    """Индекс слоя: доля суммарной дисперсии его рёбер, не объяснённая прямыми.

    ΣRSS / ΣTSS — та же формула, что у ребра, но суммы берутся по всем рёбрам
    сразу. Ребро вносит вклад пропорционально дисперсии своего выхода, поэтому
    почти нулевые рёбра (недиагональные стартуют с нуля) не решают за слой, а
    общий масштаб слоя на индекс не влияет. NaN — рёбер нет (все вне сетки).
    """
    if not stats:
        return float("nan")
    total = sum(item.total for item in stats)
    if total <= 1e-18:
        return 0.0
    return sum(item.residual for item in stats) / total


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
    scale = float(layer.input_scale_divisor()[source])
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
                residual, total = _line_fit_sums(grid, response[:, target])
                index = 0.0 if total <= 1e-18 else residual / total
                stats.append(EdgeStats(source, target, index, residual, total))
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
    пусто, `index` равен `NaN`; долю входов вне сетки показывает
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

        stats = layer_nonlinearity(module.kan, lo, hi)
        fraction = module.mean_fraction_inside_grid()
        summary[name] = {
            "index": pooled_nonlinearity(stats),
            "fraction_inside_grid": float("nan") if fraction is None else fraction,
        }
    return summary
