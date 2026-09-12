"""Мера выученной нелинейности.

Отвечает на главный вопрос работы: используется ли нелинейность в адаптере
на самом деле. К каждой выученной одномерной функции подгоняется прямая
методом наименьших квадратов на том отрезке, куда реально попадают активации,
и берётся доля дисперсии, которую прямая не объясняет:

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

__all__ = ["EdgeStats", "layer_nonlinearity", "model_nonlinearity", "nonlinearity_index"]


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


def layer_nonlinearity(
    layer: KANLayer, lo: float, hi: float, samples: int = 257
) -> list[EdgeStats]:
    """Индекс нелинейности каждого ребра слоя на отрезке [lo, hi].

    Отрезок задаётся снаружи: мерить надо там, куда попадают активации,
    а не на всей области определения сетки.
    """
    device = layer.spline_coefficients.device
    grid = torch.linspace(lo, hi, samples, dtype=torch.float64, device=device)

    stats: list[EdgeStats] = []
    with torch.no_grad():
        for source in range(layer.in_features):
            probe = torch.zeros(samples, layer.in_features, dtype=layer.spline_coefficients.dtype,
                                device=device)
            probe[:, source] = grid.to(probe.dtype)
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
    (`observed_range`), а если проходов ещё не было — вся сетка сплайна.
    Подгонять прямую нужно там, куда реально попадают активации, а не по всей
    области определения: иначе индекс завышается на неиспользуемой части сетки.
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
                grid_lo, grid_hi = module.kan.grid_range
                lo, hi = 0.99 * grid_lo, 0.99 * grid_hi

        values = [item.nonlinearity for item in layer_nonlinearity(module.kan, lo, hi)]
        fraction = module.last_fraction_inside_grid()
        summary[name] = {
            "mean": sum(values) / len(values),
            "max": max(values),
            "fraction_inside_grid": float("nan") if fraction is None else fraction,
        }
    return summary
