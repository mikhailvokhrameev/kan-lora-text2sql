"""Сборка таблиц и графиков из карточек результатов.

Таблицы собираются одной командой и руками не правятся: правка руками
означала бы, что число в работе не восстанавливается из карточек.

Рядом со средним всегда стоит размах по зёрнам. Различие между методами по
Exact Match ожидается порядка одного-двух процентов, то есть внутри разброса
по случайному зерну, и выводы формулируются только по различиям, выходящим
за него. Таблица с одним средним скрывала бы ровно это.

Запуск: python scripts/make_tables.py [--results results] [--output results]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METRICS = ("exact_match", "execution_accuracy")


def load_cards(directory: Path) -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(Path(directory).glob("*.json"))
    ]


CellKey = tuple[str, str, str, bool]

# Значения конфигурации по умолчанию (configs/base.yaml): ими читаются старые карточки,
# в которых этих полей ещё не было.
DEFAULT_OPTIMIZER = "adamw"
DEFAULT_LEARN_INPUT_SCALE = True


def cell_key(card: dict) -> CellKey:
    """Ключ агрегации: всё, что отличает конфигурацию, кроме зерна.

    Прогоны с Muon и абляция без обучаемого масштаба входа — другие эксперименты,
    их нельзя усреднять с основным. learn_input_scale различает только KAN-LoRA
    (как и run_id): у LoRA/DoRA поле не используется и ключ не расщепляет.
    """
    config = card.get("config") or {}
    optimizer = (
        card.get("optimizer") or (config.get("optimizer") or {}).get("name") or DEFAULT_OPTIMIZER
    )
    learn_input_scale = (config.get("adapter") or {}).get(
        "learn_input_scale", DEFAULT_LEARN_INPUT_SCALE
    )
    if card["method"] != "kan_lora":
        learn_input_scale = DEFAULT_LEARN_INPUT_SCALE
    return (card["method"], card["dataset"], optimizer, bool(learn_input_scale))


def aggregate(cards: list[dict]) -> dict[CellKey, dict]:
    """Среднее и размах по зёрнам для каждой ячейки «метод, набор, оптимизатор, масштаб входа».

    Размах, а не стандартное отклонение: при трёх зёрнах отклонение оценивается
    плохо, а размах говорит ровно то, что нужно, — насколько далеко разъезжаются
    прогоны одного метода.
    """
    buckets: dict[CellKey, dict[str, list[float | None]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for card in cards:
        key = cell_key(card)
        for metric in METRICS:
            buckets[key][metric].append(card["metrics"].get(metric))
        buckets[key]["peak_memory_bytes"].append(card["peak_memory_bytes"])
        buckets[key]["trainable_parameters"].append(card["trainable_parameters"])

    aggregated: dict[CellKey, dict] = {}
    for key, metrics in buckets.items():
        summary: dict[str, tuple[float, float] | None] = {}
        for name, values in metrics.items():
            present = [value for value in values if value is not None]
            # «Не мерили» не усредняется с нулём: PAUQ может остаться без баз.
            summary[name] = (
                (sum(present) / len(present), max(present) - min(present)) if present else None
            )
        aggregated[key] = summary
    return aggregated


def _cell(value: tuple[float, float] | None, digits: int = 4) -> str:
    if value is None:
        return "не измерялась"
    mean, spread = value
    return f"{mean:.{digits}f} ± {spread / 2:.{digits}f}"


def main_table(cards: list[dict]) -> str:
    aggregated = aggregate(cards)
    lines = [
        "| Метод | Оптимизатор | Набор | Exact Match | Execution Accuracy | Пик видеопамяти, ГиБ | Обучаемых параметров |",
        "|---|---|---|---|---|---|---|",
        "| без адаптера (базовый уровень) | — | — | заполнить прогоном базовой модели | | | 0 |",
    ]
    for (method, dataset, optimizer, learn_input_scale), summary in sorted(aggregated.items()):
        label = method if learn_input_scale else f"{method} (без масштаба входа)"
        memory = summary["peak_memory_bytes"]
        parameters = summary["trainable_parameters"]
        lines.append(
            f"| {label} | {optimizer} | {dataset} "
            f"| {_cell(summary['exact_match'])} | {_cell(summary['execution_accuracy'])} "
            f"| {memory[0] / 2**30:.2f} | {int(parameters[0])} |"
        )
    return "\n".join(lines)


def nonlinearity_table(cards: list[dict]) -> str:
    lines = [
        "| Прогон | Средний индекс нелинейности | Максимальный | Доля активаций в сетке |",
        "|---|---|---|---|",
    ]
    for card in sorted(cards, key=lambda item: item["run_id"]):
        stats = card.get("spline_stats")
        if not stats:
            continue
        # NaN у адаптера означает, что все его активации вне сетки и мерить нечего
        # (docs/nonlinearity.md). Такие адаптеры пропускаются: иначе NaN заражал бы
        # среднее, а max() с NaN зависит от порядка элементов.
        means = [v["mean"] for v in stats.values() if not math.isnan(v["mean"])]
        maxima = [v["max"] for v in stats.values() if not math.isnan(v["max"])]
        mean_cell = f"{sum(means) / len(means):.4f}" if means else "не измерялась"
        max_cell = f"{max(maxima):.4f}" if maxima else "не измерялась"
        fractions = card.get("fraction_inside_grid") or [float("nan")]
        lines.append(f"| {card['run_id']} | {mean_cell} | {max_cell} | {fractions[-1]:.3f} |")
    return "\n".join(lines)


def optimizer_table(cards: list[dict]) -> str:
    """Покрытие параметров оптимизатором Muon — самостоятельный результат работы."""
    lines = ["| Прогон | Параметров в Muon | Параметров в AdamW | Доля в Muon |", "|---|---|---|---|"]
    for card in sorted(cards, key=lambda item: item["run_id"]):
        coverage = card.get("optimizer_coverage") or {}
        muon, adamw = coverage.get("muon", 0), coverage.get("adamw", 0)
        total = muon + adamw
        if total:
            lines.append(f"| {card['run_id']} | {muon} | {adamw} | {muon / total:.3f} |")
        else:
            lines.append(f"| {card['run_id']} | 0 | 0 | — |")
    return "\n".join(lines)


def plot_fraction_inside_grid(cards: list[dict], path: Path) -> None:
    figure, axes = plt.subplots(figsize=(7, 4))
    for card in sorted(cards, key=lambda item: item["run_id"]):
        values = card.get("fraction_inside_grid")
        if values:
            axes.plot(range(1, len(values) + 1), values, marker="o", label=card["run_id"])

    axes.set_xlabel("эпоха")
    axes.set_ylabel("доля активаций внутри сетки")
    axes.set_title("Вырождается ли сплайновый базис в ходе обучения")
    axes.set_ylim(0.0, 1.05)
    axes.grid(True, alpha=0.3)
    if axes.get_legend_handles_labels()[0]:
        axes.legend(fontsize="small")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description="Сборка таблиц и графиков из карточек")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    arguments = parser.parse_args()

    cards = load_cards(arguments.results)
    if not cards:
        print(f"карточек не найдено в {arguments.results}")
        return 1

    tables = arguments.output / "tables.md"
    tables.parent.mkdir(parents=True, exist_ok=True)
    tables.write_text(
        "# Результаты\n\n## Основная матрица\n\n"
        + main_table(cards)
        + "\n\n## Выученная нелинейность\n\n"
        + nonlinearity_table(cards)
        + "\n\n## Покрытие параметров оптимизатором Muon\n\n"
        + optimizer_table(cards)
        + "\n",
        encoding="utf-8",
    )
    plot_fraction_inside_grid(cards, arguments.output / "figures" / "fraction_inside_grid.png")

    print(f"собрано карточек: {len(cards)}; таблицы: {tables}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
