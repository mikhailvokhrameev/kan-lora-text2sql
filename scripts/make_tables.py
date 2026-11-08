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
from collections import defaultdict
from pathlib import Path

METRICS = ("exact_match", "execution_accuracy")


def load_cards(directory: Path) -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted([*Path(directory).glob("*.json"),
                            *Path(directory).glob("*/*/card.json")])
    ]


CellKey = tuple[str, str]

# Базовая модель без адаптера (`method: none`): обучения не было, оптимизатора
# и масштаба входа у неё нет.
BASELINE = "none"
BASELINE_LABEL = "без адаптера (базовый уровень)"

def cell_key(card: dict) -> CellKey:
    """Группирует карточки по методу и набору данных."""
    return card["method"], card["dataset"]


def _flatten(section: dict, prefix: str = "") -> dict[str, object]:
    """Вложенная конфигурация в плоский словарь с путями вида `train.epochs`."""
    flat: dict[str, object] = {}
    for name, value in section.items():
        path = f"{prefix}{name}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{path}."))
        else:
            flat[path] = value
    return flat


def _comparable_config(card: dict) -> dict[str, object] | None:
    """Конфигурация карточки без зерна и без полей, которые её ячейку не различают.

    Масштаб входа у LoRA/DoRA не используется, у базовой модели нет ни
    оптимизатора, ни адаптера. Карточка без конфигурации сравнению не подлежит.
    """
    config = card.get("config")
    if not config:
        return None
    flat = _flatten(config)
    flat.pop("seed", None)
    if card["method"] != "kan_lora":
        flat.pop("adapter.learn_input_scale", None)
    if card["method"] == BASELINE:
        flat = {name: value for name, value in flat.items() if not name.startswith(("adapter.", "optimizer."))}
    return flat


def _check_comparable(key: CellKey, cards: list[dict]) -> None:
    """Бросает ValueError, если карточки одной ячейки получены в разных условиях.

    Имена файлов смок-прогона и полного прогона совпадают, и без этой проверки
    ячейка молча усреднила бы 50 примеров смока с полной отложенной выборкой.
    Различаться могут только зёрна.
    """
    reference_card = None
    reference_config = None
    for card in cards:
        config = _comparable_config(card)
        if config is None:
            continue
        if reference_config is None:
            reference_card, reference_config = card, config
            continue
        differing = sorted(
            name
            for name in reference_config.keys() | config.keys()
            if reference_config.get(name) != config.get(name)
        )
        if differing:
            raise ValueError(
                f"ячейка {key}: карточки {reference_card['run_id']} и {card['run_id']} "
                f"различаются полями конфигурации {', '.join(differing)}; "
                "их нельзя усреднять, уберите лишние карточки из каталога результатов"
            )


def aggregate(cards: list[dict]) -> dict[CellKey, dict]:
    """Среднее и размах по зёрнам для каждой ячейки «метод, набор».

    Размах, а не стандартное отклонение: при трёх зёрнах отклонение оценивается
    плохо, а размах говорит ровно то, что нужно, — насколько далеко разъезжаются
    прогоны одного метода.

    Карточки одной ячейки обязаны различаться только зерном (`_check_comparable`).
    """
    buckets: dict[CellKey, dict[str, list[float | None]]] = defaultdict(
        lambda: defaultdict(list)
    )
    members: dict[CellKey, list[dict]] = defaultdict(list)
    for card in cards:
        config = card.get("config") or {}
        optimizer = card.get("optimizer") or (config.get("optimizer") or {}).get("name")
        if card["method"] != BASELINE and optimizer not in (None, "adamw"):
            raise ValueError(f"неподдерживаемый оптимизатор {optimizer} в {card['run_id']}")
        if card["method"] == "kan_lora" and (config.get("adapter") or {}).get("learn_input_scale") is False:
            raise ValueError(f"неподдерживаемая конфигурация learn_input_scale в {card['run_id']}")
        key = cell_key(card)
        members[key].append(card)
        for metric in METRICS:
            buckets[key][metric].append(card["metrics"].get(metric))
        buckets[key]["peak_memory_bytes"].append(card.get("peak_memory_bytes"))
        buckets[key]["trainable_parameters"].append(card.get("trainable_parameters"))

    aggregated: dict[CellKey, dict] = {}
    for key, metrics in buckets.items():
        _check_comparable(key, members[key])
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


def _row_order(key: CellKey) -> tuple:
    """Базовый уровень — первыми строками, затем методы в прежнем порядке."""
    method, dataset = key
    return (method != BASELINE, method, dataset)


def main_table(cards: list[dict]) -> str:
    """Основная матрица; строки базового уровня собираются из карточек `method: none`.

    Пока такой карточки нет, на месте базового уровня стоит строка «нет прогона»:
    без этой точки таблица не говорит, дало ли дообучение что-то вообще.
    """
    aggregated = aggregate(cards)
    lines = [
        "| Метод | Оптимизатор | Набор | Exact Match | Execution Accuracy | Пик видеопамяти, ГиБ | Обучаемых параметров |",
        "|---|---|---|---|---|---|---|",
    ]
    if not any(key[0] == BASELINE for key in aggregated):
        lines.append(f"| {BASELINE_LABEL} | — | — | нет прогона | нет прогона | — | — |")
    for key in sorted(aggregated, key=_row_order):
        method, dataset = key
        summary = aggregated[key]
        if method == BASELINE:
            label = BASELINE_LABEL
        else:
            label = method
        memory = summary["peak_memory_bytes"]
        parameters = summary["trainable_parameters"]
        # У базового уровня обучения не было: пика видеопамяти обучения и
        # обучаемых параметров нет (null в карточке), в таблице — прочерк.
        memory_cell = f"{memory[0] / 2**30:.2f}" if memory is not None else "—"
        parameters_cell = f"{int(parameters[0])}" if parameters is not None else "—"
        lines.append(
            f"| {label} | {'—' if method == BASELINE else 'adamw'} | {dataset} "
            f"| {_cell(summary['exact_match'])} | {_cell(summary['execution_accuracy'])} "
            f"| {memory_cell} | {parameters_cell} |"
        )
    return "\n".join(lines)


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
        + "\n",
        encoding="utf-8",
    )

    print(f"собрано карточек: {len(cards)}; таблицы: {tables}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
