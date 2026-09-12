"""Исполнение SQL на sqlite и сравнение результатов.

Сравниваются результаты, а не тексты: запрос, написанный иначе, но дающий тот
же ответ, засчитывается. Порядок строк учитывается только при наличии ORDER BY
в эталонном запросе — без него порядок в SQL не определён, и требовать его
означало бы занижать метрику по случайному признаку.

Любая ошибка исполнения считается промахом, а не аварией: модель регулярно
выдаёт неисполнимый текст, и это нормальная часть измерения.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from kanlora.data.loaders import DatasetLayout, database_path

__all__ = [
    "ExecutionOutcome",
    "count_available_databases",
    "execute_query",
    "execution_accuracy",
    "results_match",
]


@dataclass(frozen=True)
class ExecutionOutcome:
    rows: list[tuple] | None
    failed: bool


def execute_query(db_path: Path, query: str, timeout: float = 30.0) -> ExecutionOutcome:
    if not Path(db_path).is_file():
        return ExecutionOutcome(None, True)

    connection = None
    try:
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout)
        connection.text_factory = lambda value: value.decode("utf-8", errors="replace")
        return ExecutionOutcome(connection.execute(query).fetchall(), False)
    except Exception:
        return ExecutionOutcome(None, True)
    finally:
        if connection is not None:
            connection.close()


def count_available_databases(
    root: Path, layout: DatasetLayout, db_ids: Iterable[str]
) -> int:
    """Считает, для скольких уникальных `db_id` на диске нашёлся файл базы.

    Используется вместо проверки одного произвольного `db_id`
    (`database_path(root, layout, db_ids[0]).is_file()`): у Spider/PAUQ
    отложенная выборка ссылается на много разных баз, и наличие файла для
    первого примера ничего не говорит об остальных.
    """
    return sum(1 for db_id in set(db_ids) if database_path(root, layout, db_id).is_file())


def results_match(
    gold: ExecutionOutcome, predicted: ExecutionOutcome, order_matters: bool
) -> bool:
    if gold.failed or predicted.failed:
        return False
    if order_matters:
        return gold.rows == predicted.rows
    return sorted(map(repr, gold.rows)) == sorted(map(repr, predicted.rows))


def execution_accuracy(
    gold: list[str],
    predicted: list[str],
    db_ids: list[str],
    root: Path,
    layout: DatasetLayout,
) -> float:
    if not (len(gold) == len(predicted) == len(db_ids)):
        raise ValueError(
            f"длины не совпадают: эталонов {len(gold)}, предсказаний {len(predicted)}, "
            f"баз {len(db_ids)}"
        )
    if not gold:
        return 0.0

    matched = 0
    for gold_query, predicted_query, db_id in zip(gold, predicted, db_ids, strict=True):
        database = database_path(root, layout, db_id)
        order_matters = "order by" in gold_query.lower()
        matched += int(
            results_match(
                execute_query(database, gold_query),
                execute_query(database, predicted_query),
                order_matters,
            )
        )
    return matched / len(gold)
