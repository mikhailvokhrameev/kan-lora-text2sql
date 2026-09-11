"""Обёртка над официальной метрикой Spider.

Exact Match в Spider — покомпонентное сравнение разобранного запроса:
запрос разбирается на части (SELECT, WHERE, GROUP BY и так далее), и части
сравниваются как множества. Поэтому регистр, лишние пробелы и порядок условий
в WHERE не влияют, а перестановка столбцов в SELECT — влияет.

Метрика считается вендоренным официальным кодом, а не своей реализацией:
собственная дала бы числа, несравнимые с опубликованными. Схема каждой базы
строится из `tables.json` (а не из файла `.sqlite`), потому что интерфейс этой
обёртки берёт только `tables.json`: расположение файлов баз данных на разных
машинах может отличаться, а `tables.json` лежит в git вместе с наборами.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

__all__ = ["exact_match", "exact_match_by_hardness"]

_SPIDER_EVAL = Path(__file__).resolve().parents[3] / "third_party" / "spider_eval"
if str(_SPIDER_EVAL) not in sys.path:
    # evaluation.py делает `from process_sql import ...` — плоский импорт
    # соседнего файла, как в скрипте командной строки. Файл вендорится без
    # изменений, поэтому каталог с ним добавляется в sys.path напрямую,
    # а не импортируется как пакет `spider_eval.*`.
    sys.path.insert(0, str(_SPIDER_EVAL))

HARDNESS_LEVELS = ("easy", "medium", "hard", "extra", "all")


def _schema_dict_from_entry(entry: dict) -> dict[str, list[str]]:
    """Строит `{таблица: [столбцы]}` из записи `tables.json`.

    Повторяет формат официальной `process_sql.get_schema`, которая читает ту
    же информацию напрямую из `.sqlite`-файла: имена в нижнем регистре,
    столбец `*` (индекс таблицы -1) отбрасывается.
    """
    table_names = entry["table_names_original"]
    schema: dict[str, list[str]] = {name.lower(): [] for name in table_names}
    for table_index, column_name in entry["column_names_original"]:
        if table_index >= 0:
            schema[table_names[table_index].lower()].append(column_name.lower())
    return schema


def _load_schemas(tables_json: Path) -> dict[str, dict[str, list[str]]]:
    entries = json.loads(Path(tables_json).read_text(encoding="utf-8"))
    return {entry["db_id"]: _schema_dict_from_entry(entry) for entry in entries}


def exact_match_by_hardness(
    gold: list[str],
    predicted: list[str],
    db_ids: list[str],
    tables_json: Path,
) -> dict[str, float]:
    """Доля точных совпадений, всего и в разбивке по сложности запроса."""
    if not (len(gold) == len(predicted) == len(db_ids)):
        raise ValueError(
            f"длины не совпадают: эталонов {len(gold)}, предсказаний {len(predicted)}, "
            f"баз {len(db_ids)}"
        )

    from evaluation import (
        Evaluator,
        build_foreign_key_map_from_json,
        build_valid_col_units,
        rebuild_sql_col,
        rebuild_sql_val,
    )
    from process_sql import Schema, get_sql

    evaluator = Evaluator()
    foreign_keys = build_foreign_key_map_from_json(str(tables_json))
    schemas = _load_schemas(tables_json)

    scores = {level: [0, 0] for level in HARDNESS_LEVELS}  # [совпало, всего]

    for gold_query, predicted_query, db_id in zip(gold, predicted, db_ids, strict=True):
        schema = Schema(schemas[db_id])
        kmap = foreign_keys[db_id]

        gold_sql = get_sql(schema, gold_query)
        hardness = evaluator.eval_hardness(gold_sql)
        gold_valid_col_units = build_valid_col_units(gold_sql["from"]["table_units"], schema)
        gold_sql = rebuild_sql_val(gold_sql)
        gold_sql = rebuild_sql_col(gold_valid_col_units, gold_sql, kmap)

        try:
            predicted_sql = get_sql(schema, predicted_query)
            predicted_valid_col_units = build_valid_col_units(
                predicted_sql["from"]["table_units"], schema
            )
            predicted_sql = rebuild_sql_val(predicted_sql)
            predicted_sql = rebuild_sql_col(predicted_valid_col_units, predicted_sql, kmap)
            matched = bool(evaluator.eval_exact_match(predicted_sql, gold_sql))
        except Exception:
            # Модель регулярно выдаёт неразбираемый текст. Это промах,
            # а не повод уронить оценку всего прогона.
            matched = False

        for level in (hardness, "all"):
            scores[level][1] += 1
            scores[level][0] += int(matched)

    return {
        level: (matched / total if total else 0.0) for level, (matched, total) in scores.items()
    }


def exact_match(
    gold: list[str], predicted: list[str], db_ids: list[str], tables_json: Path
) -> float:
    return exact_match_by_hardness(gold, predicted, db_ids, tables_json)["all"]
