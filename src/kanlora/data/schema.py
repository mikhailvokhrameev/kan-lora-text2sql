"""Схема базы данных и её текстовое представление для промпта.

Формат зафиксирован и одинаков для Spider и PAUQ, для всех трёх методов и для
всех зёрен. Столбец `*` относится ко всей базе, а не к таблице, и отбрасывается.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = ["DatabaseSchema", "Table", "load_schemas", "serialize_schema"]


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[str, ...]


@dataclass(frozen=True)
class DatabaseSchema:
    db_id: str
    tables: tuple[Table, ...]


def load_schemas(tables_json: Path) -> dict[str, DatabaseSchema]:
    """Читает tables.json формата Spider."""
    entries = json.loads(Path(tables_json).read_text(encoding="utf-8"))

    schemas: dict[str, DatabaseSchema] = {}
    for entry in entries:
        names = entry["table_names_original"]
        columns: list[list[str]] = [[] for _ in names]
        for table_index, column_name in entry["column_names_original"]:
            if table_index >= 0:
                columns[table_index].append(column_name)

        schemas[entry["db_id"]] = DatabaseSchema(
            db_id=entry["db_id"],
            tables=tuple(
                Table(name, tuple(table_columns))
                for name, table_columns in zip(names, columns, strict=True)
            ),
        )
    return schemas


def serialize_schema(schema: DatabaseSchema) -> str:
    """Одна строка вида `table: col, col | table: col`."""
    return " | ".join(
        f"{table.name}: {', '.join(table.columns)}" for table in schema.tables
    )
