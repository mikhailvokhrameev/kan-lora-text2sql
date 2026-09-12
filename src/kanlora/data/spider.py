"""Раскладка файлов Spider.

Ожидается распакованный официальный архив:
    data/spider/train_spider.json
    data/spider/dev.json
    data/spider/tables.json
    data/spider/database/<db_id>/<db_id>.sqlite
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DatasetLayout", "SPIDER_LAYOUT"]


@dataclass(frozen=True)
class DatasetLayout:
    train_file: str
    eval_file: str
    tables_file: str
    database_dir: str
    # None — question/query лежат плоской строкой (Spider). Непустая строка —
    # они лежат словарём {"en": ..., "ru": ...}, и это ключ, который нужно
    # взять (PAUQ, см. kanlora/data/pauq.py).
    text_field_language: str | None = None


SPIDER_LAYOUT = DatasetLayout(
    train_file="train_spider.json",
    eval_file="dev.json",
    tables_file="tables.json",
    database_dir="database",
)
