"""Раскладка файлов PAUQ, разбиение pauq_xsp.

Берётся одно разбиение — по базам данных, как в Spider. Второе разбиение
не используется, чтобы не плодить вариантов.

Ожидается:
    data/pauq/pauq_xsp_train.json
    data/pauq/pauq_xsp_test.json
    data/pauq/tables.json
    data/pauq/database/<db_id>/<db_id>.sqlite

Если фактические имена файлов в выкачанном PAUQ отличаются, поправить их
здесь и синхронно в DATASET_FILES в scripts/check_environment.py.
"""

from __future__ import annotations

from kanlora.data.spider import DatasetLayout

__all__ = ["PAUQ_LAYOUT"]

PAUQ_LAYOUT = DatasetLayout(
    train_file="pauq_xsp_train.json",
    eval_file="pauq_xsp_test.json",
    tables_file="tables.json",
    database_dir="database",
)
