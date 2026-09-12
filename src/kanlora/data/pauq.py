"""Раскладка файлов PAUQ.

Разбиение по базам данных, как в Spider, — train/dev по `db_id`, без
пересечения баз между выборками. Ожидается раскладка из репозитория
`github.com/ai-spiderweb/pauq` (каталог `dataset/`), с обрезанным префиксом
каталога:
    data/pauq/pauq_train.json
    data/pauq/pauq_dev.json
    data/pauq/tables.json
    data/pauq/database/<db_id>/<db_id>.sqlite

`tables.json` структурно совпадает с форматом Spider (`table_names_original`,
`column_names_original`) — PAUQ не меняет имена таблиц и столбцов, поэтому
`kanlora.data.schema` читает оба набора без различий.

`text_field_language="ru"` — единственное реальное отличие раскладки PAUQ от
Spider, помимо имён файлов: `question` и `query` в PAUQ хранятся не плоской
строкой, а словарём `{"en": ..., "ru": ...}` (перевод и локализация Spider
на русский), и `kanlora.data.loaders.load_examples` использует это поле
раскладки, чтобы выбрать русскую сторону.

Файлы баз данных в самом git-репозитории PAUQ отсутствуют — они лежат
отдельным архивом, на который репозиторий ссылается в README.
"""

from __future__ import annotations

from kanlora.data.spider import DatasetLayout

__all__ = ["PAUQ_LAYOUT"]

PAUQ_LAYOUT = DatasetLayout(
    train_file="pauq_train.json",
    eval_file="pauq_dev.json",
    tables_file="tables.json",
    database_dir="database",
    text_field_language="ru",
)
