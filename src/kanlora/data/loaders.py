"""Чтение наборов Spider и PAUQ.

Оба набора хранятся в одном формате JSON и различаются только именами файлов,
поэтому читатель один. Второй читатель означал бы второе место, где
предобработка способна незаметно разойтись между наборами, а сравнение
Spider с PAUQ строится ровно на том, что предобработка у них одна.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from kanlora.data.pauq import PAUQ_LAYOUT
from kanlora.data.schema import DatabaseSchema, load_schemas
from kanlora.data.spider import SPIDER_LAYOUT, DatasetLayout

__all__ = [
    "LAYOUTS",
    "DatasetLayout",
    "Text2SqlExample",
    "database_path",
    "load_dataset_schemas",
    "load_examples",
    "subsample",
]

Split = Literal["train", "eval"]


@dataclass(frozen=True)
class Text2SqlExample:
    db_id: str
    question: str
    query: str


def _split_file(layout: DatasetLayout, split: Split) -> str:
    return {"train": layout.train_file, "eval": layout.eval_file}[split]


def load_examples(root: Path, layout: DatasetLayout, split: Split) -> list[Text2SqlExample]:
    path = Path(root) / _split_file(layout, split)
    if not path.is_file():
        raise FileNotFoundError(f"файл разбиения не найден: {path}")

    entries = json.loads(path.read_text(encoding="utf-8"))
    return [
        Text2SqlExample(
            db_id=entry["db_id"],
            question=entry["question"].strip(),
            query=" ".join(entry["query"].split()),
        )
        for entry in entries
    ]


def load_dataset_schemas(root: Path, layout: DatasetLayout) -> dict[str, DatabaseSchema]:
    return load_schemas(Path(root) / layout.tables_file)


def database_path(root: Path, layout: DatasetLayout, db_id: str) -> Path:
    return Path(root) / layout.database_dir / db_id / f"{db_id}.sqlite"


def subsample(examples: list[Text2SqlExample], size: int, seed: int) -> list[Text2SqlExample]:
    """Урезание обучающей выборки с сохранением исходного порядка.

    Зерно здесь — зерно выборки, а не зерно обучения: выборка обязана быть
    одной и той же для всех трёх методов и всех трёх зёрен обучения, иначе
    разница между методами частично объяснялась бы разными данными.
    """
    if size >= len(examples):
        return list(examples)

    indices = sorted(random.Random(seed).sample(range(len(examples)), size))
    return [examples[index] for index in indices]


LAYOUTS: dict[str, DatasetLayout] = {"spider": SPIDER_LAYOUT, "pauq": PAUQ_LAYOUT}
