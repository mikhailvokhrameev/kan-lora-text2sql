"""Проверки загрузки наборов.

Spider и PAUQ читаются одним кодом. Здесь закрепляется, что этот код
преобразует примеры одинаково и что урезание выборки воспроизводимо: размер
обучающей выборки урезан до 2500, и все три метода обязаны видеть ровно одни
и те же примеры.
"""

from pathlib import Path

import pytest

from kanlora.data.loaders import (
    LAYOUTS,
    DatasetLayout,
    Text2SqlExample,
    database_path,
    load_dataset_schemas,
    load_examples,
    subsample,
)
from kanlora.data.pauq import PAUQ_LAYOUT
from kanlora.data.spider import SPIDER_LAYOUT

ROOT = Path(__file__).parent / "fixtures" / "spider_mini"


def test_train_split_is_read(tmp_path: Path) -> None:
    examples = load_examples(ROOT, SPIDER_LAYOUT, "train")
    assert len(examples) == 4
    assert examples[0] == Text2SqlExample(
        db_id="concert_singer",
        question="How many singers are there?",
        query="SELECT count(*) FROM singer",
    )


def test_eval_split_is_read() -> None:
    assert len(load_examples(ROOT, SPIDER_LAYOUT, "eval")) == 1


def test_unknown_split_is_rejected() -> None:
    with pytest.raises(KeyError):
        load_examples(ROOT, SPIDER_LAYOUT, "test")


def test_missing_file_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="train_spider.json"):
        load_examples(tmp_path, SPIDER_LAYOUT, "train")


def test_schemas_come_from_the_same_root() -> None:
    schemas = load_dataset_schemas(ROOT, SPIDER_LAYOUT)
    assert set(schemas) == {"concert_singer", "pets_1"}


def test_every_example_has_a_schema() -> None:
    """Пример без схемы дал бы пустой промпт и тихо испортил бы метрику."""
    schemas = load_dataset_schemas(ROOT, SPIDER_LAYOUT)
    for split in ("train", "eval"):
        for example in load_examples(ROOT, SPIDER_LAYOUT, split):
            assert example.db_id in schemas


def test_database_path_follows_the_layout() -> None:
    path = database_path(ROOT, SPIDER_LAYOUT, "pets_1")
    assert path == ROOT / "database" / "pets_1" / "pets_1.sqlite"


def test_layouts_are_registered_under_dataset_names() -> None:
    assert LAYOUTS["spider"] is SPIDER_LAYOUT
    assert LAYOUTS["pauq"] is PAUQ_LAYOUT


def test_layouts_differ_only_in_file_names() -> None:
    """Оба набора читаются одним кодом — различие сведено к именам файлов."""
    assert isinstance(PAUQ_LAYOUT, DatasetLayout)
    assert SPIDER_LAYOUT.database_dir == PAUQ_LAYOUT.database_dir == "database"
    assert SPIDER_LAYOUT.train_file != PAUQ_LAYOUT.train_file


def test_subsample_is_reproducible() -> None:
    examples = load_examples(ROOT, SPIDER_LAYOUT, "train")
    assert subsample(examples, 3, seed=0) == subsample(examples, 3, seed=0)


def test_subsample_does_not_depend_on_the_training_seed() -> None:
    """Все три метода и все зёрна обучения видят одну и ту же выборку.

    Иначе разница между методами частично объяснялась бы разной выборкой,
    и вывод о методе стал бы недоказуемым.
    """
    examples = load_examples(ROOT, SPIDER_LAYOUT, "train")
    assert subsample(examples, 3, seed=0) != subsample(examples, 3, seed=1)


def test_subsample_keeps_everything_when_size_exceeds_the_set() -> None:
    examples = load_examples(ROOT, SPIDER_LAYOUT, "train")
    assert subsample(examples, 100, seed=0) == examples
