"""Проверки Execution Accuracy.

Сравниваются результаты исполнения, а не тексты запросов: запрос, написанный
иначе, но дающий тот же ответ, засчитывается. Именно этим метрика отличается
от Exact Match, и именно поэтому она нужна отдельно.
"""

import sqlite3
from pathlib import Path

import pytest

from kanlora.data.spider import SPIDER_LAYOUT
from kanlora.eval.execution import (
    count_available_databases,
    execute_query,
    execution_accuracy,
    results_match,
)

DB = "shop"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / "database" / DB
    path.mkdir(parents=True)
    connection = sqlite3.connect(path / f"{DB}.sqlite")
    connection.executescript(
        """
        CREATE TABLE item (id INTEGER, name TEXT, price INTEGER);
        INSERT INTO item VALUES (1, 'apple', 30), (2, 'pear', 10), (3, 'plum', 20);
        """
    )
    connection.commit()
    connection.close()
    return tmp_path


def test_executes_and_returns_rows(root: Path) -> None:
    outcome = execute_query(root / "database" / DB / f"{DB}.sqlite", "SELECT count(*) FROM item")
    assert outcome.failed is False
    assert outcome.rows == [(3,)]


def test_broken_query_is_reported_not_raised(root: Path) -> None:
    outcome = execute_query(root / "database" / DB / f"{DB}.sqlite", "SELECT FROM nowhere")
    assert outcome.failed is True
    assert outcome.rows is None


def test_missing_database_is_reported_not_raised(tmp_path: Path) -> None:
    outcome = execute_query(tmp_path / "absent.sqlite", "SELECT 1")
    assert outcome.failed is True


def test_row_order_is_ignored_without_order_by(root: Path) -> None:
    """Без ORDER BY порядок строк в SQL не определён, и требовать его нельзя."""
    database = root / "database" / DB / f"{DB}.sqlite"
    ascending = execute_query(database, "SELECT name FROM item ORDER BY price")
    descending = execute_query(database, "SELECT name FROM item ORDER BY price DESC")

    assert results_match(ascending, descending, order_matters=False) is True
    assert results_match(ascending, descending, order_matters=True) is False


def test_failed_execution_never_matches(root: Path) -> None:
    database = root / "database" / DB / f"{DB}.sqlite"
    good = execute_query(database, "SELECT 1")
    bad = execute_query(database, "SELECT FROM")
    assert results_match(good, bad, order_matters=False) is False
    assert results_match(bad, bad, order_matters=False) is False


def test_accuracy_counts_equivalent_rewrites_as_correct(root: Path) -> None:
    """Главное свойство метрики: другой текст, тот же ответ — засчитано."""
    score = execution_accuracy(
        gold=["SELECT name FROM item WHERE price > 15"],
        predicted=["SELECT name FROM item WHERE NOT price <= 15"],
        db_ids=[DB],
        root=root,
        layout=SPIDER_LAYOUT,
    )
    assert score == pytest.approx(1.0)


def test_accuracy_respects_order_by(root: Path) -> None:
    score = execution_accuracy(
        gold=["SELECT name FROM item ORDER BY price"],
        predicted=["SELECT name FROM item ORDER BY price DESC"],
        db_ids=[DB],
        root=root,
        layout=SPIDER_LAYOUT,
    )
    assert score == pytest.approx(0.0)


def test_mismatched_lengths_are_rejected(root: Path) -> None:
    with pytest.raises(ValueError, match="длины"):
        execution_accuracy(["SELECT 1"], [], [DB], root, SPIDER_LAYOUT)


def test_count_available_databases_is_zero_for_empty_input(root: Path) -> None:
    assert count_available_databases(root, SPIDER_LAYOUT, []) == 0


def test_count_available_databases_counts_present_files_only(root: Path) -> None:
    assert (
        count_available_databases(root, SPIDER_LAYOUT, [DB, "missing_db", "also_missing"]) == 1
    )


def test_count_available_databases_is_zero_when_none_found(tmp_path: Path) -> None:
    assert count_available_databases(tmp_path, SPIDER_LAYOUT, ["missing_db"]) == 0
