"""Проверки Exact Match.

В Spider это покомпонентное сравнение разобранного запроса, а не равенство
строк: запросы, различающиеся регистром, лишними скобками или порядком
условий в WHERE, считаются совпавшими. Тесты закрепляют именно это поведение
на парах из официального набора.
"""

from pathlib import Path

import pytest

from kanlora.eval.exact_match import exact_match, exact_match_by_hardness

TABLES = Path(__file__).parents[1] / "data" / "fixtures" / "spider_mini" / "tables.json"
DB = "concert_singer"


def test_identical_queries_match() -> None:
    query = "SELECT count(*) FROM singer"
    assert exact_match([query], [query], [DB], TABLES) == pytest.approx(1.0)


def test_case_and_spacing_do_not_matter() -> None:
    """Разбор запроса, а не строки: регистр ключевых слов не влияет."""
    score = exact_match(
        ["SELECT count(*) FROM singer"],
        ["select  COUNT(*)   from   singer"],
        [DB],
        TABLES,
    )
    assert score == pytest.approx(1.0)


def test_different_queries_do_not_match() -> None:
    score = exact_match(
        ["SELECT count(*) FROM singer"], ["SELECT Name FROM singer"], [DB], TABLES
    )
    assert score == pytest.approx(0.0)


def test_unparsable_prediction_counts_as_a_miss() -> None:
    """Модель часто выдаёт мусор; это промах, а не исключение на весь прогон."""
    score = exact_match(["SELECT count(*) FROM singer"], ["не знаю"], [DB], TABLES)
    assert score == pytest.approx(0.0)


def test_score_is_the_share_of_matches() -> None:
    gold = ["SELECT count(*) FROM singer", "SELECT Name FROM stadium"]
    predicted = ["SELECT count(*) FROM singer", "SELECT Location FROM stadium"]
    assert exact_match(gold, predicted, [DB, DB], TABLES) == pytest.approx(0.5)


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="длины"):
        exact_match(["SELECT 1"], ["SELECT 1", "SELECT 2"], [DB], TABLES)


def test_hardness_breakdown_covers_all_levels() -> None:
    """Разбивка по сложности — та же, что в официальном сценарии."""
    breakdown = exact_match_by_hardness(
        ["SELECT count(*) FROM singer"], ["SELECT count(*) FROM singer"], [DB], TABLES
    )
    assert set(breakdown) == {"easy", "medium", "hard", "extra", "all"}
