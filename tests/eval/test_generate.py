# tests/eval/test_generate.py
"""Проверки генерации.

Генерация жадная во всех прогонах: отбор с температурой добавил бы к метрике
случайность сверх той, что уже вносят три зерна, и различия между методами
утонули бы окончательно.
"""

import pytest
import torch

from kanlora.eval.generate import generate_sql, normalize_sql


def test_cuts_at_the_first_newline() -> None:
    assert normalize_sql("SELECT a FROM t\nQuestion: next") == "SELECT a FROM t"


def test_cuts_at_the_semicolon() -> None:
    assert normalize_sql("SELECT a FROM t; SELECT b") == "SELECT a FROM t"


def test_collapses_whitespace() -> None:
    assert normalize_sql("  SELECT   a\tFROM  t  ") == "SELECT a FROM t"


def test_empty_generation_stays_empty() -> None:
    assert normalize_sql("   ") == ""


@pytest.mark.slow
def test_generates_one_string_per_prompt(tiny_causal_lm, tiny_tokenizer) -> None:
    prompts = ["SELECT", "FROM", "WHERE"]

    generated = generate_sql(
        tiny_causal_lm, tiny_tokenizer, prompts,
        device=torch.device("cpu"), batch_size=2, max_new_tokens=4,
    )

    assert len(generated) == len(prompts)
    assert all(isinstance(item, str) for item in generated)


@pytest.mark.slow
def test_batching_does_not_change_the_result(tiny_causal_lm, tiny_tokenizer) -> None:
    """Дополнение слева не должно влиять на выход — иначе метрика зависела бы от размера батча."""
    prompts = ["SELECT a", "FROM long table name here", "WHERE"]
    common = {"device": torch.device("cpu"), "max_new_tokens": 4}

    one_by_one = generate_sql(tiny_causal_lm, tiny_tokenizer, prompts, batch_size=1, **common)
    batched = generate_sql(tiny_causal_lm, tiny_tokenizer, prompts, batch_size=3, **common)
    assert one_by_one == batched
