"""Проверки промпта, токенизации и маски функции потерь.

Ошибка в маске не роняет обучение: функция потерь считается, графики рисуются,
а модель учится воспроизводить схему базы вместо SQL. Поэтому маска
проверяется напрямую, а не через качество.
"""

import pytest
import torch

from kanlora.data.collate import (
    IGNORE_INDEX,
    Collator,
    build_dataset,
    build_prompt,
    build_prompts,
    encode,
)
from kanlora.data.loaders import Text2SqlExample
from kanlora.data.schema import DatabaseSchema, Table


class FakeTokenizer:
    """Посимвольный токенизатор: сравнивать удобно, поведение то же."""

    eos_token_id = 1
    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": [ord(character) for character in text]}


@pytest.fixture
def tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def schemas() -> dict[str, DatabaseSchema]:
    return {"db": DatabaseSchema("db", (Table("t", ("a", "b")),))}


@pytest.fixture
def examples() -> list[Text2SqlExample]:
    return [
        Text2SqlExample("db", "q1", "SELECT a FROM t"),
        Text2SqlExample("db", "q2", "SELECT b FROM t"),
    ]


def test_prompt_contains_schema_and_question() -> None:
    prompt = build_prompt("How many?", "t: a, b")
    assert "t: a, b" in prompt
    assert "How many?" in prompt
    assert prompt.endswith("SQL:\n")


def test_prompt_frame_is_identical_for_both_datasets() -> None:
    """Рамка одна, чтобы различие Spider и PAUQ сводилось к языку вопроса."""
    russian = build_prompt("Сколько певцов?", "t: a")
    english = build_prompt("How many singers?", "t: a")
    assert russian.replace("Сколько певцов?", "X") == english.replace("How many singers?", "X")


def test_labels_mask_the_prompt(tokenizer: FakeTokenizer) -> None:
    encoded = encode(tokenizer, "PROMPT", "SQL", max_length=64)
    prompt_length = len("PROMPT")

    assert encoded["labels"][:prompt_length] == [IGNORE_INDEX] * prompt_length
    assert all(label != IGNORE_INDEX for label in encoded["labels"][prompt_length:])


def test_labels_align_with_input_ids(tokenizer: FakeTokenizer) -> None:
    encoded = encode(tokenizer, "PROMPT", "SQL", max_length=64)
    assert len(encoded["input_ids"]) == len(encoded["labels"])

    target_part = encoded["input_ids"][len("PROMPT") :]
    assert encoded["labels"][len("PROMPT") :] == target_part


def test_sequence_ends_with_eos(tokenizer: FakeTokenizer) -> None:
    """Без конца последовательности генерация не остановится."""
    encoded = encode(tokenizer, "PROMPT", "SQL", max_length=64)
    assert encoded["input_ids"][-1] == tokenizer.eos_token_id
    assert encoded["labels"][-1] == tokenizer.eos_token_id


def test_truncation_cuts_the_prompt_and_keeps_the_target(tokenizer: FakeTokenizer) -> None:
    """Усечение режет голову промпта: запрос обязан уцелеть целиком."""
    encoded = encode(tokenizer, "P" * 100, "SELECT", max_length=20)

    assert len(encoded["input_ids"]) == 20
    target_ids = [ord(character) for character in "SELECT"] + [tokenizer.eos_token_id]
    assert encoded["input_ids"][-len(target_ids) :] == target_ids
    assert encoded["labels"][-len(target_ids) :] == target_ids


def test_target_longer_than_limit_raises(tokenizer: FakeTokenizer) -> None:
    """Молча выбросить часть запроса нельзя — это тихая порча обучающих данных."""
    with pytest.raises(ValueError, match="не помещается"):
        encode(tokenizer, "P", "S" * 100, max_length=10)


def test_build_dataset_reports_truncations(
    tokenizer: FakeTokenizer, examples, schemas
) -> None:
    dataset, truncated = build_dataset(examples, schemas, tokenizer, max_length=1024)
    assert len(dataset) == 2
    assert truncated == 0

    _, truncated_many = build_dataset(examples, schemas, tokenizer, max_length=30)
    assert truncated_many == 2


def test_build_prompts_matches_dataset_order(examples, schemas) -> None:
    prompts = build_prompts(examples, schemas)
    assert len(prompts) == 2
    assert "q1" in prompts[0] and "q2" in prompts[1]


def test_collator_pads_inputs_and_masks_padding(tokenizer: FakeTokenizer) -> None:
    batch = [
        {"input_ids": [5, 6, 7], "labels": [IGNORE_INDEX, 6, 7]},
        {"input_ids": [8], "labels": [8]},
    ]
    collated = Collator(pad_token_id=tokenizer.pad_token_id)(batch)

    assert collated["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
    assert collated["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert collated["labels"].tolist() == [[IGNORE_INDEX, 6, 7], [8, IGNORE_INDEX, IGNORE_INDEX]]
    assert collated["input_ids"].dtype == torch.long
