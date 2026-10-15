"""Промпт, токенизация, маска функции потерь и сборка батча.

Промпт строится шаблоном диалога самого токенизатора (apply_chat_template):
Instruct-модель дообучалась именно на этой разметке, и свой формат без неё
занизил бы и базовую модель, и стартовую точку всех трёх методов адаптации.
Одна и та же функция build_prompt строит промпт и для обучения, и для оценки.

Функция потерь считается только по токенам ответа assistant (SQL-запрос) и
завершающему его концу реплики: системная и пользовательская реплики и
разметка шаблона получают IGNORE_INDEX. Без этого модель училась бы
воспроизводить схему базы, и ошибка не выдала бы себя ничем — обучение бы
шло, метрики бы считались.

Рамка промпта одна и та же для Spider и для PAUQ, чтобы различие между
наборами сводилось к языку вопроса, а не к оформлению.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from kanlora.data.loaders import Text2SqlExample
from kanlora.data.schema import DatabaseSchema, serialize_schema

__all__ = [
    "IGNORE_INDEX",
    "SYSTEM_PROMPT",
    "USER_TEMPLATE",
    "Collator",
    "build_dataset",
    "build_messages",
    "build_prompt",
    "build_prompts",
    "encode",
]

IGNORE_INDEX = -100

SYSTEM_PROMPT = (
    "Translate the question into a SQL query for the given database schema. "
    "Answer with the SQL query only."
)

USER_TEMPLATE = "Database schema:\n{schema}\n\nQuestion: {question}"


def build_messages(question: str, schema_text: str) -> list[dict[str, str]]:
    """Реплики system и user; ответ assistant (SQL) сюда не входит."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(schema=schema_text, question=question)},
    ]


def build_prompt(question: str, schema_text: str, tokenizer) -> str:
    """Промпт шаблоном диалога токенизатора, оборванный на начале ответа assistant.

    add_generation_prompt=True дописывает открытие реплики assistant: с этого
    места модель пишет SQL и на оценке, и при обучении (encode дописывает к
    этой же строке ответ и конец реплики).
    """
    return tokenizer.apply_chat_template(
        build_messages(question, schema_text), tokenize=False, add_generation_prompt=True
    )


def encode(tokenizer, prompt: str, target: str, max_length: int) -> dict[str, list[int]]:
    """Токенизирует пару и маскирует токены промпта.

    Последовательность — промпт, ответ и eos_token_id; у Qwen2.5-Instruct это
    <|im_end|>, конец реплики, на котором останавливается генерация. Перевод
    строки, который шаблон ставит после <|im_end|>, не добавляется: после
    конца реплики учить нечему.

    При переполнении режется голова промпта: запрос обязан уцелеть целиком,
    иначе обучающая пара становится неверной, не вызвав ошибки.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"] + [
        tokenizer.eos_token_id
    ]

    if len(target_ids) > max_length:
        raise ValueError(
            f"SQL-запрос не помещается в {max_length} токенов "
            f"(требуется {len(target_ids)}); увеличьте max_length"
        )

    room = max_length - len(target_ids)
    prompt_ids = prompt_ids[-room:] if room > 0 else []

    return {
        "input_ids": prompt_ids + target_ids,
        "labels": [IGNORE_INDEX] * len(prompt_ids) + target_ids,
    }


def build_prompts(
    examples: Sequence[Text2SqlExample], schemas: dict[str, DatabaseSchema], tokenizer
) -> list[str]:
    return [
        build_prompt(example.question, serialize_schema(schemas[example.db_id]), tokenizer)
        for example in examples
    ]


def build_dataset(
    examples: Sequence[Text2SqlExample],
    schemas: dict[str, DatabaseSchema],
    tokenizer,
    max_length: int,
) -> tuple[list[dict[str, list[int]]], int]:
    """Кодирует все примеры и сообщает, сколько промптов было усечено.

    Число усечений идёт в карточку результата: если оно велико, часть схем
    не доехала до модели, и это влияет на метрику сильнее, чем метод адаптации.
    """
    prompts = build_prompts(examples, schemas, tokenizer)

    dataset: list[dict[str, list[int]]] = []
    truncated = 0
    for prompt, example in zip(prompts, examples, strict=True):
        prompt_length = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        target_length = len(tokenizer(example.query, add_special_tokens=False)["input_ids"]) + 1
        if prompt_length + target_length > max_length:
            truncated += 1
        dataset.append(encode(tokenizer, prompt, example.query, max_length))

    return dataset, truncated


class Collator:
    """Дополняет батч до самой длинной последовательности справа."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: Sequence[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(item["input_ids"]) for item in batch)

        input_ids, attention_mask, labels = [], [], []
        for item in batch:
            padding = width - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append([1] * len(item["input_ids"]) + [0] * padding)
            labels.append(item["labels"] + [IGNORE_INDEX] * padding)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
