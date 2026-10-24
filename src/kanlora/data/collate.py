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
    "truncate_prompt",
]

IGNORE_INDEX = -100

SYSTEM_PROMPT = (
    "Translate the question into a SQL query for the given database schema. "
    "Answer with the SQL query only."
)

USER_TEMPLATE = "Database schema:\n{schema}\n\nQuestion: {question}"

# Текст шаблона по обе стороны схемы: по ним truncate_prompt находит схему в
# готовом промпте и режет только её.
_SCHEMA_OPEN, _, _AFTER_SCHEMA = USER_TEMPLATE.partition("{schema}")
_SCHEMA_CLOSE = _AFTER_SCHEMA.partition("{question}")[0]


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


def _token_ids(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def truncate_prompt(tokenizer, prompt: str, budget: int) -> list[int]:
    """Токены промпта, усечённые до budget по одному правилу для обучения и оценки.

    Промпт, который помещается, токенизируется целиком и не меняется. Иначе
    режется голова схемы: системная реплика, открытие реплики user, вопрос и
    открытие реплики assistant остаются нетронутыми. Для этого промпт делится
    по тексту шаблона вокруг схемы на три куска, каждый токенизируется
    отдельно, и у схемы отбрасываются первые токены.

    Запасной путь — усечение головы: если схемы в промпте нет (промпт построен
    не build_prompt) или не помещается даже промпт без схемы, остаются
    последние budget токенов того, что есть.
    """
    ids = _token_ids(tokenizer, prompt)
    if len(ids) <= budget:
        return ids
    if budget <= 0:
        return []

    start = prompt.find(_SCHEMA_OPEN)
    end = prompt.find(_SCHEMA_CLOSE, start + len(_SCHEMA_OPEN)) if start >= 0 else -1
    if end < 0:
        return ids[-budget:]

    start += len(_SCHEMA_OPEN)
    head = _token_ids(tokenizer, prompt[:start])
    schema = _token_ids(tokenizer, prompt[start:end])
    tail = _token_ids(tokenizer, prompt[end:])
    room = budget - len(head) - len(tail)
    kept = schema[-room:] if room > 0 else []
    return (head + kept + tail)[-budget:]


def encode(tokenizer, prompt: str, target: str, max_length: int) -> dict[str, list[int]]:
    """Токенизирует пару и маскирует токены промпта.

    Последовательность — промпт, ответ и eos_token_id; у Qwen2.5-Instruct это
    <|im_end|>, конец реплики, на котором останавливается генерация. Перевод
    строки, который шаблон ставит после <|im_end|>, не добавляется: после
    конца реплики учить нечему.

    При переполнении усекается промпт (truncate_prompt — то же правило, что
    у генерации на оценке): запрос обязан уцелеть целиком, иначе обучающая
    пара становится неверной, не вызвав ошибки.
    """
    target_ids = _token_ids(tokenizer, target) + [
        tokenizer.eos_token_id
    ]

    if len(target_ids) > max_length:
        raise ValueError(
            f"SQL-запрос не помещается в {max_length} токенов "
            f"(требуется {len(target_ids)}); увеличьте max_length"
        )

    prompt_ids = truncate_prompt(tokenizer, prompt, max_length - len(target_ids))

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
