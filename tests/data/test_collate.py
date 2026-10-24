"""Проверки промпта, токенизации и маски функции потерь.

Ошибка в маске не роняет обучение: функция потерь считается, графики рисуются,
а модель учится воспроизводить схему базы вместо SQL. Поэтому маска
проверяется напрямую, а не через качество.
"""

import pytest
import torch

from kanlora.data.collate import (
    IGNORE_INDEX,
    SYSTEM_PROMPT,
    Collator,
    build_dataset,
    build_messages,
    build_prompt,
    build_prompts,
    encode,
    truncate_prompt,
)
from kanlora.data.loaders import Text2SqlExample
from kanlora.data.schema import DatabaseSchema, Table


class FakeTokenizer:
    """Посимвольный токенизатор: сравнивать удобно, поведение то же.

    Шаблон диалога повторяет разметку ChatML, которой пользуется Qwen2.5:
    те же маркеры начала и конца реплики и та же приписка для генерации.
    """

    eos_token_id = 1
    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": [ord(character) for character in text]}

    def apply_chat_template(
        self, messages, tokenize: bool = True, add_generation_prompt: bool = False
    ) -> str:
        assert not tokenize
        text = "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        )
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text


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


def test_messages_are_system_instruction_then_schema_and_question() -> None:
    messages = build_messages("How many?", "t: a, b")

    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert "t: a, b" in messages[1]["content"]
    assert "How many?" in messages[1]["content"]


def test_prompt_goes_through_the_chat_template(tokenizer: FakeTokenizer) -> None:
    """Промпт строит шаблон диалога токенизатора и обрывается на начале ответа."""
    prompt = build_prompt("How many?", "t: a, b", tokenizer)

    assert prompt == tokenizer.apply_chat_template(
        build_messages("How many?", "t: a, b"), tokenize=False, add_generation_prompt=True
    )
    assert prompt.startswith(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>")
    assert "t: a, b" in prompt
    assert "How many?" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_prompt_frame_is_identical_for_both_datasets(tokenizer: FakeTokenizer) -> None:
    """Рамка одна, чтобы различие Spider и PAUQ сводилось к языку вопроса."""
    russian = build_prompt("Сколько певцов?", "t: a", tokenizer)
    english = build_prompt("How many singers?", "t: a", tokenizer)
    assert russian.replace("Сколько певцов?", "X") == english.replace("How many singers?", "X")


def test_loss_covers_only_the_answer_and_the_end_of_turn(
    tokenizer: FakeTokenizer,
) -> None:
    """Системная, пользовательская реплики и разметка шаблона в функцию потерь не входят."""
    prompt = build_prompt("q", "t: a", tokenizer)
    encoded = encode(tokenizer, prompt, "SELECT a FROM t", max_length=1024)

    supervised = [label for label in encoded["labels"] if label != IGNORE_INDEX]
    assert supervised == [ord(character) for character in "SELECT a FROM t"] + [
        tokenizer.eos_token_id
    ]
    assert encoded["input_ids"][: len(prompt)] == [ord(character) for character in prompt]
    # Перевод строки, который шаблон ставит после <|im_end|>, в последовательность
    # не попадает: генерация останавливается на <|im_end|>, дальше учить нечему.
    assert encoded["input_ids"][-1] == tokenizer.eos_token_id


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


def _ids(text: str) -> list[int]:
    return [ord(character) for character in text]


def test_truncation_cuts_the_schema_head_and_keeps_the_frame(tokenizer: FakeTokenizer) -> None:
    """Режется голова схемы: системная реплика, разметка шаблона и вопрос уцелевают.

    Иначе при усечении по голове всего промпта модель теряла бы инструкцию и
    открытие реплики user и видела бы промпт, какого шаблон не порождает.
    """
    schema = " | ".join(f"table_{index}: id, name" for index in range(30))
    prompt = build_prompt("How many?", schema, tokenizer)
    head, _, rest = prompt.partition(schema)
    budget = len(prompt) - 100

    ids = truncate_prompt(tokenizer, prompt, budget)

    assert len(ids) == budget
    assert ids == _ids(head) + _ids(schema[100:]) + _ids(rest)
    system_turn = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
    assert ids[: len(system_turn)] == _ids(system_turn)
    assert ids[-len(rest) :] == _ids("\n\nQuestion: How many?<|im_end|>\n<|im_start|>assistant\n")


def test_prompt_that_fits_is_not_touched(tokenizer: FakeTokenizer) -> None:
    prompt = build_prompt("q", "t: a", tokenizer)
    assert truncate_prompt(tokenizer, prompt, len(prompt)) == _ids(prompt)


def test_encode_applies_the_schema_truncation(tokenizer: FakeTokenizer) -> None:
    """encode усекает промпт той же функцией, что и генерация на оценке."""
    prompt = build_prompt("q", " | ".join(f"t{index}: a" for index in range(60)), tokenizer)
    encoded = encode(tokenizer, prompt, "SELECT", max_length=400)

    room = 400 - len("SELECT") - 1
    assert encoded["input_ids"][:room] == truncate_prompt(tokenizer, prompt, room)
    assert encoded["input_ids"][: len("<|im_start|>system")] == _ids("<|im_start|>system")


def test_frame_longer_than_budget_falls_back_to_head_truncation(
    tokenizer: FakeTokenizer,
) -> None:
    """Если не помещается даже рамка с вопросом, режется голова того, что осталось без схемы."""
    prompt = build_prompt("q", "t: a, b, c", tokenizer)
    head, _, rest = prompt.partition("t: a, b, c")

    assert truncate_prompt(tokenizer, prompt, 20) == _ids(head + rest)[-20:]
    assert truncate_prompt(tokenizer, prompt, 0) == []


def test_prompt_without_schema_falls_back_to_head_truncation(tokenizer: FakeTokenizer) -> None:
    assert truncate_prompt(tokenizer, "P" * 10 + "Q" * 5, 5) == _ids("Q" * 5)


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

    _, truncated_many = build_dataset(examples, schemas, tokenizer, max_length=60)
    assert truncated_many == 2


def test_build_prompts_matches_dataset_order(
    tokenizer: FakeTokenizer, examples, schemas
) -> None:
    prompts = build_prompts(examples, schemas, tokenizer)
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


QWEN = "Qwen/Qwen2.5-Coder-0.5B-Instruct"


@pytest.fixture(scope="module")
def qwen_tokenizer():
    """Настоящий токенизатор Qwen2.5 из локального кеша; без него тест пропускается."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    except (OSError, ValueError) as error:
        pytest.skip(f"токенизатор {QWEN} недоступен без сети: {error}")


def test_eval_prompt_is_a_prefix_of_the_training_sequence(qwen_tokenizer) -> None:
    """Модель на оценке видит ровно тот текст, после которого её учили писать SQL.

    Строка оценки (add_generation_prompt=True) — побайтовый префикс строки
    обучения, построенной тем же шаблоном с ответом assistant; а закодированная
    обучающая последовательность совпадает с токенизацией этой строки целиком,
    кроме перевода строки после <|im_end|>.
    """
    question, schema_text = "How many singers?", "singer: singer_id, name"
    sql = "SELECT count(*) FROM singer"
    prompt = build_prompt(question, schema_text, qwen_tokenizer)
    training_text = qwen_tokenizer.apply_chat_template(
        [*build_messages(question, schema_text), {"role": "assistant", "content": sql}],
        tokenize=False,
    )

    assert training_text.encode("utf-8").startswith(prompt.encode("utf-8"))
    assert qwen_tokenizer.eos_token == "<|im_end|>"
    assert training_text.endswith("<|im_end|>\n")

    encoded = encode(qwen_tokenizer, prompt, sql, max_length=1024)
    expected = qwen_tokenizer(training_text[: -len("\n")], add_special_tokens=False)["input_ids"]
    assert encoded["input_ids"] == expected

    supervised = [label for label in encoded["labels"] if label != IGNORE_INDEX]
    assert qwen_tokenizer.decode(supervised) == sql + "<|im_end|>"


def test_truncated_qwen_prompt_keeps_the_chat_frame(qwen_tokenizer) -> None:
    """На настоящем токенизаторе усечённый промпт сохраняет системную реплику,
    вопрос и открытие ответа assistant, а режется только голова схемы."""
    schema = " | ".join(f"table_{index}: id, name, value" for index in range(60))
    prompt = build_prompt("How many singers?", schema, qwen_tokenizer)
    full = qwen_tokenizer(prompt, add_special_tokens=False)["input_ids"]

    ids = truncate_prompt(qwen_tokenizer, prompt, len(full) // 2)
    text = qwen_tokenizer.decode(ids)

    assert len(ids) == len(full) // 2
    assert text.startswith(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n")
    assert text.endswith("\n\nQuestion: How many singers?<|im_end|>\n<|im_start|>assistant\n")
    assert "table_59: id, name, value" in text and "table_0:" not in text
