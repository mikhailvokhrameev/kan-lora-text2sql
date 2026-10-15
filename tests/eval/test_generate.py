# tests/eval/test_generate.py
"""Проверки генерации.

Генерация жадная во всех прогонах: отбор с температурой добавил бы к метрике
случайность сверх той, что уже вносят три зерна, и различия между методами
утонули бы окончательно.
"""

import pytest
import torch

from kanlora.data.collate import build_prompt, encode, truncate_prompt
from kanlora.eval.generate import generate_sql, normalize_sql


def test_cuts_at_the_first_newline() -> None:
    assert normalize_sql("SELECT a FROM t\nQuestion: next") == "SELECT a FROM t"


def test_cuts_at_the_semicolon() -> None:
    assert normalize_sql("SELECT a FROM t; SELECT b") == "SELECT a FROM t"


def test_collapses_whitespace() -> None:
    assert normalize_sql("  SELECT   a\tFROM  t  ") == "SELECT a FROM t"


def test_empty_generation_stays_empty() -> None:
    assert normalize_sql("   ") == ""


def test_does_not_cut_at_semicolon_inside_single_quoted_string() -> None:
    text = "SELECT * FROM t WHERE name = 'a;b' ; SELECT c"
    assert normalize_sql(text) == "SELECT * FROM t WHERE name = 'a;b'"


def test_does_not_cut_at_semicolon_inside_double_quoted_string() -> None:
    text = 'SELECT * FROM t WHERE name = "a;b" ; SELECT c'
    assert normalize_sql(text) == 'SELECT * FROM t WHERE name = "a;b"'


def test_does_not_cut_at_newline_inside_quoted_string() -> None:
    text = "SELECT * FROM t WHERE name = 'a\nb'\nQuestion: next"
    assert normalize_sql(text) == "SELECT * FROM t WHERE name = 'a b'"


def test_escaped_quote_does_not_end_the_string() -> None:
    text = r"SELECT * FROM t WHERE name = 'it\'s fine; ok' ; SELECT c"
    assert normalize_sql(text) == r"SELECT * FROM t WHERE name = 'it\'s fine; ok'"


@pytest.mark.slow
def test_generates_one_string_per_prompt(tiny_causal_lm, tiny_tokenizer) -> None:
    prompts = ["SELECT", "FROM", "WHERE"]

    generated = generate_sql(
        tiny_causal_lm, tiny_tokenizer, prompts,
        device=torch.device("cpu"), batch_size=2, max_new_tokens=4, max_length=64,
    )

    assert len(generated) == len(prompts)
    assert all(isinstance(item, str) for item in generated)


@pytest.mark.slow
def test_batching_does_not_change_the_result(tiny_causal_lm, tiny_tokenizer) -> None:
    """Дополнение слева не должно влиять на выход — иначе метрика зависела бы от размера батча."""
    prompts = ["SELECT a", "FROM long table name here", "WHERE"]
    common = {"device": torch.device("cpu"), "max_new_tokens": 4, "max_length": 64}

    one_by_one = generate_sql(tiny_causal_lm, tiny_tokenizer, prompts, batch_size=1, **common)
    batched = generate_sql(tiny_causal_lm, tiny_tokenizer, prompts, batch_size=3, **common)
    assert one_by_one == batched


class _RecordingModel:
    """Заглушка модели: запоминает аргументы generate и ничего не дописывает."""

    def __init__(self) -> None:
        self.config = type("Config", (), {"use_cache": False})()
        self.calls: list[dict] = []

    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, input_ids, attention_mask, **kwargs):
        self.calls.append({"input_ids": input_ids, "attention_mask": attention_mask, **kwargs})
        return input_ids


def test_generation_stops_at_the_tokenizer_eos(tiny_tokenizer) -> None:
    """Генерация обрывается на eos_token_id токенизатора — у Qwen2.5-Instruct это <|im_end|>,
    тем же токеном collate.encode завершает ответ при обучении."""
    model = _RecordingModel()
    generate_sql(model, tiny_tokenizer, ["SELECT"], device=torch.device("cpu"), max_length=256)

    assert model.calls[0]["eos_token_id"] == tiny_tokenizer.eos_token_id
    assert model.calls[0]["do_sample"] is False


def test_long_prompt_gets_the_same_token_ids_at_train_and_eval(tiny_tokenizer) -> None:
    """Длинный промпт усекается на оценке по тому же правилу, что и при обучении.

    Бюджет промпта на обучении — max_length минус длина ответа, на оценке —
    max_length минус max_new_tokens; при равных бюджетах модель обязана
    получить на оценке ровно те токены, после которых её учили писать SQL.
    """
    schema = " | ".join(f"table_{index}: id, name" for index in range(40))
    prompt = build_prompt("How many?", schema, tiny_tokenizer)
    sql = "SELECT count(*) FROM table_0"
    max_length = 400
    target_length = len(sql) + 1  # посимвольно плюс eos

    train_ids = encode(tiny_tokenizer, prompt, sql, max_length)["input_ids"][:-target_length]
    assert len(tiny_tokenizer(prompt)["input_ids"]) > len(train_ids)

    model = _RecordingModel()
    generate_sql(
        model, tiny_tokenizer, [prompt], device=torch.device("cpu"),
        max_new_tokens=target_length, max_length=max_length,
    )

    eval_ids = model.calls[0]["input_ids"][0].tolist()
    assert eval_ids == train_ids
    assert eval_ids == truncate_prompt(tiny_tokenizer, prompt, max_length - target_length)
    assert model.calls[0]["attention_mask"][0].tolist() == [1] * len(eval_ids)


def test_eval_prompts_are_left_padded_after_truncation(tiny_tokenizer) -> None:
    """Каждый промпт пакета усекается отдельно, затем пакет дополняется слева."""
    long_prompt = build_prompt("q", " | ".join(f"t{i}: a" for i in range(50)), tiny_tokenizer)
    model = _RecordingModel()
    generate_sql(
        model, tiny_tokenizer, ["SELECT", long_prompt], device=torch.device("cpu"),
        batch_size=2, max_new_tokens=8, max_length=308,
    )

    input_ids = model.calls[0]["input_ids"].tolist()
    mask = model.calls[0]["attention_mask"].tolist()
    assert len(input_ids[1]) == 300
    assert input_ids[1] == truncate_prompt(tiny_tokenizer, long_prompt, 300)
    assert input_ids[0] == [tiny_tokenizer.pad_token_id] * 294 + tiny_tokenizer("SELECT")["input_ids"]
    assert mask[0] == [0] * 294 + [1] * 6


class _BosTokenizer:
    """Добавляет BOS при add_special_tokens=True, как Llama-3: на обучении его нет."""

    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2
    padding_side = "right"

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        ids = [ord(character) % 64 for character in text]
        return {"input_ids": ([self.bos_token_id] if add_special_tokens else []) + ids}

    def batch_decode(self, sequences, skip_special_tokens: bool = True) -> list[str]:
        return ["" for _ in sequences.tolist()]


def test_eval_prompt_is_tokenized_without_special_tokens() -> None:
    """Обучение токенизирует промпт с add_special_tokens=False; на оценке так же,
    иначе токенизатор с BOS (Llama-3) получил бы на оценке лишний первый токен."""
    tokenizer = _BosTokenizer()
    model = _RecordingModel()
    generate_sql(model, tokenizer, ["SELECT"], device=torch.device("cpu"), max_length=256)

    assert model.calls[0]["input_ids"][0].tolist() == [ord(c) % 64 for c in "SELECT"]
    assert encode(tokenizer, "SELECT", "a", max_length=256)["input_ids"][:6] == model.calls[0][
        "input_ids"
    ][0].tolist()


def test_budget_must_leave_room_for_the_prompt(tiny_tokenizer) -> None:
    with pytest.raises(ValueError, match="max_new_tokens"):
        generate_sql(
            _RecordingModel(), tiny_tokenizer, ["SELECT"], device=torch.device("cpu"),
            max_new_tokens=128, max_length=128,
        )
