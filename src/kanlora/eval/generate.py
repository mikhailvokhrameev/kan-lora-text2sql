# src/kanlora/eval/generate.py
"""Пакетная жадная генерация SQL.

Жадная во всех прогонах без исключения: отбор с температурой сделал бы
метрику случайной величиной сверх той случайности, которую уже вносят три
зерна, и различия между методами перестали бы быть различимы.

Дополнение слева обязательно: при дополнении справа модель продолжала бы
последовательность от заполнителя, а не от промпта.
"""

from __future__ import annotations

import re

import torch

from kanlora.data.collate import truncate_prompt

__all__ = ["generate_sql", "normalize_sql"]

_WHITESPACE = re.compile(r"\s+")


def normalize_sql(text: str) -> str:
    """Обрезает продолжение после запроса и приводит пробелы к одному виду.

    Модель охотно пишет дальше собственный следующий вопрос; всё после первого
    перевода строки или точки с запятой к ответу не относится. Резать нужно
    только вне строковых литералов — иначе `;` или перевод строки, случайно
    оказавшиеся частью значения (`WHERE name = 'a;b'`), обрежут ещё валидный
    запрос.
    """
    head = _cut_outside_quotes(text, "\n;")
    return _WHITESPACE.sub(" ", head).strip()


def _cut_outside_quotes(text: str, cut_chars: str) -> str:
    """Возвращает префикс `text` до первого символа из `cut_chars` вне кавычек.

    Отслеживает состояние «внутри одинарной/двойной кавычки» посимвольно и
    учитывает экранирование (`\\'`, `\\"`), чтобы не закрыть строковый литерал
    раньше времени.
    """
    quote: str | None = None
    escaped = False
    for index, char in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in "'\"":
            quote = char
        elif char in cut_chars:
            return text[:index]
    return text


@torch.inference_mode()
def generate_sql(
    model,
    tokenizer,
    prompts: list[str],
    *,
    device: torch.device,
    max_length: int,
    batch_size: int = 8,
    max_new_tokens: int = 128,
) -> list[str]:
    """Жадно генерирует SQL для готовых промптов.

    Промпты строит kanlora.data.collate.build_prompts — та же функция, что и
    при обучении: шаблон диалога токенизатора с add_generation_prompt=True.
    Каждый промпт усекается до max_length − max_new_tokens токенов функцией
    collate.truncate_prompt — тем же правилом и с той же токенизацией
    (add_special_tokens=False), что и в collate.encode при обучении, так что
    длинную схему модель видит на оценке так же, как видела на обучении.
    Генерация обрывается на tokenizer.eos_token_id (у Qwen2.5-Instruct это
    <|im_end|>, конец реплики), которым encode завершает ответ при обучении.
    """
    budget = max_length - max_new_tokens
    if budget <= 0:
        raise ValueError(
            f"max_new_tokens={max_new_tokens} не оставляет места промпту "
            f"при max_length={max_length}"
        )

    model.to(device)
    model.eval()
    model.config.use_cache = True

    generated: list[str] = []
    for start in range(0, len(prompts), batch_size):
        chunk = [
            truncate_prompt(tokenizer, prompt, budget)
            for prompt in prompts[start : start + batch_size]
        ]
        input_ids, attention_mask = _pad_left(chunk, tokenizer.pad_token_id)
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        continuation = output[:, input_ids.shape[1] :]
        generated.extend(
            normalize_sql(text)
            for text in tokenizer.batch_decode(continuation, skip_special_tokens=True)
        )
    return generated


def _pad_left(
    sequences: list[list[int]], pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Дополняет слева до самой длинной последовательности пакета."""
    width = max(len(sequence) for sequence in sequences)
    input_ids = [[pad_token_id] * (width - len(seq)) + seq for seq in sequences]
    attention_mask = [[0] * (width - len(seq)) + [1] * len(seq) for seq in sequences]
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
    )
