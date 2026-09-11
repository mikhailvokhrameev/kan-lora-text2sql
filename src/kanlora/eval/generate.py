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

__all__ = ["generate_sql", "normalize_sql"]

_WHITESPACE = re.compile(r"\s+")


def normalize_sql(text: str) -> str:
    """Обрезает продолжение после запроса и приводит пробелы к одному виду.

    Модель охотно пишет дальше собственный следующий вопрос; всё после первого
    перевода строки или точки с запятой к ответу не относится.
    """
    head = text.split("\n", 1)[0].split(";", 1)[0]
    return _WHITESPACE.sub(" ", head).strip()


@torch.inference_mode()
def generate_sql(
    model,
    tokenizer,
    prompts: list[str],
    *,
    device: torch.device,
    batch_size: int = 8,
    max_new_tokens: int = 128,
) -> list[str]:
    model.to(device)
    model.eval()
    model.config.use_cache = True

    previous_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        generated: list[str] = []
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
            encoded = tokenizer(chunk, return_tensors="pt", padding=True).to(device)

            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            continuation = output[:, encoded["input_ids"].shape[1] :]
            generated.extend(
                normalize_sql(text)
                for text in tokenizer.batch_decode(continuation, skip_special_tokens=True)
            )
        return generated
    finally:
        tokenizer.padding_side = previous_side
