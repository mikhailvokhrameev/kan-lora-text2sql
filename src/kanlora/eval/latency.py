# src/kanlora/eval/latency.py
"""Задержка генерации со слиянием адаптера и без.

LoRA и DoRA после обучения складываются с весами: W + BA считается один раз,
и на инференсе метод не стоит ничего. KAN-LoRA не сливается в принципе —
поправка B * phi(A x) зависит от входа нелинейно, — поэтому платит задержкой
на каждом токене всегда. Этой цены нет в задании, и она измеряется здесь.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from kanlora.adapters.base import AdapterLinear
from kanlora.adapters.inject import adapter_modules

__all__ = ["LatencyReport", "measure_latency", "merge_adapters"]


@dataclass(frozen=True)
class LatencyReport:
    seconds_per_token: float
    total_seconds: float
    tokens: int
    merged: bool


def merge_adapters(model: torch.nn.Module) -> int:
    """Заменяет каждый адаптер обычным nn.Linear с поглощённой поправкой."""
    merged = 0
    for name, module in adapter_modules(model):
        if not isinstance(module, AdapterLinear):
            continue
        replacement = module.merge()  # на KAN-LoRA бросит NotImplementedError
        parent_name, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attribute, replacement)
        merged += 1
    return merged


@torch.inference_mode()
def measure_latency(
    model,
    tokenizer,
    prompt: str,
    *,
    device: torch.device,
    max_new_tokens: int = 64,
    repeats: int = 3,
    warmup: int = 1,
    merged: bool = False,
) -> LatencyReport:
    """Среднее время жадной генерации фиксированного числа токенов.

    Прогрев обязателен: первый вызов включает выделение памяти и подбор ядер,
    и без него замер сравнивал бы разогрев, а не метод.
    """
    model.to(device)
    model.eval()
    encoded = tokenizer([prompt], return_tensors="pt").to(device)

    def generate() -> None:
        model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,  # ровно столько токенов во всех замерах
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
        )

    for _ in range(warmup):
        generate()
    if device.type == "cuda":
        torch.cuda.synchronize()

    started = time.perf_counter()
    for _ in range(repeats):
        generate()
    if device.type == "cuda":
        torch.cuda.synchronize()
    total = (time.perf_counter() - started) / repeats

    return LatencyReport(
        seconds_per_token=total / max_new_tokens,
        total_seconds=total,
        tokens=max_new_tokens,
        merged=merged,
    )
