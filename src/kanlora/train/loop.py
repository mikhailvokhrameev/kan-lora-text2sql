"""Цикл обучения, общий для всех трёх методов.

Один цикл на LoRA, DoRA и KAN-LoRA — по той же причине, что и один внедритель:
различие в коде обучения просочилось бы в сравнение методов, не выдав ошибки.

Накопление градиента делит функцию потерь на число накоплений, поэтому шаг
эквивалентен обучению большим батчем. Градиентный чекпоинтинг включён, чтобы
модель в fp32 помещалась в 11 ГБ.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader

from kanlora.adapters.inject import adapter_modules
from kanlora.train.memory import PeakMemoryTracker
from kanlora.train.optimizers import OptimizerConfig, build_optimizer

__all__ = ["TrainConfig", "TrainReport", "set_seed", "train"]


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 2
    batch_size: int = 1
    gradient_accumulation: int = 8
    seed: int = 0
    gradient_checkpointing: bool = True
    log_every: int = 20
    # Нормы обрезания градиента здесь намеренно нет: она живёт в OptimizerConfig
    # и берётся циклом оттуда. Два источника одного значения разошлись бы молча,
    # изменив методологию для части прогонов и не выдав ошибки.


@dataclass
class TrainReport:
    step_losses: list[float] = field(default_factory=list)
    epoch_losses: list[float] = field(default_factory=list)
    fraction_inside_grid: list[float] = field(default_factory=list)
    peak_memory_bytes: int = 0
    seconds: float = 0.0
    optimizer_coverage: dict[str, int] = field(default_factory=dict)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _mean_fraction_inside_grid(model: torch.nn.Module) -> float | None:
    values = [module.last_fraction_inside_grid() for _, module in adapter_modules(model)]
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def train(
    model: torch.nn.Module,
    dataset: list[dict[str, list[int]]],
    collator,
    optimizer_config: OptimizerConfig,
    train_config: TrainConfig,
    device: torch.device,
) -> TrainReport:
    set_seed(train_config.seed)
    model.to(device)
    model.train()

    if train_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # Явно, а не полагаясь на автоматику transformers: на пришпиленной нижней
        # границе версии (4.46) она включается только под HF PEFT, а этот проект
        # свои адаптеры через PEFT не заводит — без вызова градиент к адаптерам
        # не доходит через чекпоинтинг, если вход не требует grad.
        model.enable_input_require_grads()
        model.config.use_cache = False

    loader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(train_config.seed),
    )
    bundle = build_optimizer(model.parameters(), optimizer_config)
    tracker = PeakMemoryTracker(device)
    tracker.reset()

    report = TrainReport(optimizer_coverage=bundle.coverage())
    started = time.perf_counter()

    for epoch in range(train_config.epochs):
        epoch_total, epoch_batches = 0.0, 0
        bundle.zero_grad()

        for index, batch in enumerate(loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            (loss / train_config.gradient_accumulation).backward()

            step_loss = loss.item()
            epoch_total += step_loss
            epoch_batches += 1
            report.step_losses.append(step_loss)

            if index % train_config.log_every == 0:
                print(f"шаг {index}/{len(loader)} (эпоха {epoch + 1}): функция потерь {step_loss:.4f}")

            if index % train_config.gradient_accumulation == 0 or index == len(loader):
                bundle.clip_grad_norm_(optimizer_config.max_grad_norm)
                bundle.step()
                bundle.zero_grad()

        report.epoch_losses.append(epoch_total / max(epoch_batches, 1))
        fraction = _mean_fraction_inside_grid(model)
        if fraction is not None:
            report.fraction_inside_grid.append(fraction)

        print(
            f"эпоха {epoch + 1}/{train_config.epochs}: "
            f"функция потерь {report.epoch_losses[-1]:.4f}"
            + (f", доля активаций в сетке {fraction:.3f}" if fraction is not None else "")
        )

    report.seconds = time.perf_counter() - started
    report.peak_memory_bytes = tracker.peak_bytes()
    return report
