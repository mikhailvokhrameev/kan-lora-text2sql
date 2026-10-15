"""Цикл обучения, общий для всех трёх методов.

Один цикл на LoRA, DoRA и KAN-LoRA — по той же причине, что и один внедритель:
различие в коде обучения просочилось бы в сравнение методов, не выдав ошибки.

Накопление градиента делит функцию потерь на число батчей в группе
накопления, поэтому шаг эквивалентен обучению большим батчем — в том числе на
неполной последней группе эпохи, которая делится на свою фактическую длину.

Валидационная функция потерь (учебник §3.7; лекция 5 MLOps — «val loss на
шаге 0») считается до первого шага и в конце каждой эпохи на примерах train,
не вошедших в обучающую подвыборку. Проходы по валидации идут в режиме eval
без градиента: в функцию потерь эпохи и в долю активаций в сетке они не
входят, а их время вычитается из времени обучения. Градиентный чекпоинтинг включён, чтобы
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
    # [0] — до первого шага, затем по одному значению в конце каждой эпохи.
    validation_losses: list[float] = field(default_factory=list)
    # Норма градиента до обрезания, по одной на шаг оптимизатора: видно, как
    # часто срабатывает обрезание и как меняется масштаб градиента.
    grad_norms: list[float] = field(default_factory=list)
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
    values = [module.mean_fraction_inside_grid() for _, module in adapter_modules(model)]
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _group_size(index: int, batches: int, accumulation: int) -> int:
    """Число батчей в группе накопления, куда входит батч `index` (с 1).

    Последняя группа эпохи бывает неполной: деление её на `accumulation`
    занизило бы градиент последнего шага в `accumulation / хвост` раз.
    """
    full = batches - batches % accumulation
    return accumulation if index <= full else batches - full


@torch.no_grad()
def _validation_loss(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Средняя по батчам функция потерь, как и у эпохи; режим модели восстанавливается."""
    was_training = model.training
    model.eval()
    total, batches = 0.0, 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        total += model(**batch).loss.item()
        batches += 1
    model.train(was_training)
    return total / max(batches, 1)


def train(
    model: torch.nn.Module,
    dataset: list[dict[str, list[int]]],
    collator,
    optimizer_config: OptimizerConfig,
    train_config: TrainConfig,
    device: torch.device,
    validation_dataset: list[dict[str, list[int]]] | None = None,
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
    bundle = build_optimizer(model, optimizer_config)
    tracker = PeakMemoryTracker(device)
    tracker.reset()

    validation_loader = (
        DataLoader(
            validation_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            collate_fn=collator,
        )
        if validation_dataset
        else None
    )

    report = TrainReport(optimizer_coverage=bundle.coverage())
    validation_seconds = 0.0
    started = time.perf_counter()

    def validate() -> None:
        nonlocal validation_seconds
        if validation_loader is None:
            return
        validation_started = time.perf_counter()
        report.validation_losses.append(_validation_loss(model, validation_loader, device))
        validation_seconds += time.perf_counter() - validation_started

    validate()

    for epoch in range(train_config.epochs):
        epoch_total, epoch_batches = 0.0, 0
        bundle.zero_grad()
        # Доля активаций в сетке — среднее за эпоху, а не за весь прогон: на
        # графике по эпохам виден дрейф, и на последний микробатч значение не
        # опирается.
        for _, module in adapter_modules(model):
            module.reset_fraction_inside_grid()

        for index, batch in enumerate(loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            (loss / _group_size(index, len(loader), train_config.gradient_accumulation)).backward()

            step_loss = loss.item()
            epoch_total += step_loss
            epoch_batches += 1
            report.step_losses.append(step_loss)

            if index % train_config.log_every == 0:
                print(f"шаг {index}/{len(loader)} (эпоха {epoch + 1}): функция потерь {step_loss:.4f}")

            if index % train_config.gradient_accumulation == 0 or index == len(loader):
                norm = bundle.clip_grad_norm_(optimizer_config.max_grad_norm)
                report.grad_norms.append(norm.item())
                bundle.step()
                bundle.zero_grad()

        report.epoch_losses.append(epoch_total / max(epoch_batches, 1))
        fraction = _mean_fraction_inside_grid(model)
        if fraction is not None:
            report.fraction_inside_grid.append(fraction)
        validate()

        print(
            f"эпоха {epoch + 1}/{train_config.epochs}: "
            f"функция потерь {report.epoch_losses[-1]:.4f}"
            + (
                f", на валидации {report.validation_losses[-1]:.4f}"
                if validation_loader is not None
                else ""
            )
            + (f", доля активаций в сетке {fraction:.3f}" if fraction is not None else "")
        )

    report.seconds = time.perf_counter() - started - validation_seconds
    report.peak_memory_bytes = tracker.peak_bytes()
    return report
