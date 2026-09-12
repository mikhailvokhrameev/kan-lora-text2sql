"""Проверки цикла обучения.

Сходимость юнит-тестом не проверяется. Вместо неё — переобучение двадцати
примеров: функция потерь обязана упасть почти до нуля. Этот тест ловит
подавляющее большинство ошибок цикла — неверную маску, оторванный граф
градиентов, замороженные не те параметры, — ни одна из которых не даёт
исключения сама по себе.
"""

import pytest
import torch

from kanlora.adapters.base import AdapterConfig
from kanlora.adapters.inject import inject_adapters
from kanlora.data.collate import IGNORE_INDEX, Collator
from kanlora.train.loop import TrainConfig, set_seed, train
from kanlora.train.optimizers import OptimizerConfig

from conftest import unfreeze_embeddings, unfreeze_norms

VOCAB_SIZE = 64
DEVICE = torch.device("cpu")


def twenty_examples(seed: int = 0) -> list[dict[str, list[int]]]:
    """Двадцать пар «промпт -> цель» с маской по цели, как в настоящих данных."""
    generator = torch.Generator().manual_seed(seed)
    dataset = []
    for _ in range(20):
        prompt = torch.randint(2, VOCAB_SIZE, (6,), generator=generator).tolist()
        target = torch.randint(2, VOCAB_SIZE, (4,), generator=generator).tolist()
        dataset.append(
            {
                "input_ids": prompt + target,
                "labels": [IGNORE_INDEX] * len(prompt) + target,
            }
        )
    return dataset


def test_set_seed_makes_initialization_reproducible() -> None:
    set_seed(3)
    first = torch.randn(5)
    set_seed(3)
    assert torch.equal(first, torch.randn(5))


@pytest.mark.slow
@pytest.mark.parametrize("method", ["lora", "dora", "kan_lora"])
def test_overfits_twenty_examples(tiny_causal_lm, method: str) -> None:
    """ГЛАВНЫЙ ТЕСТ ЦИКЛА. Все три метода обязаны выучить двадцать примеров наизусть.

    RMSNorm и привязанные эмбеддинги размораживаются поверх адаптеров только
    здесь (см. unfreeze_norms/unfreeze_embeddings в conftest.py): у случайно
    инициализированной игрушечной модели с полностью замороженной основой
    функция потерь упирается в потолок около 3.5–4.0 для всех трёх методов
    одинаково — это ограничение самой фикстуры (см. её докстринг), а не
    адаптеров. Скорость обучения и число эпох здесь ниже и больше значений
    по умолчанию — подобраны эмпирически именно под эту проверку сходимости,
    а не совпадают с гиперпараметрами реальных экспериментов.
    """
    inject_adapters(tiny_causal_lm, method, AdapterConfig(rank=4))
    unfreeze_norms(tiny_causal_lm)
    unfreeze_embeddings(tiny_causal_lm)

    report = train(
        model=tiny_causal_lm,
        dataset=twenty_examples(),
        collator=Collator(pad_token_id=0),
        optimizer_config=OptimizerConfig(name="adamw", learning_rate=0.005),
        train_config=TrainConfig(
            epochs=150, batch_size=4, gradient_accumulation=1,
            gradient_checkpointing=False, seed=0,
        ),
        device=DEVICE,
    )

    assert report.epoch_losses[-1] < 0.1, f"функция потерь застряла: {report.epoch_losses[-1]}"
    assert report.epoch_losses[-1] < report.epoch_losses[0]


@pytest.mark.slow
def test_gradient_checkpointing_preserves_adapter_gradient(tiny_causal_lm) -> None:
    """С включённым gradient_checkpointing адаптер обязан обучаться так же, как без него.

    Регрессионная защита для входных эмбеддингов без requires_grad=True при
    заморожённой основе: без явного enable_input_require_grads() градиент до
    LoRA может не дойти при чекпоинтинге (см. docs/train-loop.md). В связке с
    transformers, установленной в этом окружении, баг уже не воспроизводится
    (апстрим сам вызывает enable_input_require_grads() для causal LM), поэтому
    тест служит защитой от отката/понижения версии transformers, а не тестом,
    падающим прямо сейчас.
    """
    inject_adapters(tiny_causal_lm, "lora", AdapterConfig(rank=4))
    lora_b_name = next(
        name for name, parameter in tiny_causal_lm.named_parameters()
        if name.endswith("lora_b") and parameter.requires_grad
    )
    before = dict(tiny_causal_lm.named_parameters())[lora_b_name].detach().clone()

    report = train(
        model=tiny_causal_lm,
        dataset=twenty_examples(),
        collator=Collator(pad_token_id=0),
        optimizer_config=OptimizerConfig(learning_rate=0.01),
        train_config=TrainConfig(epochs=2, batch_size=4, gradient_accumulation=1,
                                 gradient_checkpointing=True),
        device=DEVICE,
    )

    after = dict(tiny_causal_lm.named_parameters())[lora_b_name]
    assert not torch.equal(before, after), "lora_b не изменился при gradient_checkpointing=True"
    assert report.epoch_losses[-1] < report.epoch_losses[0]


@pytest.mark.slow
def test_only_adapter_parameters_change(tiny_causal_lm) -> None:
    """Сверка того, что обучаются ровно адаптеры: основа обязана остаться прежней."""
    inject_adapters(tiny_causal_lm, "lora", AdapterConfig(rank=4))
    frozen = {
        name: parameter.detach().clone()
        for name, parameter in tiny_causal_lm.named_parameters()
        if not parameter.requires_grad
    }

    train(
        model=tiny_causal_lm,
        dataset=twenty_examples(),
        collator=Collator(pad_token_id=0),
        optimizer_config=OptimizerConfig(learning_rate=0.01),
        train_config=TrainConfig(epochs=2, batch_size=4, gradient_accumulation=1,
                                 gradient_checkpointing=False),
        device=DEVICE,
    )

    for name, before in frozen.items():
        after = dict(tiny_causal_lm.named_parameters())[name]
        assert torch.equal(before, after), f"замороженный параметр изменился: {name}"


@pytest.mark.slow
def test_reports_fraction_inside_grid_per_epoch(tiny_causal_lm) -> None:
    """Доля активаций внутри сетки логируется каждую эпоху — это готовый график."""
    inject_adapters(tiny_causal_lm, "kan_lora", AdapterConfig(rank=4))

    report = train(
        model=tiny_causal_lm,
        dataset=twenty_examples(),
        collator=Collator(pad_token_id=0),
        optimizer_config=OptimizerConfig(learning_rate=0.01),
        train_config=TrainConfig(epochs=3, batch_size=4, gradient_accumulation=1,
                                 gradient_checkpointing=False),
        device=DEVICE,
    )

    assert len(report.fraction_inside_grid) == 3
    assert all(0.0 <= value <= 1.0 for value in report.fraction_inside_grid)


@pytest.mark.slow
def test_linear_methods_report_no_grid_statistics(tiny_causal_lm) -> None:
    inject_adapters(tiny_causal_lm, "lora", AdapterConfig(rank=4))

    report = train(
        model=tiny_causal_lm,
        dataset=twenty_examples(),
        collator=Collator(pad_token_id=0),
        optimizer_config=OptimizerConfig(learning_rate=0.01),
        train_config=TrainConfig(epochs=2, batch_size=4, gradient_accumulation=1,
                                 gradient_checkpointing=False),
        device=DEVICE,
    )

    assert report.fraction_inside_grid == []


@pytest.mark.slow
def test_gradient_accumulation_matches_the_larger_batch(tiny_causal_lm) -> None:
    """Накопление градиента обязано быть эквивалентно большему батчу.

    Забытое деление на число накоплений завышает шаг ровно во столько же раз
    и не выдаёт себя ничем, кроме худшего качества.
    """
    import copy

    reference = copy.deepcopy(tiny_causal_lm)
    inject_adapters(tiny_causal_lm, "lora", AdapterConfig(rank=4))
    inject_adapters(reference, "lora", AdapterConfig(rank=4))
    reference.load_state_dict(tiny_causal_lm.state_dict())

    common = {
        "dataset": twenty_examples(),
        "collator": Collator(pad_token_id=0),
        "optimizer_config": OptimizerConfig(learning_rate=0.01),
        "device": DEVICE,
    }
    big = train(model=tiny_causal_lm, train_config=TrainConfig(
        epochs=1, batch_size=4, gradient_accumulation=1, gradient_checkpointing=False), **common)
    split = train(model=reference, train_config=TrainConfig(
        epochs=1, batch_size=2, gradient_accumulation=2, gradient_checkpointing=False), **common)

    assert big.epoch_losses[0] == pytest.approx(split.epoch_losses[0], rel=1e-3)


@pytest.mark.slow
def test_logs_loss_every_log_every_steps(tiny_causal_lm, capsys) -> None:
    """Построчный лог должен печататься каждые log_every шагов, а не только за эпоху."""
    inject_adapters(tiny_causal_lm, "lora", AdapterConfig(rank=4))

    train(
        model=tiny_causal_lm,
        dataset=twenty_examples(),
        collator=Collator(pad_token_id=0),
        optimizer_config=OptimizerConfig(learning_rate=0.01),
        train_config=TrainConfig(
            epochs=1, batch_size=4, gradient_accumulation=1,
            gradient_checkpointing=False, log_every=2,
        ),
        device=DEVICE,
    )

    lines = capsys.readouterr().out.splitlines()
    step_lines = [line for line in lines if line.startswith("шаг ")]

    # 20 примеров, batch_size=4 -> 5 шагов за эпоху; log_every=2 -> шаги 2 и 4.
    assert len(step_lines) == 2
    assert step_lines[0].startswith("шаг 2/5")
    assert step_lines[1].startswith("шаг 4/5")


@pytest.mark.slow
def test_report_carries_timing_memory_and_coverage(tiny_causal_lm) -> None:
    inject_adapters(tiny_causal_lm, "lora", AdapterConfig(rank=4))

    report = train(
        model=tiny_causal_lm,
        dataset=twenty_examples(),
        collator=Collator(pad_token_id=0),
        optimizer_config=OptimizerConfig(name="muon", learning_rate=0.01),
        train_config=TrainConfig(epochs=1, batch_size=4, gradient_accumulation=1,
                                 gradient_checkpointing=False),
        device=DEVICE,
    )

    assert report.seconds > 0
    assert report.peak_memory_bytes == 0  # на процессоре замера нет, и это честно видно
    assert report.optimizer_coverage["muon"] > 0
    assert len(report.step_losses) == 5
