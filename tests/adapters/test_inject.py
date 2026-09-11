# tests/adapters/test_inject.py
"""Проверки внедрения адаптеров.

Внедритель один на три метода. Тесты закрепляют то, от чего зависит честность
сравнения: одинаковый набор затронутых слоёв, замороженная основа и нулевая
поправка сразу после внедрения — модель обязана выдавать ровно то же, что и до.
"""

import pytest
import torch
from torch import nn

from kanlora.adapters.base import AdapterConfig, AdapterLinear
from kanlora.adapters.inject import (
    DEFAULT_TARGET_MODULES,
    adapter_modules,
    inject_adapters,
)

METHODS = ["lora", "dora", "kan_lora"]
LAYERS = 2


@pytest.mark.parametrize("method", METHODS)
def test_replaces_every_target_projection(tiny_causal_lm, method: str) -> None:
    report = inject_adapters(tiny_causal_lm, method, AdapterConfig(rank=4))

    assert len(report.replaced) == LAYERS * len(DEFAULT_TARGET_MODULES)
    for name in report.replaced:
        assert name.split(".")[-1] in DEFAULT_TARGET_MODULES


@pytest.mark.parametrize("method", METHODS)
def test_all_methods_touch_the_same_slots(tiny_causal_lm, method: str) -> None:
    """Разные методы обязаны править один и тот же список слоёв."""
    report = inject_adapters(tiny_causal_lm, method, AdapterConfig(rank=4))
    assert set(report.replaced) == {
        f"model.layers.{index}.{block}.{projection}"
        for index in range(LAYERS)
        for block, projection in (
            ("self_attn", "q_proj"), ("self_attn", "k_proj"),
            ("self_attn", "v_proj"), ("self_attn", "o_proj"),
            ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj"),
        )
    }


@pytest.mark.parametrize("method", METHODS)
def test_output_is_unchanged_right_after_injection(tiny_causal_lm, method: str) -> None:
    """Нулевая поправка на уровне всей модели, а не отдельного слоя."""
    tokens = torch.randint(0, 64, (2, 9))
    with torch.no_grad():
        before = tiny_causal_lm(tokens).logits.clone()
        inject_adapters(tiny_causal_lm, method, AdapterConfig(rank=4))
        after = tiny_causal_lm(tokens).logits

    assert torch.allclose(before, after, atol=1e-5)


@pytest.mark.parametrize("method", METHODS)
def test_only_adapter_parameters_are_trainable(tiny_causal_lm, method: str) -> None:
    report = inject_adapters(tiny_causal_lm, method, AdapterConfig(rank=4))

    trainable = {
        name for name, parameter in tiny_causal_lm.named_parameters() if parameter.requires_grad
    }
    assert trainable, "адаптеры обязаны быть обучаемыми"
    for name in trainable:
        assert any(name.startswith(f"{slot}.") for slot in report.replaced), name
        assert ".base." not in name, name


@pytest.mark.parametrize("method", METHODS)
def test_report_counts_match_the_model(tiny_causal_lm, method: str) -> None:
    report = inject_adapters(tiny_causal_lm, method, AdapterConfig(rank=4))

    assert report.trainable_parameters == sum(
        p.numel() for p in tiny_causal_lm.parameters() if p.requires_grad
    )
    assert report.total_parameters == sum(p.numel() for p in tiny_causal_lm.parameters())
    assert report.method == method


@pytest.mark.parametrize("method", METHODS)
def test_adapter_modules_enumerates_what_was_injected(tiny_causal_lm, method: str) -> None:
    report = inject_adapters(tiny_causal_lm, method, AdapterConfig(rank=4))
    found = adapter_modules(tiny_causal_lm)

    assert [name for name, _ in found] == sorted(report.replaced)
    assert all(isinstance(module, AdapterLinear) for _, module in found)


def test_kan_lora_costs_a_couple_of_percent_more_parameters(tiny_causal_lm) -> None:
    """Число, которое идёт в таблицу вместо отдельных прогонов LoRA с большим рангом."""
    import copy

    reference = copy.deepcopy(tiny_causal_lm)
    lora = inject_adapters(reference, "lora", AdapterConfig(rank=4))
    kan = inject_adapters(tiny_causal_lm, "kan_lora", AdapterConfig(rank=4))

    assert kan.trainable_parameters > lora.trainable_parameters


def test_unknown_method_is_rejected(tiny_causal_lm) -> None:
    with pytest.raises(KeyError):
        inject_adapters(tiny_causal_lm, "qlora", AdapterConfig())


def test_missing_target_module_is_rejected(tiny_causal_lm) -> None:
    """Опечатка в имени проекции обязана падать, а не тихо внедрять меньше слоёв."""
    with pytest.raises(ValueError, match="не найден"):
        inject_adapters(tiny_causal_lm, "lora", AdapterConfig(), target_modules=("qkv_proj",))
