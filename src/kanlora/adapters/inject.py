# src/kanlora/adapters/inject.py
"""Замена nn.Linear на адаптер. Единая для всех трёх методов.

Один внедритель на LoRA, DoRA и KAN-LoRA существует ради сопоставимости:
если бы каждый метод правил модель своим кодом, различие в наборе затронутых
слоёв просочилось бы в результаты, не выдав ошибки.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from torch import nn

from kanlora.adapters.base import AdapterConfig, AdapterLinear, freeze
from kanlora.adapters.dora import DoRALinear
from kanlora.adapters.kan_lora import KANLoRALinear
from kanlora.adapters.lora import LoRALinear

__all__ = [
    "ADAPTER_TYPES",
    "DEFAULT_TARGET_MODULES",
    "InjectionReport",
    "adapter_modules",
    "inject_adapters",
]

ADAPTER_TYPES: dict[str, type[AdapterLinear]] = {
    "lora": LoRALinear,
    "dora": DoRALinear,
    "kan_lora": KANLoRALinear,
}

# Все семь проекций блока Qwen2: внимание целиком и все три матрицы MLP.
DEFAULT_TARGET_MODULES: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)


@dataclass(frozen=True)
class InjectionReport:
    method: str
    replaced: tuple[str, ...]
    trainable_parameters: int
    total_parameters: int


def inject_adapters(
    model: nn.Module,
    method: str,
    config: AdapterConfig,
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
) -> InjectionReport:
    """Замораживает модель и подменяет целевые nn.Linear адаптерами. Правит на месте."""
    adapter_type = ADAPTER_TYPES[method]
    freeze(model)

    wanted = set(target_modules)
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.split(".")[-1] in wanted
    ]
    found = {name.split(".")[-1] for name, _ in targets}
    missing = wanted - found
    if missing:
        raise ValueError(f"целевой слой не найден в модели: {', '.join(sorted(missing))}")

    for name, module in targets:
        parent_name, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attribute, adapter_type(module, config))

    return InjectionReport(
        method=method,
        replaced=tuple(sorted(name for name, _ in targets)),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_parameters=sum(p.numel() for p in model.parameters()),
    )


def adapter_modules(model: nn.Module) -> list[tuple[str, AdapterLinear]]:
    """Все внедрённые адаптеры в устойчивом порядке.

    Порядок фиксирован, потому что по нему собирается диагностика сплайнов:
    строки в таблицах анализа обязаны совпадать между прогонами.
    """
    return sorted(
        (
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, AdapterLinear)
        ),
        key=lambda item: item[0],
    )
