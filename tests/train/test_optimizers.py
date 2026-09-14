"""Проверки разводки параметров по оптимизаторам.

Muon принимает только настоящие матрицы линейного отображения
(`AdapterLinear.matrix_parameters()`), а не любой двумерный тензор, и
отвергает остальное жёсткой ошибкой. Отсюда разное покрытие у трёх методов —
это результат работы, а не техническая деталь, поэтому разводка проверяется
поимённо: ни один параметр не потерян и не продублирован.
"""

import pytest
import torch
from torch import nn

from kanlora.adapters.base import AdapterConfig
from kanlora.adapters.dora import DoRALinear
from kanlora.adapters.kan_lora import KANLoRALinear
from kanlora.adapters.lora import LoRALinear
from kanlora.train.optimizers import OptimizerConfig, build_optimizer, split_by_matrix_role

IN_FEATURES, OUT_FEATURES, RANK = 12, 7, 4


def trainable(adapter) -> list[nn.Parameter]:
    return [p for p in adapter.parameters() if p.requires_grad]


@pytest.fixture(params=["lora", "dora", "kan_lora"])
def adapter(request):
    torch.manual_seed(0)
    base = nn.Linear(IN_FEATURES, OUT_FEATURES)
    config = AdapterConfig(rank=RANK)
    return {"lora": LoRALinear, "dora": DoRALinear, "kan_lora": KANLoRALinear}[
        request.param
    ](base, config)


def test_split_loses_and_duplicates_nothing(adapter) -> None:
    matrices, other = split_by_matrix_role(adapter)

    identifiers = [id(p) for p in matrices + other]
    assert sorted(identifiers) == sorted(id(p) for p in trainable(adapter))
    assert len(identifiers) == len(set(identifiers))


def test_split_puts_only_declared_matrix_parameters_in_the_first_group(adapter) -> None:
    matrices, other = split_by_matrix_role(adapter)
    matrix_ids = {id(p) for p in adapter.matrix_parameters()}

    assert {id(p) for p in matrices} == matrix_ids
    assert all(id(p) not in matrix_ids for p in other)


def test_lora_is_fully_covered_by_muon() -> None:
    """Обе матрицы LoRA — настоящие матрицы отображения, Muon покрывает метод целиком."""
    adapter = LoRALinear(nn.Linear(IN_FEATURES, OUT_FEATURES), AdapterConfig(rank=RANK))
    coverage = build_optimizer(adapter, OptimizerConfig(name="muon")).coverage()

    assert coverage["muon"] == RANK * (IN_FEATURES + OUT_FEATURES)
    assert coverage["adamw"] == 0


def test_dora_magnitude_falls_back_to_adamw() -> None:
    """Вектор модуля одномерен и не задаёт линейного отображения — Muon его не примет."""
    adapter = DoRALinear(nn.Linear(IN_FEATURES, OUT_FEATURES), AdapterConfig(rank=RANK))
    coverage = build_optimizer(adapter, OptimizerConfig(name="muon")).coverage()
    assert coverage["adamw"] == OUT_FEATURES


def test_kan_layer_parameters_fall_back_to_adamw() -> None:
    """Весь слой `kan` идёт в AdamW — включая двумерные `spline_scale`/`base_weight`.

    Они входят в вычисление поэлементно (`spline_scale * spline`), а не как
    матрица `y = W @ x`, и поэтому не входят в `matrix_parameters()`. Доля
    Muon у KAN-LoRA в точности равна доле A и B — той же, что и у LoRA/DoRA,
    а не большей, как было при разводке по голой двумерности тензора.
    """
    adapter = KANLoRALinear(nn.Linear(IN_FEATURES, OUT_FEATURES), AdapterConfig(rank=RANK))
    coverage = build_optimizer(adapter, OptimizerConfig(name="muon")).coverage()

    basis = adapter.config.grid_size + adapter.config.spline_order
    kan_layer_parameters = RANK * RANK * (basis + 2) + RANK
    assert coverage["adamw"] == kan_layer_parameters
    assert coverage["muon"] == RANK * (IN_FEATURES + OUT_FEATURES)


def test_adamw_mode_uses_a_single_optimizer(adapter) -> None:
    bundle = build_optimizer(adapter, OptimizerConfig(name="adamw"))
    assert len(bundle.optimizers) == 1
    assert bundle.coverage()["muon"] == 0


def test_muon_mode_creates_two_optimizers_when_needed() -> None:
    adapter = KANLoRALinear(nn.Linear(IN_FEATURES, OUT_FEATURES), AdapterConfig(rank=RANK))
    bundle = build_optimizer(adapter, OptimizerConfig(name="muon"))
    assert len(bundle.optimizers) == 2


def test_step_changes_every_trainable_parameter(adapter) -> None:
    """Разводка обязана двигать всё: параметр без оптимизатора остался бы мёртвым."""
    parameters = trainable(adapter)
    bundle = build_optimizer(adapter, OptimizerConfig(name="muon", learning_rate=0.1))

    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    before = [p.detach().clone() for p in parameters]
    bundle.step()

    for index, (old, new) in enumerate(zip(before, parameters, strict=True)):
        assert not torch.equal(old, new), f"параметр {index} не сдвинулся"


def test_clipping_bounds_the_gradient_norm(adapter) -> None:
    parameters = trainable(adapter)
    bundle = build_optimizer(adapter, OptimizerConfig())
    for parameter in parameters:
        parameter.grad = torch.full_like(parameter, 100.0)

    bundle.clip_grad_norm_(1.0)
    total = torch.norm(torch.stack([p.grad.norm() for p in parameters]))
    assert total.item() == pytest.approx(1.0, abs=1e-4)


def test_zero_grad_clears_gradients(adapter) -> None:
    parameters = trainable(adapter)
    bundle = build_optimizer(adapter, OptimizerConfig())
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)

    bundle.zero_grad()
    assert all(p.grad is None for p in parameters)


def test_unknown_optimizer_is_rejected(adapter) -> None:
    with pytest.raises(KeyError):
        build_optimizer(adapter, OptimizerConfig(name="lion"))
