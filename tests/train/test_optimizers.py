"""Проверка AdamW на обучаемых параметрах всех адаптеров."""

import pytest
import torch
from torch import nn

from kanlora.adapters.base import AdapterConfig
from kanlora.adapters.dora import DoRALinear
from kanlora.adapters.kan_lora import KANLoRALinear
from kanlora.adapters.lora import LoRALinear
from kanlora.train.optimizers import OptimizerConfig, build_optimizer


@pytest.fixture(params=[LoRALinear, DoRALinear, KANLoRALinear])
def adapter(request):
    torch.manual_seed(0)
    return request.param(nn.Linear(12, 7), AdapterConfig(rank=4))


def test_adamw_covers_every_trainable_parameter_once(adapter) -> None:
    optimizer = build_optimizer(adapter, OptimizerConfig())
    assert isinstance(optimizer, torch.optim.AdamW)
    actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert sorted(actual) == sorted(id(p) for p in adapter.parameters() if p.requires_grad)
    assert len(actual) == len(set(actual))


def test_step_updates_trainable_parameters_and_clears_gradients(adapter) -> None:
    optimizer = build_optimizer(adapter, OptimizerConfig(learning_rate=0.1))
    parameters = [p for p in adapter.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in parameters]
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert all(not torch.equal(old, new) for old, new in zip(before, parameters, strict=True))
    assert all(parameter.grad is None for parameter in parameters)
