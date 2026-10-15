"""Проверки сквозного прогона без сети.

Настоящая модель и наборы данных здесь не загружаются: проверяется только
ветвление по методу — базовая модель без адаптера (`method: none`) не получает
адаптеров и не обучается, а оценивается тем же путём, что и обученные методы.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from kanlora.adapters.base import AdapterLinear
from kanlora.config import apply_overrides, load_config

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_experiment.py"
BASE = ROOT / "configs" / "base.yaml"


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("run_experiment", SCRIPT)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules["run_experiment"] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def test_command_line_accepts_method_none(module, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--method", "none"])
    assert module.parse_arguments().method == "none"


def test_baseline_is_neither_adapted_nor_trained(module, tiny_causal_lm, monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("базовая модель не должна ни получать адаптеры, ни обучаться")

    monkeypatch.setattr(module, "inject_adapters", forbidden)
    monkeypatch.setattr(module, "build_dataset", forbidden)
    monkeypatch.setattr(module, "train", forbidden)
    before = {name: tensor.clone() for name, tensor in tiny_causal_lm.state_dict().items()}

    config = apply_overrides(load_config(BASE), method="none")
    injection, report, truncated = module.adapt_and_train(
        config, tiny_causal_lm, tokenizer=None, train_examples=[], schemas={},
        device=torch.device("cpu"),
    )

    assert injection == {
        "trainable_parameters": None,
        "total_parameters": sum(p.numel() for p in tiny_causal_lm.parameters()),
        "replaced": [],
    }
    assert report is None and truncated is None
    assert not any(isinstance(m, AdapterLinear) for m in tiny_causal_lm.modules())
    after = tiny_causal_lm.state_dict()
    assert all(torch.equal(before[name], after[name]) for name in before)
