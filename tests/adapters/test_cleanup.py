"""Эталоны до удаления исследовательской диагностики."""

import sqlite3
from pathlib import Path

import pytest
import torch
from torch import nn

from kanlora.adapters.base import AdapterConfig
from kanlora.adapters.dora import DoRALinear
from kanlora.adapters.inject import adapter_modules, inject_adapters
from kanlora.adapters.kan_lora import KANLoRALinear
from kanlora.adapters.lora import LoRALinear
from kanlora.data.spider import SPIDER_LAYOUT
from kanlora.eval.exact_match import exact_match
from kanlora.eval.execution import execution_accuracy
from kanlora.eval.generate import generate_sql


@pytest.mark.parametrize(("cls", "output", "gradients"), [
    (LoRALinear, [[0.5349944176071559, 0.3405266139665804], [0.7615371089191671, -0.14477876896530534]],
     {"lora_a": [[0.5801800515500137, -0.1982170927720032]], "lora_b": [[0.44825740589876], [0.04979447421940655]]}),
    (DoRALinear, [[0.5253213016035675, 0.34514542023833134], [0.6980129481480135, -0.1509491821070223]],
     {"lora_a": [[0.4074547396629698, -0.2255656234828415]], "lora_b": [[0.32133427029229367], [0.04998301897429102]], "magnitude": [0.8745910833116451, 0.3544526774450483]}),
    (KANLoRALinear, [[0.5349944176071559, 0.3405266139665804], [0.7615371089191671, -0.1447787689653053]],
     {"lora_a": [[0.5801800515500137, -0.1982170927720032]], "lora_b": [[0.44825740589876006], [0.04979447421940654]], "kan.spline_coefficients": [[[0.0, 0.12595025779712327, 1.3147113841856959, 0.35007360385029834, 0.0]]], "kan.spline_scale": [[0.14941556403544998]], "kan.base_weight": [[0.07785475492042321]], "kan.input_scale": [0.0]}),
])
def test_cleanup_preserves_adapter_outputs_and_gradients(cls, output, gradients) -> None:
    torch.manual_seed(12)
    adapter = cls(nn.Linear(2, 2, dtype=torch.float64), AdapterConfig(rank=1, alpha=2, grid_size=3, spline_order=2))
    with torch.no_grad():
        adapter.lora_b.fill_(0.3)
    actual = adapter(torch.tensor([[0.2, -0.4], [0.5, 0.3]], dtype=torch.float64))
    actual.square().sum().backward()
    assert torch.allclose(actual, torch.tensor(output, dtype=torch.float64), atol=1e-10)
    parameters = {n: p for n, p in adapter.named_parameters() if p.requires_grad}
    assert parameters.keys() == gradients.keys()
    for name, expected in gradients.items():
        assert torch.allclose(parameters[name].grad, torch.tensor(expected, dtype=torch.float64), atol=1e-10)


def test_cleanup_preserves_predictions_and_metrics(tiny_causal_lm, tiny_tokenizer, tmp_path: Path) -> None:
    inject_adapters(tiny_causal_lm, "kan_lora", AdapterConfig(rank=4))
    with torch.no_grad():
        for _, adapter in adapter_modules(tiny_causal_lm):
            adapter.lora_b.fill_(1.0)

    predictions = generate_sql(
        tiny_causal_lm, tiny_tokenizer, ["SELECT", "FROM", "WHERE"],
        device=torch.device("cpu"), batch_size=3, max_new_tokens=4, max_length=64,
    )
    # Эталоны получены на исходном коде d13f4a3. При нулевой B первый и
    # третий ответы отличаются — генерация действительно зависит от адаптера.
    assert predictions == ["$", "", "$$$\x14"]
    with torch.no_grad():
        for _, adapter in adapter_modules(tiny_causal_lm):
            adapter.lora_b.zero_()
    baseline = generate_sql(
        tiny_causal_lm, tiny_tokenizer, ["SELECT", "FROM", "WHERE"],
        device=torch.device("cpu"), batch_size=3, max_new_tokens=4, max_length=64,
    )
    assert baseline == ["\x14\x14\x14\x14", "", "\x05\x05\x05\x05"]
    assert predictions != baseline

    path = tmp_path / "database" / "concert_singer"
    path.mkdir(parents=True)
    with sqlite3.connect(path / "concert_singer.sqlite") as connection:
        connection.executescript("CREATE TABLE singer (name TEXT); INSERT INTO singer VALUES ('Alice');")
    gold = ["SELECT count(*) FROM singer"] * 2
    sql_predictions = ["SELECT count(*) FROM singer", "SELECT name FROM singer"]
    tables = Path(__file__).parents[1] / "data" / "fixtures" / "spider_mini" / "tables.json"
    assert exact_match(gold, sql_predictions, ["concert_singer"] * 2, tables) == 0.5
    assert execution_accuracy(gold, sql_predictions, ["concert_singer"] * 2, tmp_path, SPIDER_LAYOUT) == 0.5
