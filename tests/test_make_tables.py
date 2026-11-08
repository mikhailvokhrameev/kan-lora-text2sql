"""Проверки сборки таблиц.

Выводы формулируются только по различиям, выходящим за разброс по зерну,
поэтому таблица обязана показывать разброс рядом со средним. Таблица,
показывающая одно среднее, скрывала бы ровно то, ради чего заложены три зерна.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_tables.py"


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("make_tables", SCRIPT)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules["make_tables"] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def card(
    method: str,
    dataset: str,
    seed: int,
    exact: float,
    execution: float | None = 0.5,
    optimizer: str = "adamw",
    learn_input_scale: bool = True,
) -> dict:
    return {
        "run_id": f"{method}-{dataset}-{optimizer}-seed{seed}",
        "method": method, "dataset": dataset, "seed": seed, "optimizer": optimizer,
        "config": {
            "adapter": {"learn_input_scale": learn_input_scale},
            "optimizer": {"name": optimizer},
        },
        "metrics": {"exact_match": exact, "execution_accuracy": execution},
        "peak_memory_bytes": 4 * 2**30, "train_seconds": 3600.0,
        "trainable_parameters": 100_000,
    }


@pytest.fixture
def cards() -> list[dict]:
    return [
        card("lora", "spider", 0, 0.40), card("lora", "spider", 1, 0.42),
        card("lora", "spider", 2, 0.44),
        card("kan_lora", "spider", 0, 0.45), card("kan_lora", "spider", 1, 0.43),
        card("kan_lora", "spider", 2, 0.47),
    ]


def test_loads_every_card(module, cards, tmp_path: Path) -> None:
    for item in cards:
        (tmp_path / f"{item['run_id']}.json").write_text(json.dumps(item), encoding="utf-8")
    assert len(module.load_cards(tmp_path)) == len(cards)


def test_aggregate_reports_mean_and_spread(module, cards) -> None:
    aggregated = module.aggregate(cards)
    mean, spread = aggregated[("lora", "spider")]["exact_match"]

    assert mean == pytest.approx(0.42)
    assert spread == pytest.approx(0.04)  # размах: 0.44 - 0.40


def test_aggregate_rejects_removed_optimizer_in_old_card(module) -> None:
    with pytest.raises(ValueError, match="muon"):
        module.aggregate([card("lora", "spider", 0, 0.4, optimizer="muon")])


def test_aggregate_rejects_old_input_scale_ablation(module) -> None:
    with pytest.raises(ValueError, match="learn_input_scale"):
        module.aggregate([card("kan_lora", "spider", 0, 0.4, learn_input_scale=False)])


def test_single_seed_has_zero_spread(module) -> None:
    aggregated = module.aggregate([card("dora", "pauq", 0, 0.3)])
    assert aggregated[("dora", "pauq")]["exact_match"][1] == pytest.approx(0.0)


def test_absent_execution_accuracy_does_not_become_zero(module) -> None:
    """PAUQ может остаться без баз; нельзя усреднять «не мерили» с нулём."""
    aggregated = module.aggregate([card("lora", "pauq", 0, 0.3, execution=None)])
    assert aggregated[("lora", "pauq")]["execution_accuracy"] is None


def test_main_table_shows_spread_next_to_the_mean(module, cards) -> None:
    table = module.main_table(cards)
    assert "±" in table or "разброс" in table
    assert "lora" in table and "kan_lora" in table
    assert table.count("|") > 10


def test_main_table_includes_the_baseline_row_placeholder(module, cards) -> None:
    """В таблицах обязателен базовый уровень без адаптера: видно, что дала адаптация.
    Пока карточки `method: none` нет, строка остаётся и говорит, что прогона нет."""
    rows = [row for row in module.main_table(cards).splitlines() if "без адаптера" in row]
    assert len(rows) == 1
    assert "нет прогона" in rows[0]


def baseline_card(dataset: str, seed: int, exact: float, execution: float | None = 0.3) -> dict:
    """Карточка `method: none` в том виде, в каком её пишет build_result_card."""
    return {
        "run_id": f"none-{dataset}-seed{seed}",
        "method": "none", "dataset": dataset, "seed": seed, "optimizer": None,
        "config": {
            "adapter": {"learn_input_scale": True},
            "optimizer": {"name": "adamw"},
        },
        "metrics": {"exact_match": exact, "execution_accuracy": execution},
        "peak_memory_bytes": None, "train_seconds": None,
        "trainable_parameters": None,
    }


def test_baseline_cell_key_has_no_optimizer_and_no_input_scale(module) -> None:
    """Оптимизатор из замороженной конфигурации базовой модели не подхватывается:
    обучения не было, ключ — (none, набор, None, None)."""
    assert module.cell_key(baseline_card("spider", 0, 0.2)) == ("none", "spider")


def test_baseline_row_is_filled_from_cards(module, cards) -> None:
    table = module.main_table(
        cards + [baseline_card("spider", 0, 0.20), baseline_card("spider", 1, 0.22)]
    )
    rows = [row for row in table.splitlines() if "без адаптера" in row]

    assert len(rows) == 1
    cells = [cell.strip() for cell in rows[0].strip("|").split("|")]
    assert cells == [
        "без адаптера (базовый уровень)", "—", "spider",
        "0.2100 ± 0.0100", "0.3000 ± 0.0000", "—", "—",
    ]
    # Базовая точка — первая строка данных, над строками методов.
    assert table.splitlines()[2] == rows[0]


def test_baseline_rows_per_dataset(module, cards) -> None:
    table = module.main_table(
        cards + [baseline_card("spider", 0, 0.2), baseline_card("pauq", 0, 0.1, execution=None)]
    )
    rows = [row for row in table.splitlines() if "без адаптера" in row]

    assert len(rows) == 2
    assert any("pauq" in row and "не измерялась" in row for row in rows)














def with_config(item: dict, seed: int | None = None, **fields) -> dict:
    """Копия карточки с изменённым верхним полем config и зерном."""
    result = {**item, "config": {**item["config"], **fields}}
    if seed is not None:
        result["seed"] = seed
        result["config"]["seed"] = seed
    return result


def test_aggregate_refuses_cards_with_different_configs_in_one_cell(module) -> None:
    """Смок-карточка (eval_limit 50) и полная делят имя файла и ячейку: усреднять нельзя."""
    smoke = with_config(card("lora", "spider", 0, 0.50), eval_limit=50)
    full = with_config(card("lora", "spider", 1, 0.40), eval_limit=1034)

    with pytest.raises(ValueError) as error:
        module.aggregate([smoke, full])

    message = str(error.value)
    assert "eval_limit" in message
    assert smoke["run_id"] in message and full["run_id"] in message


def test_aggregate_names_nested_config_fields(module) -> None:
    first = with_config(card("lora", "spider", 0, 0.5), train={"epochs": 1})
    second = with_config(card("lora", "spider", 1, 0.4), train={"epochs": 2})

    with pytest.raises(ValueError, match=r"train\.epochs"):
        module.aggregate([first, second])


def test_aggregate_accepts_cards_that_differ_only_in_seed(module) -> None:
    first = with_config(card("lora", "spider", 0, 0.40), seed=0, eval_limit=50)
    second = with_config(card("lora", "spider", 1, 0.44), seed=1, eval_limit=50)

    aggregated = module.aggregate([first, second])

    assert aggregated[("lora", "spider")]["exact_match"][0] == pytest.approx(0.42)
