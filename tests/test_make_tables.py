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


def card(method: str, dataset: str, seed: int, exact: float, execution: float | None = 0.5) -> dict:
    return {
        "run_id": f"{method}-{dataset}-adamw-seed{seed}",
        "method": method, "dataset": dataset, "seed": seed, "optimizer": "adamw",
        "metrics": {"exact_match": exact, "execution_accuracy": execution},
        "peak_memory_bytes": 4 * 2**30, "train_seconds": 3600.0,
        "trainable_parameters": 100_000,
        "optimizer_coverage": {"adamw": 100_000, "muon": 0},
        "fraction_inside_grid": [0.8, 0.9],
        "spline_stats": None,
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
    """В таблицах обязателен базовый уровень без адаптера: видно, что дала адаптация."""
    assert "без адаптера" in module.main_table(cards)


def test_optimizer_table_shows_muon_coverage(module, cards) -> None:
    table = module.optimizer_table(cards)
    assert "Muon" in table and "AdamW" in table


def test_plot_writes_a_file(module, cards, tmp_path: Path) -> None:
    path = tmp_path / "grid.png"
    module.plot_fraction_inside_grid(cards, path)
    assert path.is_file() and path.stat().st_size > 0
