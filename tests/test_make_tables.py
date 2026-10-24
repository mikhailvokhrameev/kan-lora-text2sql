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
    mean, spread = aggregated[("lora", "spider", "adamw", True)]["exact_match"]

    assert mean == pytest.approx(0.42)
    assert spread == pytest.approx(0.04)  # размах: 0.44 - 0.40


def test_single_seed_has_zero_spread(module) -> None:
    aggregated = module.aggregate([card("dora", "pauq", 0, 0.3)])
    assert aggregated[("dora", "pauq", "adamw", True)]["exact_match"][1] == pytest.approx(0.0)


def test_absent_execution_accuracy_does_not_become_zero(module) -> None:
    """PAUQ может остаться без баз; нельзя усреднять «не мерили» с нулём."""
    aggregated = module.aggregate([card("lora", "pauq", 0, 0.3, execution=None)])
    assert aggregated[("lora", "pauq", "adamw", True)]["execution_accuracy"] is None


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
        "optimizer_coverage": None,
        "fraction_inside_grid": None,
        "spline_stats": None,
    }


def test_baseline_cell_key_has_no_optimizer_and_no_input_scale(module) -> None:
    """Оптимизатор из замороженной конфигурации базовой модели не подхватывается:
    обучения не было, ключ — (none, набор, None, None)."""
    assert module.cell_key(baseline_card("spider", 0, 0.2)) == ("none", "spider", None, None)


def test_baseline_row_is_filled_from_cards(module, cards) -> None:
    table = module.main_table(
        cards + [baseline_card("spider", 0, 0.20), baseline_card("spider", 1, 0.22)]
    )
    rows = [row for row in table.splitlines() if "без адаптера" in row]

    assert len(rows) == 1
    cells = [cell.strip() for cell in rows[0].strip("|").split("|")]
    assert cells == [
        "без адаптера (базовый уровень)", "—", "spider",
        "0.2100 ± 0.0100", "0.3000 ± 0.0000", "—", "—", "—",
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


def test_baseline_is_absent_from_the_optimizer_and_nonlinearity_tables(module, cards) -> None:
    with_baseline = cards + [baseline_card("spider", 0, 0.2)]
    assert "none-spider-seed0" not in module.optimizer_table(with_baseline)
    assert "none-spider-seed0" not in module.nonlinearity_table(with_baseline)


def test_plot_tolerates_baseline_cards(module, cards, tmp_path: Path) -> None:
    path = tmp_path / "grid.png"
    module.plot_fraction_inside_grid(cards + [baseline_card("spider", 0, 0.2)], path)
    assert path.is_file()


def test_optimizer_table_shows_muon_coverage(module, cards) -> None:
    table = module.optimizer_table(cards)
    assert "Muon" in table and "AdamW" in table


def test_aggregate_does_not_mix_optimizers(module) -> None:
    """Прогоны AdamW и Muon — разные конфигурации: их среднее не имеет смысла."""
    aggregated = module.aggregate(
        [
            card("kan_lora", "spider", 0, 0.40),
            card("kan_lora", "spider", 1, 0.42),
            card("kan_lora", "spider", 0, 0.50, optimizer="muon"),
        ]
    )

    assert aggregated[("kan_lora", "spider", "adamw", True)]["exact_match"][0] == pytest.approx(0.41)
    assert aggregated[("kan_lora", "spider", "muon", True)]["exact_match"][0] == pytest.approx(0.50)


def test_aggregate_does_not_mix_input_scale_ablation(module) -> None:
    aggregated = module.aggregate(
        [
            card("kan_lora", "spider", 0, 0.45),
            card("kan_lora", "spider", 0, 0.35, learn_input_scale=False),
        ]
    )

    assert aggregated[("kan_lora", "spider", "adamw", True)]["exact_match"][0] == pytest.approx(0.45)
    assert aggregated[("kan_lora", "spider", "adamw", False)]["exact_match"][0] == pytest.approx(0.35)


def test_input_scale_flag_does_not_split_non_kan_methods(module) -> None:
    """learn_input_scale относится только к KAN-LoRA; run_id у LoRA/DoRA его не различает."""
    aggregated = module.aggregate(
        [
            card("lora", "spider", 0, 0.40),
            card("lora", "spider", 1, 0.42, learn_input_scale=False),
        ]
    )

    assert list(aggregated) == [("lora", "spider", "adamw", True)]


def with_config(item: dict, seed: int | None = None, device: str | None = None, **fields) -> dict:
    """Копия карточки с изменённым верхним полем config, зерном и устройством задержки."""
    result = {**item, "config": {**item["config"], **fields}}
    if seed is not None:
        result["seed"] = seed
        result["config"]["seed"] = seed
    if device is not None:
        result["latency"] = {"device": device, "seconds_per_token": 0.05}
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

    assert aggregated[("lora", "spider", "adamw", True)]["exact_match"][0] == pytest.approx(0.42)


def test_aggregate_refuses_cards_measured_on_different_devices(module) -> None:
    """Задержка на CPU и на GPU несравнима, средняя по ним ничего не значит."""
    cpu = with_config(card("lora", "spider", 0, 0.40), seed=0, device="cpu")
    gpu = with_config(card("lora", "spider", 1, 0.44), seed=1, device="cuda")

    with pytest.raises(ValueError, match="device"):
        module.aggregate([cpu, gpu])


def test_missing_latency_does_not_conflict_with_a_measured_device(module) -> None:
    """Старая карточка без latency не имеет устройства и не мешает измеренному."""
    old = with_config(card("lora", "spider", 0, 0.40), seed=0)
    measured = with_config(card("lora", "spider", 1, 0.44), seed=1, device="cuda")

    assert module.aggregate([old, measured])


def test_old_cards_without_optimizer_fields_count_as_defaults(module) -> None:
    """Старые карточки без optimizer/config агрегируются как AdamW с обучаемым масштабом."""
    old = card("kan_lora", "spider", 0, 0.40)
    del old["optimizer"], old["config"]
    config_only = card("kan_lora", "spider", 1, 0.44)
    del config_only["optimizer"]

    aggregated = module.aggregate([old, config_only, card("kan_lora", "spider", 2, 0.42)])

    assert list(aggregated) == [("kan_lora", "spider", "adamw", True)]
    assert aggregated[("kan_lora", "spider", "adamw", True)]["exact_match"][0] == pytest.approx(0.42)


def test_main_table_labels_rows_by_optimizer_and_input_scale(module) -> None:
    rows = module.main_table(
        [
            card("kan_lora", "spider", 0, 0.45),
            card("kan_lora", "spider", 0, 0.50, optimizer="muon"),
            card("kan_lora", "spider", 0, 0.35, learn_input_scale=False),
        ]
    ).splitlines()[3:]

    pairs = sorted((row.split("|")[1].strip(), row.split("|")[2].strip()) for row in rows)
    assert pairs == [
        ("kan_lora", "adamw"),
        ("kan_lora", "muon"),
        ("kan_lora (без масштаба входа)", "adamw"),
    ]


def test_plot_writes_a_file(module, cards, tmp_path: Path) -> None:
    path = tmp_path / "grid.png"
    module.plot_fraction_inside_grid(cards, path)
    assert path.is_file() and path.stat().st_size > 0


def _kan_card(stats: dict) -> dict:
    return {**card("kan_lora", "spider", 0, 0.45), "spline_stats": stats}


def _nonlinearity_row(module, stats: dict) -> list[str]:
    row = module.nonlinearity_table([_kan_card(stats)]).splitlines()[-1]
    return [cell.strip() for cell in row.strip("|").split("|")]


@pytest.mark.parametrize("nan_first", [True, False])
def test_nonlinearity_table_skips_unmeasured_adapters(module, nan_first: bool) -> None:
    """NaN адаптера целиком вне сетки не должен ни заражать среднее, ни теряться в max."""
    nan = float("nan")
    stats = {"a": {"index": nan}, "b": {"index": 0.2}, "c": {"index": 0.6}}
    if not nan_first:
        stats = dict(reversed(list(stats.items())))

    _, mean, maximum, _ = _nonlinearity_row(module, stats)
    assert mean == "0.4000" and maximum == "0.6000"


def test_nonlinearity_table_marks_run_with_nothing_measured(module) -> None:
    nan = float("nan")
    _, mean, maximum, _ = _nonlinearity_row(module, {"a": {"index": nan}, "b": {"index": nan}})
    assert mean == maximum == "не измерялась"


def test_nonlinearity_table_does_not_read_old_card_format(module) -> None:
    """Карточки до индекса слоя (mean/max по рёбрам) с новыми несравнимы: не читаются."""
    _, mean, maximum, _ = _nonlinearity_row(module, {"a": {"mean": 0.3, "max": 1.0}})
    assert mean == maximum == "не измерялась"


def test_main_table_shows_generation_latency(module) -> None:
    measured = {**card("lora", "spider", 0, 0.4), "latency": {"seconds_per_token": 0.025}}
    assert module.main_table([measured]).splitlines()[-1].rstrip().endswith("| 25.0 |")


def test_main_table_marks_missing_latency(module, cards) -> None:
    """Карточки без поля latency (старые) — прочерк, а не падение."""
    rows = module.main_table(cards).splitlines()[3:]
    assert rows and all(row.rstrip().endswith("| — |") for row in rows)
