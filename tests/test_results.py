"""Проверки карточки результата.

Карточка — единица поставки: по ней восстанавливается прогон и из неё
собираются все таблицы. Поля, которых в ней нет, в работу не попадут.
"""

import json
from pathlib import Path

import pytest

from kanlora.config import load_config
from kanlora.results import build_result_card, collect_versions, git_commit, write_result_card
from kanlora.train.loop import TrainReport

BASE = Path(__file__).parents[1] / "configs" / "base.yaml"


@pytest.fixture
def card() -> dict:
    return build_result_card(
        config=load_config(BASE),
        injection={"trainable_parameters": 1234, "total_parameters": 500000,
                   "replaced": ["model.layers.0.self_attn.q_proj"]},
        train_report=TrainReport(
            step_losses=[2.0, 1.0], epoch_losses=[1.5], fraction_inside_grid=[0.87],
            peak_memory_bytes=4_000_000_000, seconds=4200.0,
            optimizer_coverage={"adamw": 1234, "muon": 0},
        ),
        metrics={"exact_match": 0.41, "execution_accuracy": 0.55,
                 "exact_match_by_hardness": {"easy": 0.7, "all": 0.41}},
        spline_stats={"model.layers.0.self_attn.q_proj": {"mean": 0.03, "max": 0.11,
                                                          "fraction_inside_grid": 0.87}},
    )


def test_card_carries_everything_needed_to_reproduce(card: dict) -> None:
    assert set(card) >= {
        "run_id", "config", "versions", "git_commit", "metrics",
        "peak_memory_bytes", "train_seconds", "trainable_parameters",
        "optimizer_coverage", "fraction_inside_grid", "spline_stats",
    }


def test_versions_include_the_libraries_that_affect_numbers() -> None:
    versions = collect_versions()
    assert set(versions) >= {"python", "torch", "transformers"}
    assert versions["torch"].startswith("2.")


def test_git_commit_is_recorded(card: dict) -> None:
    assert isinstance(card["git_commit"], str)
    assert card["git_commit"]


def test_absent_execution_accuracy_is_null_not_zero() -> None:
    """Отличие «не мерили» от «ноль» принципиально: PAUQ может остаться без баз данных."""
    card = build_result_card(
        config=load_config(BASE), injection={"trainable_parameters": 1, "total_parameters": 2,
                                             "replaced": []},
        train_report=TrainReport(), metrics={"exact_match": 0.1, "execution_accuracy": None},
        spline_stats=None,
    )
    assert card["metrics"]["execution_accuracy"] is None


def test_card_carries_step_losses(card: dict, tmp_path: Path) -> None:
    """`step_losses` — единственная запись пошагового спуска, без неё карточка
    не даёт различить гладкое обучение от скачков внутри эпохи."""
    assert card["step_losses"] == [2.0, 1.0]

    path = write_result_card(card, tmp_path)
    assert json.loads(path.read_text(encoding="utf-8"))["step_losses"] == [2.0, 1.0]


def test_written_card_is_valid_json_named_by_run(card: dict, tmp_path: Path) -> None:
    path = write_result_card(card, tmp_path)

    assert path.name == f"{card['run_id']}.json"
    assert json.loads(path.read_text(encoding="utf-8")) == card


def test_card_is_written_in_readable_utf8(card: dict, tmp_path: Path) -> None:
    """Карточки лежат в git и читаются глазами — экранированной кириллицы там быть не должно."""
    card["note"] = "проверка"
    text = write_result_card(card, tmp_path).read_text(encoding="utf-8")
    assert "проверка" in text
