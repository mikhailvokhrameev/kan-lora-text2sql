"""Проверки сопоставимости карточек и итоговой таблицы."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("make_tables", Path(__file__).resolve().parents[1] / "scripts/make_tables.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def card(method="lora", dataset="spider"):
    return {
        "schema_version": 2, "run_id": method + dataset, "method": method,
        "dataset": dataset, "seed": 0, "optimizer": "adamw" if method != "none" else None,
        "config": {"method": method, "dataset": dataset, "seed": 0, "run_kind": "final",
                   "max_length": 768, "adapter": {"rank": 8, "grid_size": 5},
                   "train": {"dtype": "fp32"}, "optimizer": {"name": "adamw"}},
        "provenance": {"indices": {"train": [1], "validation": [2], "eval": [0]},
                       "file_hashes": {"train.json": "a", "dev.json": "b"},
                       "model_revision": "abc", "tokenizer_revision": "abc",
                       "device": "cuda", "evaluator": "v1", "versions": {"torch": "1"}},
        "versions": {"torch": "1"}, "git_commit": "abc",
        "artifacts": {"predictions": "predictions.jsonl", **({"adapter": "adapter.pt"} if method != "none" else {})},
        "metrics": {"exact_match": 0.4, "execution_accuracy": 0.5},
        "peak_memory_bytes": 0, "trainable_parameters": 100,
    }


@pytest.mark.parametrize("path", [
    ("provenance", "file_hashes", "dev.json"), ("provenance", "indices", "train"),
    ("provenance", "model_revision"), ("provenance", "evaluator"),
    ("config", "max_length"), ("config", "train", "dtype"),
    ("provenance", "device"), ("versions", "torch"),
])
def test_cross_method_mismatch_is_rejected(path):
    first, second = card(), card("dora")
    target = second
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = "different"
    with pytest.raises(ValueError, match=path[-1]):
        module.validate_comparability([first, second])


def test_mixed_run_kinds_are_rejected():
    other = card("dora", "pauq")
    other["config"]["run_kind"] = "smoke"
    with pytest.raises(ValueError, match="run_kind"):
        module.validate_comparability([card(), other])


def test_legacy_card_is_rejected(tmp_path):
    (tmp_path / "old.json").write_text('{}')
    with pytest.raises(ValueError, match="v2"):
        module.load_cards(tmp_path)


def test_duplicate_cell_is_rejected():
    other = copy.deepcopy(card())
    other["run_id"] = "different"
    with pytest.raises(ValueError, match="spider.*lora.*0"):
        module.validate_comparability([card(), other])


def test_table_has_single_seed_values_without_spread():
    cards = [card(method, dataset) for dataset in ("spider", "pauq") for method in ("none", "lora", "dora", "kan_lora")]
    table = module.main_table(cards)
    rows = table.splitlines()
    assert len(rows) == 10
    assert all(len(row.strip('|').split('|')) == 7 for row in rows)
    assert "±" not in table and "latency" not in table
    assert sum("none" in row for row in rows) == 2
    assert "0.4000" in table


def test_cpu_zero_is_unmeasured():
    assert "не измерялась" in module.main_table([card()])


def test_baseline_training_is_not_applicable():
    assert module.main_table([card("none")]).count("не применимо") == 2


def test_incomplete_run_is_not_loaded(tmp_path):
    run = tmp_path / "run1"
    run.mkdir()
    (tmp_path / "incomplete").mkdir()
    item = card()
    (run / "card.json").write_text(json.dumps(item))
    with pytest.raises(ValueError, match="predictions"):
        module.load_cards(tmp_path)
    for name in item["artifacts"].values():
        (run / name).touch()
    assert module.load_cards(tmp_path) == [item]
