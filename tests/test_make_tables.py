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
                   "data_root": "data", "max_length": 768, "adapter": {"rank": 8, "grid_size": 5},
                   "train": {"dtype": "fp32"}, "optimizer": {"name": "adamw"}},
        "provenance": {"indices": {"train": [1], "validation": [2], "eval": [0]},
                       "file_hashes": {f"data/{dataset}/{'dev.json' if dataset == 'spider' else 'pauq_dev.json'}": "b", f"data/{dataset}/tables.json": "c", f"data/{dataset}/database/db/db.sqlite": "d"},
                       "model_revision": "abc", "tokenizer_revision": "abc",
                       "device": "cuda", "evaluator": "v1", "evaluator_commit": "a" * 40, "lock_sha256": "b" * 64, "versions": {"torch": "1"}},
        "versions": {"torch": "1"}, "git_commit": "abc",
        "artifacts": {"predictions": "predictions.jsonl", **({"adapter": "adapter.pt"} if method != "none" else {})},
        "metrics": {"exact_match": 0.4, "execution_accuracy": 0.5},
        "peak_memory_bytes": 0, "trainable_parameters": 100,
    }


@pytest.mark.parametrize("path", [
    ("provenance", "file_hashes", "data/spider/dev.json"), ("provenance", "indices", "train"),
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
    (run / "predictions.jsonl").write_text(json.dumps({"index": 0, "db_id": "db"}) + "\n")
    assert module.load_cards(tmp_path) == [item]


def test_real_card_records_and_requires_evaluator_and_lock():
    import torch

    from kanlora.config import load_config
    from kanlora.provenance import collect_provenance
    from kanlora.results import build_result_card

    config = load_config(Path('configs/smoke.yaml'))
    provenance = collect_provenance(config, torch.device('cpu'), {'eval': [0]}, [],
                                    config.model_revision, config.model_revision)
    actual = build_result_card(config, {'trainable_parameters': 1, 'total_parameters': 1,
                                        'replaced': []}, None, {}, run_id='test',
                               provenance=provenance, artifacts={})
    for field in ('evaluator_commit', 'lock_sha256'):
        assert actual['provenance'].get(field)
        missing = copy.deepcopy(actual)
        missing['provenance'].pop(field)
        with pytest.raises(ValueError, match=field):
            module.validate_comparability([missing])


@pytest.mark.parametrize('missing', ['dev', 'tables', 'database', 'null'])
def test_evaluation_hashes_are_required(missing):
    item = card('none')
    hashes = item['provenance']['file_hashes']
    paths = {'dev': 'data/spider/dev.json', 'tables': 'data/spider/tables.json',
             'database': 'data/spider/database/db/db.sqlite'}
    if missing == 'null':
        hashes[paths['database']] = None
    else:
        hashes.pop(paths[missing])
    with pytest.raises(ValueError, match='file_hashes'):
        module.validate_comparability([item])


def test_nested_legacy_json_is_rejected(tmp_path):
    nested = tmp_path / 'old'
    nested.mkdir()
    (nested / 'old.json').write_text('{}')
    with pytest.raises(ValueError, match='v2'):
        module.load_cards(tmp_path)


def test_declared_json_artifact_is_not_legacy(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    item = card('none')
    item['artifacts']['extra'] = 'extra.json'
    (run / 'card.json').write_text(json.dumps(item))
    (run / 'predictions.jsonl').write_text(json.dumps({'index': 0, 'db_id': 'db'}) + '\n')
    (run / 'extra.json').write_text('{}')
    assert module.load_cards(tmp_path) == [item]


def test_missing_one_of_multiple_evaluation_database_hashes_is_rejected(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    item = card('none')
    item['provenance']['indices']['eval'] = [0, 1]
    (run / 'card.json').write_text(json.dumps(item))
    (run / 'predictions.jsonl').write_text('\n'.join(json.dumps({'index': i, 'db_id': name})
                                                     for i, name in enumerate(['db', 'other'])) + '\n')
    with pytest.raises(ValueError, match='file_hashes'):
        module.load_cards(tmp_path)


@pytest.mark.parametrize('field,length', [('evaluator_commit', 40), ('lock_sha256', 64)])
def test_provenance_identity_mismatch_is_rejected(field, length):
    first, second = card(), card('dora')
    second['provenance'][field] = 'c' * length
    with pytest.raises(ValueError, match=field):
        module.validate_comparability([first, second])
