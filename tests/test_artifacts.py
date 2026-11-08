import copy
import json
from pathlib import Path

import pytest
import torch

from kanlora.adapters.inject import inject_adapters
from kanlora.artifacts import (
    new_run_id,
    reserve_run_directory,
    restore_adapter,
    save_adapter,
    write_predictions,
)
from kanlora.config import apply_overrides, config_to_dict, load_config

BASE = Path(__file__).parents[1] / 'configs' / 'base.yaml'


@pytest.mark.parametrize('method', ['lora', 'dora', 'kan_lora'])
def test_adapter_roundtrip(method, tiny_causal_lm, tmp_path):
    config = apply_overrides(load_config(BASE), method=method)
    tiny_causal_lm.config._name_or_path = config.model_name
    tiny_causal_lm.config._commit_hash = config.model_revision
    fresh = copy.deepcopy(tiny_causal_lm)
    inject_adapters(tiny_causal_lm, method, config.adapter)
    optimizer = torch.optim.SGD((p for p in tiny_causal_lm.parameters() if p.requires_grad), lr=0.01)
    tokens = torch.tensor([[1, 2, 3]])
    tiny_causal_lm(tokens, labels=tokens).loss.backward()
    optimizer.step()
    path = tmp_path / 'adapter.pt'
    save_adapter(tiny_causal_lm, config, path)
    checkpoint = torch.load(path, weights_only=True)
    expected = {n: p for n, p in tiny_causal_lm.named_parameters() if p.requires_grad}
    assert set(checkpoint['tensors']) == set(expected)
    assert all('.base.' not in name for name in checkpoint['tensors'])
    if method == 'dora':
        assert any('magnitude' in name for name in expected)
    if method == 'kan_lora':
        assert any('input_scale' in name for name in expected)
    assert restore_adapter(fresh, path) == config_to_dict(config)
    for name, parameter in fresh.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter, expected[name])
    torch.testing.assert_close(fresh(tokens).logits, tiny_causal_lm(tokens).logits)


def test_restore_rejects_wrong_base(tiny_causal_lm, tmp_path):
    config = load_config(BASE)
    inject_adapters(tiny_causal_lm, config.method, config.adapter)
    path = tmp_path / 'adapter.pt'
    save_adapter(tiny_causal_lm, config, path)
    tiny_causal_lm.config._name_or_path = 'wrong'
    tiny_causal_lm.config._commit_hash = config.model_revision
    with pytest.raises(ValueError):
        restore_adapter(tiny_causal_lm, path)


def test_duplicate_run_directory_is_rejected(tmp_path):
    directory = reserve_run_directory(tmp_path, 'smoke', 'same')
    (directory / 'x').write_bytes(b'old')
    with pytest.raises(FileExistsError):
        reserve_run_directory(tmp_path, 'smoke', 'same')
    assert (directory / 'x').read_bytes() == b'old'


def test_predictions_jsonl_and_lengths(tmp_path):
    path = tmp_path / 'predictions.jsonl'
    write_predictions(path, [2], ['db'], ['SELECT 1'], ['SELECT 2'])
    assert json.loads(path.read_text()) == {'index': 2, 'db_id': 'db', 'gold_sql': 'SELECT 1', 'predicted_sql': 'SELECT 2'}
    with pytest.raises(ValueError):
        write_predictions(path, [1], [], [], [])


def test_run_ids_are_unique():
    config = load_config(BASE)
    assert new_run_id(config) != new_run_id(config)


def test_restore_rejects_missing_tensor(tiny_causal_lm, tmp_path):
    config = load_config(BASE)
    tiny_causal_lm.config._name_or_path = config.model_name
    tiny_causal_lm.config._commit_hash = config.model_revision
    fresh = copy.deepcopy(tiny_causal_lm)
    inject_adapters(tiny_causal_lm, config.method, config.adapter)
    path = tmp_path / 'adapter.pt'
    save_adapter(tiny_causal_lm, config, path)
    checkpoint = torch.load(path, weights_only=True)
    checkpoint['tensors'].pop(next(iter(checkpoint['tensors'])))
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match='параметров'):
        restore_adapter(fresh, path)


def test_predictions_recompute_metrics(tmp_path):
    import sqlite3

    from kanlora.data.spider import SPIDER_LAYOUT
    from kanlora.eval.exact_match import exact_match_by_hardness
    from kanlora.eval.execution import execution_accuracy
    from kanlora.results import build_result_card

    tables = Path(__file__).parent / 'data' / 'fixtures' / 'spider_mini' / 'tables.json'
    db = tmp_path / 'database' / 'concert_singer'
    db.mkdir(parents=True)
    with sqlite3.connect(db / 'concert_singer.sqlite') as connection:
        connection.execute('CREATE TABLE singer (id INTEGER)')
    gold = ['SELECT count(*) FROM singer', 'SELECT count(*) FROM singer']
    predicted = ['SELECT count(*) FROM singer', 'SELECT count(*) FROM singer WHERE id > 0']
    db_ids = ['concert_singer'] * 2
    hardness = exact_match_by_hardness(gold, predicted, db_ids, tables)
    metrics = {'exact_match': hardness['all'], 'exact_match_by_hardness': hardness,
               'execution_accuracy': execution_accuracy(gold, predicted, db_ids, tmp_path, SPIDER_LAYOUT)}
    card = build_result_card(load_config(BASE), {'trainable_parameters': 1, 'total_parameters': 1,
                             'replaced': []}, None, metrics, run_id='test', provenance={}, artifacts={})
    path = tmp_path / 'predictions.jsonl'
    write_predictions(path, [0, 1], db_ids, gold, predicted)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    gold = [row['gold_sql'] for row in rows]
    predicted = [row['predicted_sql'] for row in rows]
    db_ids = [row['db_id'] for row in rows]
    assert exact_match_by_hardness(gold, predicted, db_ids, tables) == card['metrics']['exact_match_by_hardness']
    assert execution_accuracy(gold, predicted, db_ids, tmp_path, SPIDER_LAYOUT) == card['metrics']['execution_accuracy']
