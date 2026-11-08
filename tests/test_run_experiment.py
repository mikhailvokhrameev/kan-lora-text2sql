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
SMOKE = ROOT / "configs" / "smoke.yaml"


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



@pytest.mark.parametrize("option", ["--optimizer", "--no-input-scale"])
def test_cli_rejects_removed_options(module, monkeypatch, option) -> None:
    args = ["run_experiment.py", option]
    if option == "--optimizer":
        args.append("adamw")
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as error:
        module.parse_arguments()
    assert error.value.code == 2


def test_final_requires_target_gpu_and_seed_zero(module, monkeypatch):
    config = load_config(BASE)
    with pytest.raises(RuntimeError, match='GPU'):
        module.validate_run(config, torch.device('cpu'), [1])
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda *_: 'NVIDIA GeForce GTX 1080 Ti')
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda *_: (6, 1))
    with pytest.raises(RuntimeError, match='seed'):
        module.validate_run(apply_overrides(config, seed=1), torch.device('cuda'), [1])


def test_final_rejects_partial_or_empty_dev(module, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda *_: 'NVIDIA GeForce GTX 1080 Ti')
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda *_: (6, 1))
    config = load_config(BASE)
    with pytest.raises(RuntimeError, match='dev'):
        module.validate_run(config, torch.device('cuda'), [])
    with pytest.raises(RuntimeError, match='dev'):
        module.validate_run(apply_overrides(config, eval_limit=1), torch.device('cuda'), [1])


def test_both_loaders_receive_pinned_revision(module, monkeypatch):
    calls = []
    class Tokenizer:
        pad_token_id = 0
    class Loader:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append((name, kwargs))
            return Tokenizer()
    monkeypatch.setattr(module, 'AutoTokenizer', Loader)
    monkeypatch.setattr(module, 'AutoModelForCausalLM', Loader)
    config = load_config(SMOKE)
    module.load_model_and_tokenizer(config)
    assert len(calls) == 2
    assert all(kwargs['revision'] == config.model_revision for _, kwargs in calls)


@pytest.mark.parametrize('kind,late_error', [('smoke', False), ('smoke', True), ('pilot', True)])
def test_run_provenance_indices_reference_original_json(module, tmp_path, monkeypatch, kind, late_error):
    import dataclasses
    import hashlib
    import json
    import sqlite3
    from types import SimpleNamespace

    from kanlora.data.spider import SPIDER_LAYOUT

    root = tmp_path / 'spider'
    root.mkdir()
    train = [{'db_id': 'db', 'question': f'question {i}', 'query': f'SELECT {i}'}
             for i in range(8)]
    dev = [{'db_id': 'db', 'question': f'dev {i}', 'query': f'SELECT {i}'}
           for i in range(3)]
    (root / SPIDER_LAYOUT.train_file).write_text(json.dumps(train))
    (root / SPIDER_LAYOUT.eval_file).write_text(json.dumps(dev))
    (root / SPIDER_LAYOUT.tables_file).write_text('[]')
    db_path = root / 'database' / 'db'
    db_path.mkdir(parents=True)
    sqlite3.connect(db_path / 'db.sqlite').close()
    if not late_error:
        dev[1]['db_id'] = 'missing'
    (root / SPIDER_LAYOUT.eval_file).write_text(json.dumps(dev))
    config = dataclasses.replace(load_config(SMOKE), data_root=str(tmp_path), train_subset=3,
                                 validation_size=2, eval_limit=2, run_kind=kind)
    args = SimpleNamespace(config=SMOKE, method=None, dataset=None, seed=None,
                           train_subset=None, eval_limit=None, run_kind=None, output=tmp_path)
    monkeypatch.setattr(module, 'parse_arguments', lambda: args)
    monkeypatch.setattr(module, 'load_config', lambda _: config)
    monkeypatch.setattr(module, 'load_dataset_schemas', lambda *_: {})
    token = SimpleNamespace(init_kwargs={'_commit_hash': config.model_revision})
    model = SimpleNamespace(config=SimpleNamespace(_commit_hash=config.model_revision))
    monkeypatch.setattr(module, 'load_model_and_tokenizer', lambda _: (model, token))
    seen = {}
    def train_stub(_config, _model, _token, training, _schemas, _device, validation):
        seen['train'] = training
        seen['validation'] = validation
        return {'trainable_parameters': 1, 'total_parameters': 1, 'replaced': []}, None, 0
    monkeypatch.setattr(module, 'adapt_and_train', train_stub)
    def prompts_stub(examples, *_):
        seen['eval'] = examples
        return [e.question for e in examples]
    monkeypatch.setattr(module, 'build_prompts', prompts_stub)
    monkeypatch.setattr(module, 'count_truncated_prompts', lambda *a, **k: 0)
    monkeypatch.setattr(module, 'generate_sql', lambda *a, **k: ['SELECT 0', 'SELECT 1'])
    monkeypatch.setattr(module, 'exact_match_by_hardness', lambda *a: {'all': 1.0})
    if late_error:
        def fail_late(*args):
            raise module.EvaluationDataError('db: база недоступна')
        monkeypatch.setattr(module, 'execution_accuracy', fail_late)
    monkeypatch.setattr(module, 'save_adapter', lambda *args: None)
    monkeypatch.setattr(module, 'write_result_card', lambda card, _: seen.setdefault('card', card))
    if kind == 'pilot':
        with pytest.raises(module.EvaluationDataError, match='база недоступна'):
            module.main()
        assert 'card' not in seen
        return
    assert module.main() == 0
    assert seen['card']['metrics']['execution_accuracy'] is None
    assert seen['card']['metrics']['execution_accuracy_reason']
    provenance = seen['card']['provenance']
    for split in ('train', 'validation'):
        assert [train[i]['question'] for i in provenance['indices'][split]] == [
            e.question for e in seen[split]
        ]
    assert provenance['indices']['eval'] == [0, 1]
    assert [dev[i]['question'] for i in provenance['indices']['eval']] == [
        e.question for e in seen['eval']
    ]
    assert provenance['file_hashes'][str(root / SPIDER_LAYOUT.train_file)] == hashlib.sha256(
        (root / SPIDER_LAYOUT.train_file).read_bytes()
    ).hexdigest()


@pytest.mark.parametrize('kind', ['pilot', 'final'])
@pytest.mark.parametrize('corrupt', [False, True], ids=['missing', 'corrupt'])
def test_incomplete_database_blocks_final_before_training(module, tmp_path, monkeypatch, corrupt, kind):
    import dataclasses
    import json
    import sqlite3
    from types import SimpleNamespace

    from kanlora.data.spider import SPIDER_LAYOUT

    root = tmp_path / 'spider'
    root.mkdir()
    dev = [{'db_id': db, 'question': 'q', 'query': 'SELECT * FROM item'}
           for db in ('good', 'bad')]
    (root / SPIDER_LAYOUT.eval_file).write_text(json.dumps(dev))
    (root / SPIDER_LAYOUT.tables_file).write_text('[]')
    for db in ('good', 'bad') if corrupt else ('good',):
        path = root / 'database' / db
        path.mkdir(parents=True)
        if db == 'bad':
            (path / f'{db}.sqlite').write_bytes(b'corrupt')
        else:
            with sqlite3.connect(path / f'{db}.sqlite') as con:
                con.execute('CREATE TABLE item (id INTEGER)')
    config = dataclasses.replace(load_config(BASE), data_root=str(tmp_path), method='none', run_kind=kind)
    args = SimpleNamespace(config=BASE, method=None, dataset=None, seed=None,
                           train_subset=None, eval_limit=None, run_kind=None, output=tmp_path)
    monkeypatch.setattr(module, 'parse_arguments', lambda: args)
    monkeypatch.setattr(module, 'load_config', lambda _: config)
    monkeypatch.setattr(module, 'require_clean_tracked_tree', lambda: None)
    monkeypatch.setattr(module, 'validate_run', lambda *a: None)
    monkeypatch.setattr(module, 'load_dataset_schemas', lambda *_: {})
    monkeypatch.setattr(module, 'load_model_and_tokenizer', lambda *_: pytest.fail('model loaded'))
    with pytest.raises(RuntimeError, match='bad'):
        module.main()


@pytest.mark.parametrize('method, failure', [('none', None), ('lora', 'predictions'), ('lora', 'adapter')])
def test_artifact_publication(module, tmp_path, monkeypatch, method, failure):
    import dataclasses
    import json
    import sqlite3
    from types import SimpleNamespace

    from kanlora.data.spider import SPIDER_LAYOUT

    root = tmp_path / 'spider'
    root.mkdir()
    example = [{'db_id': 'db', 'question': 'q', 'query': 'SELECT 1'}]
    for filename in (SPIDER_LAYOUT.train_file, SPIDER_LAYOUT.eval_file):
        (root / filename).write_text(json.dumps(example))
    (root / SPIDER_LAYOUT.tables_file).write_text('[]')
    db = root / 'database' / 'db'
    db.mkdir(parents=True)
    sqlite3.connect(db / 'db.sqlite').close()
    config = dataclasses.replace(load_config(SMOKE), data_root=str(tmp_path), method=method,
                                 train_subset=0, validation_size=0)
    args = SimpleNamespace(config=SMOKE, method=None, dataset=None, seed=None,
                           train_subset=None, eval_limit=None, run_kind=None, output=tmp_path / 'out')
    monkeypatch.setattr(module, 'parse_arguments', lambda: args)
    monkeypatch.setattr(module, 'load_config', lambda _: config)
    monkeypatch.setattr(module, 'load_dataset_schemas', lambda *_: {})
    token = SimpleNamespace(init_kwargs={'_commit_hash': config.model_revision})
    model = SimpleNamespace(config=SimpleNamespace(_commit_hash=config.model_revision))
    monkeypatch.setattr(module, 'load_model_and_tokenizer', lambda _: (model, token))
    monkeypatch.setattr(module, 'adapt_and_train', lambda *args: (
        {'trainable_parameters': None, 'total_parameters': 1, 'replaced': []}, None, None))
    monkeypatch.setattr(module, 'build_prompts', lambda *args: ['q'])
    monkeypatch.setattr(module, 'count_truncated_prompts', lambda *a, **k: 0)
    monkeypatch.setattr(module, 'generate_sql', lambda *a, **k: ['SELECT 1'])
    monkeypatch.setattr(module, 'exact_match_by_hardness', lambda *a: {'all': 1.0})
    if failure:
        def fail(*args):
            raise OSError('write failed')
        monkeypatch.setattr(module, 'write_predictions' if failure == 'predictions' else 'save_adapter', fail)
        with pytest.raises(OSError, match='write failed'):
            module.main()
        assert not list((tmp_path / 'out').rglob('card.json'))
    else:
        assert module.main() == 0
        card_path, = (tmp_path / 'out').rglob('card.json')
        card = json.loads(card_path.read_text())
        assert card['artifacts'] == {'predictions': 'predictions.jsonl'}
        assert not (card_path.parent / 'adapter.pt').exists()


def test_run_kind_cli_override_and_default_output(module, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--run-kind", "pilot"])
    args = module.parse_arguments()
    assert args.output == Path("results/v2")
    assert apply_overrides(load_config(BASE), run_kind=args.run_kind).run_kind == "pilot"
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--run-kind", "bad"])
    with pytest.raises(SystemExit):
        module.parse_arguments()
