import hashlib
import subprocess
from pathlib import Path

import pytest
import torch

from kanlora.config import load_config
from kanlora.provenance import collect_provenance, require_clean_tracked_tree


def test_provenance_hashes_json_and_sqlite(tmp_path):
    paths = [tmp_path / 'dev.json', tmp_path / 'db.sqlite', tmp_path / 'missing.sqlite']
    paths[0].write_text('[{"db_id":"db"}]')
    paths[1].write_bytes(b'sqlite')
    config = load_config(Path('configs/smoke.yaml'))
    result = collect_provenance(config, torch.device('cpu'), {'eval': [0]}, paths,
                                config.model_revision, config.model_revision)
    assert result['indices']['eval'] == [0]
    assert result['file_hashes'][str(paths[0])] == hashlib.sha256(paths[0].read_bytes()).hexdigest()
    assert result['file_hashes'][str(paths[1])] == hashlib.sha256(paths[1].read_bytes()).hexdigest()
    assert result['file_hashes'][str(paths[2])] is None


def test_final_rejects_dirty_tree_before_training(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(['git', 'init', '-q'], check=True)
    subprocess.run(['git', 'config', 'user.email', 'test@example.com'], check=True)
    subprocess.run(['git', 'config', 'user.name', 'Test'], check=True)
    Path('tracked').write_text('one')
    subprocess.run(['git', 'add', 'tracked'], check=True)
    subprocess.run(['git', 'commit', '-qm', 'initial'], check=True)
    assert len(require_clean_tracked_tree()) == 40
    Path('tracked').write_text('two')
    with pytest.raises(RuntimeError):
        require_clean_tracked_tree()
    subprocess.run(['git', 'add', 'tracked'], check=True)
    with pytest.raises(RuntimeError):
        require_clean_tracked_tree()


def test_smoke_records_dirty_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(['git', 'init', '-q'], check=True)
    subprocess.run(['git', 'config', 'user.email', 'test@example.com'], check=True)
    subprocess.run(['git', 'config', 'user.name', 'Test'], check=True)
    Path('tracked').write_text('one')
    subprocess.run(['git', 'add', 'tracked'], check=True)
    subprocess.run(['git', 'commit', '-qm', 'initial'], check=True)
    Path('tracked').write_text('two')
    config = load_config(Path(__file__).resolve().parents[1] / 'configs/smoke.yaml')
    result = collect_provenance(config, torch.device('cpu'), {}, [],
                                config.model_revision, config.model_revision)
    assert result['tracked_tree_dirty'] is True
    with pytest.raises(RuntimeError, match='Ревизии'):
        collect_provenance(config, torch.device('cpu'), {}, [], config.model_revision, 'main')
