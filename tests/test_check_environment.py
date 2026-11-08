"""Проверки сценария диагностики окружения.

Сценарий обязан отчитываться, а не падать: на машине разработки нет ни CUDA,
ни наборов данных, и в этом состоянии он всё равно должен выдать полный отчёт.
"""

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_environment.py"


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("check_environment", SCRIPT)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules["check_environment"] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def test_torch_check_reports_version(module) -> None:
    check = module.check_torch()
    assert check.ok is True
    assert "2." in check.detail


def test_nltk_punkt_check_matches_actual_availability(module) -> None:
    """process_sql.py (Spider eval) вызывает nltk.word_tokenize -> нужен punkt_tab.

    Раньше эта зависимость не была в pyproject.toml вообще и молча работала
    только потому, что тесты случайно запускались интерпретатором с уже
    установленным nltk и скачанными данными — на чистом окружении (в том
    числе на машине обучения) Exact Match упал бы `LookupError` посреди
    оценки, а не на старте.
    """
    import nltk

    check = module.check_nltk_punkt()
    try:
        nltk.data.find("tokenizers/punkt_tab")
        available = True
    except LookupError:
        available = False
    assert check.ok is available


def test_missing_dataset_reports_not_ok_without_raising(module, tmp_path: Path) -> None:
    checks = module.check_dataset("spider", tmp_path / "nowhere")
    assert checks, "проверка набора обязана вернуть хотя бы один пункт"
    assert all(check.ok is False for check in checks)


def test_present_dataset_files_are_detected(module, tmp_path: Path) -> None:
    root = tmp_path / "spider"
    (root / "database" / "concert_singer").mkdir(parents=True)
    (root / "database" / "concert_singer" / "concert_singer.sqlite").write_bytes(b"")
    for name in ("train_spider.json", "dev.json", "tables.json"):
        (root / name).write_text("[]", encoding="utf-8")

    checks = {check.name: check for check in module.check_dataset("spider", root)}
    assert checks["spider: train_spider.json"].ok is True


def test_cuda_kernel_failure_is_reported(module, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "fake GPU")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (6, 1))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_61"])

    def fail(*args, **kwargs):
        raise RuntimeError("kernel failed")

    monkeypatch.setattr(torch, "ones", fail)
    checks = module.check_cuda()
    assert any(not c.ok and "kernel failed" in c.detail for c in checks)


def _dataset(root, queries):
    root.mkdir()
    entries = [{"db_id": db, "question": "q", "query": query} for db, query in queries]
    schemas = [
        {"db_id": db, "table_names_original": [], "column_names_original": []}
        for db in sorted({db for db, _ in queries})
    ]
    files = (("train_spider.json", entries), ("dev.json", entries), ("tables.json", schemas))
    for name, content in files:
        (root / name).write_text(json.dumps(content))


def _database(root, db):
    path = root / "database" / db / f"{db}.sqlite"
    path.parent.mkdir(parents=True)
    sqlite3.connect(path).close()


def test_all_dev_databases_are_required(module, tmp_path):
    root = tmp_path / "spider"
    _dataset(root, [("a", "select 1"), ("b", "select 1")])
    _database(root, "a")
    assert any(not c.ok and "b" in c.detail for c in module.check_dataset("spider", root))


def test_dataset_gold_error_is_reported(module, tmp_path):
    root = tmp_path / "spider"
    _dataset(root, [("a", "select * from missing")])
    _database(root, "a")
    assert any(not c.ok and "эталон" in c.detail for c in module.check_dataset("spider", root))


def _write_schemas(root, db_ids):
    schemas = [
        {"db_id": db, "table_names_original": [], "column_names_original": []}
        for db in db_ids
    ]
    (root / "tables.json").write_text(json.dumps(schemas))


@pytest.mark.parametrize("missing_split", ["train", "dev"])
def test_dataset_rejects_missing_schema_from_either_split(module, tmp_path, missing_split):
    root = tmp_path / "spider"
    _dataset(root, [("a", "select 1")])
    _database(root, "a")
    entries = [{"db_id": "b", "question": "q", "query": "select 1"}]
    filename = "train_spider.json" if missing_split == "train" else "dev.json"
    (root / filename).write_text(json.dumps(entries))
    _database(root, "b")
    _write_schemas(root, ["a"])

    checks = module.check_dataset("spider", root)
    assert any(not check.ok and "b" in check.detail for check in checks)


def test_dataset_accepts_schemas_for_all_train_and_dev_examples(module, tmp_path):
    root = tmp_path / "spider"
    _dataset(root, [("a", "select 1")])
    (root / "dev.json").write_text(json.dumps(
        [{"db_id": "b", "question": "q", "query": "select 1"}]
    ))
    _database(root, "b")
    _write_schemas(root, ["a", "b"])

    checks = module.check_dataset("spider", root)
    assert all(check.ok for check in checks), checks


def test_compatible_pascal_binary_and_successful_kernel(module, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "GTX 1080 Ti")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (6, 1))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_60"])
    monkeypatch.setattr(torch, "ones", lambda *a, **k: torch.tensor([1.0], requires_grad=True))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    assert all(check.ok for check in module.check_cuda())
