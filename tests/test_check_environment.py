"""Проверки сценария диагностики окружения.

Сценарий обязан отчитываться, а не падать: на машине разработки нет ни CUDA,
ни наборов данных, и в этом состоянии он всё равно должен выдать полный отчёт.
"""

import importlib.util
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


def test_muon_check_matches_torch_optim(module) -> None:
    import torch

    check = module.check_muon()
    assert check.ok is hasattr(torch.optim, "Muon")


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
    assert checks["spider: файлы баз данных"].ok is True
