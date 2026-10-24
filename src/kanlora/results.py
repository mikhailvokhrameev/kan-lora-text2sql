"""Карточка результата прогона.

Единица поставки. Из карточек собираются все таблицы и графики; руками они
не правятся никогда. Отдельно оговорено различие между null и нулём:
Execution Accuracy на PAUQ может оказаться неизмеримой из-за отсутствия
файлов баз данных, и «не мерили» обязано отличаться от «получили ноль».
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

from kanlora.config import ExperimentConfig, config_to_dict
from kanlora.train.loop import TrainReport

__all__ = [
    "build_result_card", "collect_versions", "file_hashes", "git_commit", "write_result_card",
]


def collect_versions() -> dict[str, str]:
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda or "нет",
    }


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "неизвестен"


def file_hashes(root: Path, names: list[str]) -> dict[str, str | None]:
    """SHA-256 файлов набора данных — версия данных в карточке.

    `git_commit` фиксирует код, но `data/` не в git: без хэша прогон нельзя
    связать с конкретными файлами Spider/PAUQ. Отсутствующий файл — null.
    """
    hashes: dict[str, str | None] = {}
    for name in names:
        path = Path(root) / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return hashes


def build_result_card(
    config: ExperimentConfig,
    injection: dict,
    train_report: TrainReport | None,
    metrics: dict,
    spline_stats: dict | None,
    data_files: dict[str, str | None] | None = None,
    model_revision: str | None = None,
    latency: dict | None = None,
) -> dict:
    """Собирает карточку прогона.

    train_report=None — прогон без обучения (`method: none`): поля обучения
    (пик видеопамяти и время обучения, покрытие оптимизатором, потери на обучении
    и на валидации, доля
    активаций в сетке) пишутся null — «не обучалась», а не «ноль». Оптимизатора
    у такого прогона нет, поэтому и `optimizer` — null, хотя замороженная
    конфигурация в `config` по-прежнему содержит секцию optimizer из YAML.

    latency — задержка генерации без слияния адаптера (`eval/latency.py`);
    null — не мерили.
    """
    trained = train_report is not None
    return {
        "run_id": config.run_id,
        "method": config.method,
        "dataset": config.dataset,
        "seed": config.seed,
        "optimizer": config.optimizer.name if config.method != "none" else None,
        "config": config_to_dict(config),
        "versions": collect_versions(),
        "git_commit": git_commit(),
        # Версии данных и модели (лекции 1 и 3 MLOps): хэши файлов набора и
        # коммит репозитория модели на Hugging Face; null — неизвестно.
        "data_files": data_files,
        "model_revision": model_revision,
        "metrics": metrics,
        "peak_memory_bytes": train_report.peak_memory_bytes if trained else None,
        "train_seconds": train_report.seconds if trained else None,
        "trainable_parameters": injection["trainable_parameters"],
        "total_parameters": injection["total_parameters"],
        "adapted_modules": len(injection["replaced"]),
        "optimizer_coverage": train_report.optimizer_coverage if trained else None,
        "step_losses": train_report.step_losses if trained else None,
        "epoch_losses": train_report.epoch_losses if trained else None,
        "validation_losses": train_report.validation_losses if trained else None,
        "grad_norms": train_report.grad_norms if trained else None,
        "fraction_inside_grid": train_report.fraction_inside_grid if trained else None,
        "spline_stats": spline_stats,
        "latency": latency,
    }


def write_result_card(card: dict, directory: Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / f"{card['run_id']}.json"
    path.write_text(
        json.dumps(card, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
