"""Карточка результата прогона.

Единица поставки. Из карточек собираются все таблицы и графики; руками они
не правятся никогда. Отдельно оговорено различие между null и нулём:
Execution Accuracy на PAUQ может оказаться неизмеримой из-за отсутствия
файлов баз данных, и «не мерили» обязано отличаться от «получили ноль».
"""

from __future__ import annotations

import json
from pathlib import Path

from kanlora.config import ExperimentConfig, config_to_dict
from kanlora.provenance import collect_versions, file_hashes, git_commit
from kanlora.train.loop import TrainReport

__all__ = [
    "build_result_card", "collect_versions", "file_hashes", "git_commit", "write_result_card",
]


def build_result_card(
    config: ExperimentConfig,
    injection: dict,
    train_report: TrainReport | None,
    metrics: dict,
    data_files: dict[str, str | None] | None = None,
    model_revision: str | None = None,
) -> dict:
    """Собирает карточку прогона; поля обучения равны null без обучения."""
    trained = train_report is not None
    return {
        "run_id": config.run_id,
        "method": config.method,
        "dataset": config.dataset,
        "seed": config.seed,
        "optimizer": "adamw" if config.method != "none" else None,
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
        "step_losses": train_report.step_losses if trained else None,
        "epoch_losses": train_report.epoch_losses if trained else None,
        "validation_losses": train_report.validation_losses if trained else None,
        "grad_norms": train_report.grad_norms if trained else None,
    }


def write_result_card(card: dict, directory: Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / f"{card['run_id']}.json"
    path.write_text(
        json.dumps(card, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
