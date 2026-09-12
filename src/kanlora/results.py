"""Карточка результата прогона.

Единица поставки. Из карточек собираются все таблицы и графики; руками они
не правятся никогда. Отдельно оговорено различие между null и нулём:
Execution Accuracy на PAUQ может оказаться неизмеримой из-за отсутствия
файлов баз данных, и «не мерили» обязано отличаться от «получили ноль».
"""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

from kanlora.config import ExperimentConfig, config_to_dict
from kanlora.train.loop import TrainReport

__all__ = ["build_result_card", "collect_versions", "git_commit", "write_result_card"]


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
    except Exception:
        return "неизвестен"


def build_result_card(
    config: ExperimentConfig,
    injection: dict,
    train_report: TrainReport,
    metrics: dict,
    spline_stats: dict | None,
) -> dict:
    return {
        "run_id": config.run_id,
        "method": config.method,
        "dataset": config.dataset,
        "seed": config.seed,
        "optimizer": config.optimizer.name,
        "config": config_to_dict(config),
        "versions": collect_versions(),
        "git_commit": git_commit(),
        "metrics": metrics,
        "peak_memory_bytes": train_report.peak_memory_bytes,
        "train_seconds": train_report.seconds,
        "trainable_parameters": injection["trainable_parameters"],
        "total_parameters": injection["total_parameters"],
        "adapted_modules": len(injection["replaced"]),
        "optimizer_coverage": train_report.optimizer_coverage,
        "step_losses": train_report.step_losses,
        "epoch_losses": train_report.epoch_losses,
        "fraction_inside_grid": train_report.fraction_inside_grid,
        "spline_stats": spline_stats,
    }


def write_result_card(card: dict, directory: Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / f"{card['run_id']}.json"
    path.write_text(
        json.dumps(card, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
