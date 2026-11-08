"""Происхождение кода, файлов и окружения прогона."""
import hashlib
import platform
import subprocess
from pathlib import Path

import torch

from kanlora.config import config_to_dict


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


def tracked_tree_dirty() -> bool:
    try:
        return any(
            subprocess.run(command, check=False).returncode != 0
            for command in (
                ["git", "diff", "--quiet", "--cached", "HEAD", "--"],
                ["git", "diff", "--quiet", "--"],
            )
        )
    except OSError:
        return True


def require_clean_tracked_tree() -> str:
    commit = git_commit()
    if commit == "неизвестен" or tracked_tree_dirty():
        raise RuntimeError("Для final требуется чистое отслеживаемое дерево git и HEAD")
    return commit


def collect_provenance(config, device, indices, paths, model_revision, tokenizer_revision) -> dict:
    if model_revision != config.model_revision or tokenizer_revision != config.model_revision:
        raise RuntimeError("Ревизии модели и токенизатора не совпадают с конфигурацией")
    return {
        "evaluator_commit": "b7b5b8c890cd30e35427348bb9eb8c6d1350ca7c",
        "lock_sha256": hashlib.sha256((Path(__file__).resolve().parents[2] / "uv.lock").read_bytes()).hexdigest(),
        "config": config_to_dict(config), "versions": collect_versions(),
        "git_commit": git_commit(), "tracked_tree_dirty": tracked_tree_dirty(),
        "indices": indices, "file_hashes": file_hashes(Path("."), [str(p) for p in paths]),
        "model_revision": model_revision, "tokenizer_revision": tokenizer_revision,
        "device": str(device),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
