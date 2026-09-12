"""Диагностика окружения перед запуском экспериментов.

Отвечает на четыре вопроса, каждый из которых способен сорвать работу
целиком: поддерживает ли сборка PyTorch данную видеокарту, есть ли Muon в
torch.optim, скачаны ли данные токенизатора nltk (`punkt_tab`), нужные
официальному скрипту оценки Spider, и лежат ли на диске файлы наборов вместе
с базами данных для Execution Accuracy.

Запуск: python scripts/check_environment.py [--data-root data]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

DATASET_FILES = {
    "spider": ("train_spider.json", "dev.json", "tables.json"),
    # Имена и раскладка проверены по факту в репозитории ai-spiderweb/pauq
    # (каталог dataset/): train/dev, не train/test, без префикса "xsp".
    "pauq": ("pauq_train.json", "pauq_dev.json", "tables.json"),
}


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def check_torch() -> Check:
    import torch

    return Check("torch", True, f"версия {torch.__version__}")


def check_cuda() -> list[Check]:
    """Главный риск: свежие сборки PyTorch отказываются от Pascal (sm_61)."""
    import torch

    if not torch.cuda.is_available():
        return [Check("CUDA", False, "недоступна (ожидаемо на машине разработки)")]

    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    supported = torch.cuda.get_arch_list()
    arch = f"sm_{capability[0]}{capability[1]}"
    return [
        Check("CUDA", True, f"{name}, {arch}"),
        Check(
            f"поддержка {arch} в сборке",
            arch in supported,
            f"сборка умеет: {', '.join(supported)}",
        ),
    ]


def check_muon() -> Check:
    import torch

    present = hasattr(torch.optim, "Muon")
    detail = "есть в torch.optim" if present else "отсутствует — нужна своя реализация"
    return Check("Muon", present, detail)


def check_nltk_punkt() -> Check:
    """Официальный скрипт оценки Spider (`third_party/spider_eval/process_sql.py`)
    вызывает `nltk.word_tokenize`, которому нужны скачанные данные `punkt_tab`.

    Без этой проверки отсутствие данных обнаружилось бы `LookupError` посреди
    оценки Exact Match — то есть после уже состоявшегося обучения, а не перед
    его началом.
    """
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
        return Check("nltk: punkt_tab", True, "данные токенизатора найдены")
    except LookupError:
        return Check(
            "nltk: punkt_tab", False,
            "не найдены — выполните: python -m nltk.downloader punkt_tab",
        )


def check_dataset(name: str, root: Path) -> list[Check]:
    checks: list[Check] = []
    for filename in DATASET_FILES[name]:
        path = root / filename
        checks.append(Check(f"{name}: {filename}", path.is_file(), str(path)))

    databases = sorted((root / "database").glob("*/*.sqlite")) if root.is_dir() else []
    checks.append(
        Check(
            f"{name}: файлы баз данных",
            bool(databases),
            f"найдено {len(databases)} шт." if databases else "не найдены — Execution Accuracy невозможна",
        )
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка окружения kan-lora-text2sql")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    arguments = parser.parse_args()

    checks = [check_torch(), *check_cuda(), check_muon(), check_nltk_punkt()]
    for name in DATASET_FILES:
        checks.extend(check_dataset(name, arguments.data_root / name))

    width = max(len(check.name) for check in checks)
    for check in checks:
        mark = "OK  " if check.ok else "НЕТ "
        print(f"{mark} {check.name.ljust(width)}  {check.detail}")

    failed = [check for check in checks if not check.ok]
    print(f"\nПройдено {len(checks) - len(failed)} из {len(checks)}.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
