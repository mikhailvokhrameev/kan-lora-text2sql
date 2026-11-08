"""Диагностика окружения перед запуском экспериментов.

Проверяет, поддерживает ли сборка PyTorch данную видеокарту, скачаны ли
данные токенизатора nltk (`punkt_tab`), нужные
официальному скрипту оценки Spider, и лежат ли на диске файлы наборов вместе
с базами данных для Execution Accuracy.

Запуск: python scripts/check_environment.py [--data-root data]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from kanlora.data.loaders import LAYOUTS, load_dataset_schemas, load_examples
from kanlora.eval.execution import validate_execution_data

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

    try:
        name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        supported = torch.cuda.get_arch_list()
        arch = f"sm_{capability[0]}{capability[1]}"
        compatible = any(
            item.startswith("sm_") and item[3:].isdigit()
            and int(item[3:]) // 10 == capability[0]
            and int(item[3:]) % 10 <= capability[1]
            for item in supported
        )
        checks = [
            Check("CUDA", True, f"CUDA {torch.version.cuda}, {name}, {arch}"),
            Check(f"поддержка {arch} в сборке", compatible,
                  f"совместимый бинарный код среди: {', '.join(supported)}"),
        ]
        value = torch.ones(2, device="cuda", dtype=torch.float32, requires_grad=True)
        (value * value).sum().backward()
        torch.cuda.synchronize()
        checks.append(Check("CUDA: fp32 forward/backward", True, "выполнено"))
        return checks
    except Exception as error:  # noqa: BLE001 — диагностика должна вывести ошибку
        return [Check("CUDA", False, f"ошибка CUDA: {error}")]



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

    layout = LAYOUTS[name]
    try:
        schemas = load_dataset_schemas(root, layout)
        train = load_examples(root, layout, "train")
        dev = load_examples(root, layout, "eval")
        if not train or not dev:
            raise ValueError("пустое разбиение train или dev")
        missing = sorted({example.db_id for example in (*train, *dev)} - schemas.keys())
        if missing:
            raise ValueError(f"нет схем для db_id: {', '.join(missing)}")
        checks.append(Check(f"{name}: JSON и поля", True, "train, dev и схемы прочитаны"))
        validate_execution_data(
            [example.query for example in dev], [example.db_id for example in dev], root, layout
        )
        checks.append(Check(f"{name}: базы и эталоны dev", True, f"проверено {len(dev)} запросов"))
    except Exception as error:  # noqa: BLE001 — повреждённые данные отражаются в отчёте
        checks.append(Check(f"{name}: данные dev", False, str(error)))

    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка окружения kan-lora-text2sql")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    arguments = parser.parse_args()

    checks = [check_torch(), *check_cuda(), check_nltk_punkt()]
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
