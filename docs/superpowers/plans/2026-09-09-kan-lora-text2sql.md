# KAN-LoRA для генерации SQL: план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Собрать воспроизводимый стенд, который обучает Qwen2.5-Coder-0.5B-Instruct тремя методами адаптации (LoRA, DoRA, KAN-LoRA) на Spider и PAUQ и выдаёт таблицы по Exact Match, Execution Accuracy, пиковой видеопамяти и мере выученной нелинейности.

**Architecture:** Три адаптера реализуют один интерфейс `AdapterLinear` и подставляются вместо `nn.Linear` единым внедрителем, поэтому обучение, оценка и замеры памяти для всех трёх методов идут по одному и тому же коду. Каждый прогон описывается одним конфигурационным файлом и ключами командной строки, а результатом прогона является карточка результата (JSON), которая лежит в git; таблицы и графики собираются из карточек отдельным скриптом и никогда не правятся руками.

**Tech Stack:** Python 3.12, PyTorch 2.14 (fp32, `torch.optim.AdamW` и `torch.optim.Muon`), transformers 4.57, datasets, sqlite3, numpy, matplotlib, pytest.

**Spec:** `~/.claude/plans/ethereal-meandering-dream.md`

---

## Отклонения от спецификации (утверждены до начала работы)

Эти три пункта расходятся с текстом спецификации. Они утверждены и должны применяться именно так.

1. **Вместо `tanh` на входе сплайна — обучаемый масштаб входа `input_scale`.** Уже реализован в `src/kanlora/adapters/kan_layer.py`. Формулировка для защиты: «сетка B-сплайна конечна; вместо того чтобы сплющивать активации в неё через `tanh`, я растягиваю саму сетку под их размах, а долю попаданий в сетку логирую». Причина выбора: `tanh(x) != x`, поэтому с `tanh` тождественная инициализация становится приближённой и ключевое утверждение «KAN-LoRA стартует ровно как LoRA» слабеет; с `input_scale` оно точное. Следствие: **абляционный прогон — это `input_scale`, замороженный на 1.0**, а не «прогон без `tanh`». Число прогонов не меняется (один).
2. **`torch.optim.Muon` существует** в torch 2.14.0 (проверено запуском 2026-09-09). Задача «эталонная реализация Muon отдельным модулем» из спецификации **отменена**.
3. **Наборы данных читаются с диска, а не через `datasets`.** Загрузчики принимают корневой каталог с распакованными JSON-файлами и каталогом баз. Причина: машина обучения может быть без сети, а схема наборов на Hugging Face меняется независимо от работы. `datasets` остаётся в зависимостях только как средство первичной выкачки.

---

## Global Constraints

Требования, действующие для **каждой** задачи ниже. Значения скопированы из спецификации дословно.

- **Язык.** Идентификаторы, имена файлов и сообщения коммитов — английские. Комментарии, строки документации, сообщения об ошибках, содержимое таблиц и подписи графиков — русские, с корректной терминологией: «функция потерь» (не «лосс»), «скорость обучения» (не «лернрейт»), «тонкая настройка» (не «файнтюнинг»).
- **Разработка через тесты, без исключений.** Порядок: падающий тест → запуск, подтверждающий падение **по нужной причине** → минимальная реализация → запуск, подтверждающий прохождение → коммит. Реализация раньше теста не пишется никогда.
- **Точность — fp32 везде и одинаково для всех трёх методов.** Ни `bfloat16`, ни `float16`, ни квантование не применяются: GTX 1080 Ti (Pascal, CC 6.1) не поддерживает bf16 и тензорные ядра, fp16 на GP102 считается в 1/64 скорости, `bitsandbytes` требует CC ≥ 7.5. Внутренние вычисления узлов сплайна и абсцисс Гревилля — `float64` (уже так в `spline.py`).
- **Модель одна:** `Qwen/Qwen2.5-Coder-0.5B-Instruct`.
- **Гиперпараметры зафиксированы один раз и не подбираются:** ранг 8, alpha 16.0, размер сетки 5, порядок сплайна 3, адаптеры на всех семи проекциях (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), одна скорость обучения `2.0e-4` на все три метода, обрезание градиента по норме 1.0, размер батча 1, накопление градиента 8, длина последовательности 768, 2 эпохи, обучающая выборка 2500 примеров, зёрна 0, 1, 2. Оценка — на полном отложенном наборе.
- **Единственный параметр, который допускается менять после старта, — размер обучающей выборки**, и менять его надо один раз для всех прогонов сразу, по итогам замера фактического времени прогона (Задача 19).
- **Видеопамять** меряется только через `torch.cuda.max_memory_allocated` при жёстко одинаковых длине последовательности, размере батча и точности для всех трёх методов.
- **Формулировка KAN-LoRA не меняется молча.** Спайновый базис, ранг, схема активаций и тождественная инициализация зафиксированы; любое изменение требует отдельного обоснования в тексте задачи.
- **Методология не меняется как побочный эффект.** Разбиения, зёрна, гиперпараметры и порядок оценки трогать нельзя при работе над несвязанной задачей.
- **Каждая задача заканчивается зелёным `python -m pytest -q` целиком**, а не только своими тестами.
- **Коммиты частые**, по одному на задачу минимум, сообщение в стиле `feat: ...` / `test: ...` / `chore: ...`.

---

## Текущее состояние репозитория

Проверено 2026-09-09. **Уже сделано, переделывать не нужно:**

- `pyproject.toml` — зависимости, `pythonpath = ["src"]`, `testpaths = ["tests"]`.
- `src/kanlora/adapters/spline.py` — `make_knots`, `bspline_basis`, `greville_abscissae`, `num_basis_functions`.
- `src/kanlora/adapters/kan_layer.py` — `KANLayer` с тождественной инициализацией через абсциссы Гревилля, обучаемым `input_scale`, методами `reset_to_identity`, `fraction_inside_grid`, `parameter_count`.
- `tests/adapters/test_spline.py`, `tests/adapters/test_kan_layer.py` — 17 тестов, все проходят.

**Окружение разработки:** Python 3.12.4, torch 2.14.0 (`torch.optim.Muon` есть), transformers 4.57.6, numpy 1.26.4, pyyaml 6.0.1. **Не установлены:** `datasets`, `accelerate`, `matplotlib`.

---

## Структура файлов

Каждый модуль отвечает на один вопрос и тестируется отдельно от обучения.

```
src/kanlora/
  config.py             Чтение YAML в неизменяемую конфигурацию прогона.
  results.py            Карточка результата: сбор версий, коммита, запись JSON.
  adapters/
    spline.py           [готово] B-сплайновый базис. Чистая математика.
    kan_layer.py        [готово] Слой Колмогорова — Арнольда, тождественный на старте.
    base.py             Общий интерфейс AdapterLinear + подсчёт параметров.
    lora.py             Классическая LoRA.
    dora.py             DoRA: разложение веса на модуль и направление.
    kan_lora.py         Нелинейный адаптер B*phi(Ax).
    inject.py           Замена nn.Linear. Единая для всех трёх методов.
  data/
    schema.py           Сериализация схемы БД. Одна на все методы, зафиксирована.
    loaders.py          Общий читатель JSON-файлов формата Spider.
    spider.py           Раскладка файлов Spider.
    pauq.py             Раскладка файлов PAUQ (pauq_xsp).
    collate.py          Промпт, токенизация, маска функции потерь по SQL, усечение.
  train/
    optimizers.py       AdamW и Muon за единым интерфейсом, разводка по 2D и не-2D.
    memory.py           Замер пиковой видеопамяти в фиксированных условиях.
    loop.py             Цикл обучения: чекпоинтинг, накопление, обрезание градиента.
  eval/
    generate.py         Пакетная жадная генерация.
    exact_match.py      Обёртка над официальной метрикой Spider.
    execution.py        Исполнение SQL на sqlite, сравнение результатов.
    latency.py          Задержка генерации со слиянием адаптера и без.
  analysis/
    spline_stats.py     Подгонка прямой к выученным функциям, доля активаций в сетке.
third_party/spider_eval/  Официальный сценарий оценки Spider (вендорится как есть).
configs/                base.yaml и smoke.yaml; варианты задаются ключами командной строки.
scripts/
  check_environment.py  Проверка железа, Muon, наличия наборов и баз данных.
  run_experiment.py     Один прогон: обучение -> оценка -> карточка результата.
  make_tables.py        Сборка таблиц и графиков из карточек.
results/                Карточки результатов (JSON) — в git.
tests/                  Зеркалит структуру src/, пишется первым.
```

**Почему `data/loaders.py` появился сверх спецификации.** Spider и PAUQ хранятся в одном формате JSON и различаются только именами файлов, поэтому читатель один, а `spider.py` и `pauq.py` держат по одной константе раскладки. Дублировать читатель дважды означало бы завести два места, где преобразование примеров может незаметно разойтись, — а именно расхождение предобработки между наборами и обесценивает сравнение.

---

## Порядок задач

Задачи 1–11 не требуют GPU и делаются на Mac. Задачи 12–14, 16–18 требуют модели, но проверяются на процессоре крошечными заглушками. Задача 19 — единственная, которой нужна GTX 1080 Ti.

---

### Задача 1: Проверка окружения и наличия данных

Первое, что должно заработать: одна команда, отвечающая на вопросы «есть ли CUDA на этой карте», «есть ли Muon», «лежат ли файлы наборов и баз». Без неё риски из спецификации (Pascal выпал из сборки torch, файлы баз PAUQ недоступны) обнаружатся не на неделе 1, а на неделе 3.

**Files:**
- Create: `scripts/check_environment.py`
- Create: `.gitignore`
- Create: `tests/test_check_environment.py`
- Modify: `pyproject.toml` (добавить `matplotlib`)

**Interfaces:**
- Consumes: ничего.
- Produces: `scripts/check_environment.py` с функциями `check_torch() -> Check`, `check_muon() -> Check`, `check_dataset(name: str, root: Path) -> list[Check]`, `main() -> int`; датакласс `Check(name: str, ok: bool, detail: str)`. Задача 8 использует константы раскладки из `kanlora.data.spider` / `kanlora.data.pauq`, но на этом шаге пути передаются строками, поэтому зависимости нет.

- [ ] **Шаг 1: Установить недостающие зависимости**

```bash
python -m pip install "datasets>=3.0" "accelerate>=1.0" "matplotlib>=3.8"
```

Добавить `matplotlib>=3.8` в `dependencies` в `pyproject.toml` (нужен для графиков в Задаче 18):

```toml
dependencies = [
    "torch>=2.14",
    "transformers>=4.46",
    "datasets>=3.0",
    "accelerate>=1.0",
    "matplotlib>=3.8",
    "pyyaml>=6.0",
    "numpy>=1.26",
]
```

- [ ] **Шаг 2: Написать `.gitignore`**

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
*.egg-info/
data/
third_party/spider_eval/__pycache__/
results/figures/
```

Каталог `data/` игнорируется: наборы весят гигабайты и выкачиваются на каждой машине отдельно. Карточки результатов в `results/*.json` **не** игнорируются — они и есть поставка.

- [ ] **Шаг 3: Написать падающий тест**

Проверяется не наличие CUDA (на Mac её нет), а то, что проверка отчитывается структурировано и не падает исключением на отсутствующих путях.

```python
# tests/test_check_environment.py
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
```

- [ ] **Шаг 4: Запустить тест, убедиться, что он падает**

Run: `python -m pytest tests/test_check_environment.py -v`
Expected: FAIL — `FileNotFoundError` при загрузке модуля, файла `scripts/check_environment.py` ещё нет.

- [ ] **Шаг 5: Написать минимальную реализацию**

```python
# scripts/check_environment.py
"""Диагностика окружения перед запуском экспериментов.

Отвечает на три вопроса, каждый из которых способен сорвать работу целиком:
поддерживает ли сборка PyTorch данную видеокарту, есть ли Muon в torch.optim,
и лежат ли на диске файлы наборов вместе с базами данных для Execution Accuracy.

Запуск: python scripts/check_environment.py [--data-root data]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

DATASET_FILES = {
    "spider": ("train_spider.json", "dev.json", "tables.json"),
    "pauq": ("pauq_xsp_train.json", "pauq_xsp_test.json", "tables.json"),
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

    checks = [check_torch(), *check_cuda(), check_muon()]
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
```

- [ ] **Шаг 6: Запустить тесты, убедиться, что они проходят**

Run: `python -m pytest -q`
Expected: PASS, 21 тест (17 прежних + 4 новых).

- [ ] **Шаг 7: Запустить проверку вручную и прочитать вывод**

Run: `python scripts/check_environment.py`
Expected: `torch` — OK, `CUDA` — НЕТ (на Mac это нормально), `Muon` — OK, наборы — НЕТ.

**Эту же команду надо выполнить на машине с GTX 1080 Ti при первой же возможности.** Если строка «поддержка sm_61 в сборке» вернёт НЕТ — вся работа блокирована, и об этом надо сообщить преподавателю немедленно, а не в конце месяца. Обходной путь: откатиться на сборку PyTorch, где sm_61 ещё поддержан, и тогда `torch.optim.Muon` пропадёт вместе с ней — прогоны с Muon придётся снять (спецификация именно их называет первым кандидатом на снятие).

- [ ] **Шаг 8: Выкачать наборы данных**

Spider: официальный архив с сайта Yale LILY, распаковать в `data/spider/` так, чтобы получились `data/spider/train_spider.json`, `data/spider/dev.json`, `data/spider/tables.json`, `data/spider/database/<db_id>/<db_id>.sqlite`.

PAUQ: репозиторий `ai-spiderweb/pauq`, взять **только разбиение `pauq_xsp`**, распаковать в `data/pauq/` в ту же раскладку. Базы данных PAUQ — русскоязычная копия баз Spider.

Затем снова:

Run: `python scripts/check_environment.py`
Expected: все строки по наборам — OK.

**Если файлы баз PAUQ отсутствуют** (`pauq: файлы баз данных` — НЕТ): это заранее предусмотренное отступление. Действия: на PAUQ считается только Exact Match; в карточке результата `execution_accuracy` остаётся `null`; факт оговаривается в тексте работы; **преподавателю сообщается сразу.** Если фактические имена файлов PAUQ отличаются от значений в `DATASET_FILES`, поправить константу здесь и синхронно в `src/kanlora/data/pauq.py` (Задача 8).

- [ ] **Шаг 9: Коммит**

```bash
git add .gitignore pyproject.toml scripts/check_environment.py tests/test_check_environment.py
git commit -m "feat: add environment and dataset availability check"
```

---

### Задача 2: Общий интерфейс адаптера

Все три метода обязаны идти по одному коду обучения и оценки — иначе сравнение сравнивает реализации, а не методы. Интерфейс задаёт эту общность и заодно даёт подсчёт параметров, на котором стоит утверждение о равном бюджете.

**Files:**
- Create: `src/kanlora/adapters/base.py`
- Create: `tests/adapters/test_base.py`

**Interfaces:**
- Consumes: ничего.
- Produces:
  - `AdapterConfig(rank: int = 8, alpha: float = 16.0, grid_size: int = 5, spline_order: int = 3, learn_input_scale: bool = True)` — замороженный датакласс.
  - `AdapterLinear(nn.Module)` — абстрактный, конструктор `__init__(self, base: nn.Linear, config: AdapterConfig)`; поля `self.base`, `self.config`, `self.scaling: float = alpha / rank`; методы `forward(x: Tensor) -> Tensor` (абстрактный), `analytic_parameter_count() -> int` (абстрактный), `trainable_parameter_count() -> int`, свойство `can_merge: bool` (абстрактное), `merge() -> nn.Linear` (по умолчанию бросает `NotImplementedError`).
  - `freeze(module: nn.Module) -> None`.

- [ ] **Шаг 1: Написать падающий тест**

```python
# tests/adapters/test_base.py
"""Проверки общего интерфейса адаптеров.

Интерфейс существует ради одного: обучение, оценка и замер памяти для LoRA,
DoRA и KAN-LoRA обязаны идти по одному и тому же коду. Тесты закрепляют то,
на что этот код опирается: замороженную основу, счётчик обучаемых параметров
и явный ответ на вопрос, сливается ли адаптер с весами.
"""

import pytest
import torch
from torch import nn

from kanlora.adapters.base import AdapterConfig, AdapterLinear, freeze

IN_FEATURES, OUT_FEATURES = 6, 4


class ConstantAdapter(AdapterLinear):
    """Простейшая реализация интерфейса — только чтобы проверить сам интерфейс."""

    def __init__(self, base: nn.Linear, config: AdapterConfig) -> None:
        super().__init__(base, config)
        self.shift = nn.Parameter(torch.zeros(base.out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.shift

    def analytic_parameter_count(self) -> int:
        return self.base.out_features

    @property
    def can_merge(self) -> bool:
        return False


@pytest.fixture
def adapter() -> ConstantAdapter:
    torch.manual_seed(0)
    return ConstantAdapter(nn.Linear(IN_FEATURES, OUT_FEATURES), AdapterConfig())


def test_scaling_is_alpha_over_rank() -> None:
    base = nn.Linear(IN_FEATURES, OUT_FEATURES)
    adapter = ConstantAdapter(base, AdapterConfig(rank=8, alpha=16.0))
    assert adapter.scaling == pytest.approx(2.0)


def test_base_is_frozen(adapter: ConstantAdapter) -> None:
    assert adapter.base.weight.requires_grad is False
    assert adapter.base.bias.requires_grad is False


def test_trainable_count_excludes_frozen_base(adapter: ConstantAdapter) -> None:
    assert adapter.trainable_parameter_count() == OUT_FEATURES
    assert adapter.trainable_parameter_count() == adapter.analytic_parameter_count()


def test_freeze_marks_every_parameter(adapter: ConstantAdapter) -> None:
    freeze(adapter)
    assert all(not p.requires_grad for p in adapter.parameters())


def test_non_mergeable_adapter_refuses_to_merge(adapter: ConstantAdapter) -> None:
    assert adapter.can_merge is False
    with pytest.raises(NotImplementedError):
        adapter.merge()


def test_interface_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        AdapterLinear(nn.Linear(IN_FEATURES, OUT_FEATURES), AdapterConfig())
```

- [ ] **Шаг 2: Запустить тест, убедиться, что он падает**

Run: `python -m pytest tests/adapters/test_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kanlora.adapters.base'`.

- [ ] **Шаг 3: Написать минимальную реализацию**

```python
# src/kanlora/adapters/base.py
"""Общий интерфейс адаптеров.

Все три сравниваемых метода подставляются вместо nn.Linear и обязаны быть
взаимозаменяемы для цикла обучения, генерации и замера памяти. Различия
методов не должны просачиваться в код вокруг: иначе сравнение начнёт мерить
качество реализаций, а не свойства методов.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn

__all__ = ["AdapterConfig", "AdapterLinear", "freeze"]


@dataclass(frozen=True)
class AdapterConfig:
    """Гиперпараметры адаптера, общие для всех трёх методов.

    Заморожен намеренно: гиперпараметры фиксируются один раз и не подбираются,
    поэтому случайная правка поля в середине матрицы прогонов невозможна.
    Поля grid_size, spline_order и learn_input_scale используются только
    KAN-LoRA; для LoRA и DoRA они игнорируются и хранятся ради того, чтобы
    карточка результата содержала одинаковый набор полей для всех прогонов.
    """

    rank: int = 8
    alpha: float = 16.0
    grid_size: int = 5
    spline_order: int = 3
    learn_input_scale: bool = True


def freeze(module: nn.Module) -> None:
    """Снимает requires_grad со всех параметров модуля."""
    for parameter in module.parameters():
        parameter.requires_grad_(False)


class AdapterLinear(nn.Module, ABC):
    """Замороженный nn.Linear плюс обучаемая поправка."""

    def __init__(self, base: nn.Linear, config: AdapterConfig) -> None:
        super().__init__()
        self.base = base
        self.config = config
        self.scaling = config.alpha / config.rank
        freeze(self.base)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def analytic_parameter_count(self) -> int:
        """Число обучаемых параметров по формуле, независимо от реализации.

        Существует ради теста: расхождение с фактическим счётчиком означает,
        что утверждение о равном бюджете параметров в таблицах неверно.
        """

    @property
    @abstractmethod
    def can_merge(self) -> bool:
        """Сливается ли поправка с весами основы после обучения."""

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def merge(self) -> nn.Linear:
        """Возвращает обычный nn.Linear с поглощённой поправкой."""
        raise NotImplementedError(
            f"{type(self).__name__} нелинеен и не сливается с весами: "
            "поправка зависит от входа, а не только от весов"
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.base.in_features}, out_features={self.base.out_features}, "
            f"rank={self.config.rank}, alpha={self.config.alpha}"
        )
```

- [ ] **Шаг 4: Запустить тесты, убедиться, что они проходят**

Run: `python -m pytest -q`
Expected: PASS.

- [ ] **Шаг 5: Коммит**

```bash
git add src/kanlora/adapters/base.py tests/adapters/test_base.py
git commit -m "feat: add common adapter interface"
```

---

### Задача 3: LoRA

Опорный метод. Всё остальное сравнивается с ним, поэтому он реализуется первым и максимально прямолинейно.

**Files:**
- Create: `src/kanlora/adapters/lora.py`
- Create: `tests/adapters/test_lora.py`

**Interfaces:**
- Consumes: `AdapterConfig`, `AdapterLinear` из `kanlora.adapters.base`.
- Produces: `LoRALinear(AdapterLinear)` с полями `lora_a: nn.Parameter` формы `(rank, in_features)` и `lora_b: nn.Parameter` формы `(out_features, rank)`; метод `delta(x: Tensor) -> Tensor` возвращает `scaling * (lora_b @ lora_a @ x)`; `can_merge` равно `True`; `merge() -> nn.Linear`.

- [ ] **Шаг 1: Написать падающий тест**

```python
# tests/adapters/test_lora.py
"""Проверки классической LoRA.

Главное здесь — нулевая поправка при инициализации. Все три метода обязаны
стартовать из одной точки, иначе разницу в качестве нельзя приписать методу.
"""

import pytest
import torch
from torch import nn

from kanlora.adapters.base import AdapterConfig
from kanlora.adapters.lora import LoRALinear

IN_FEATURES, OUT_FEATURES, RANK = 12, 7, 4


@pytest.fixture
def base() -> nn.Linear:
    torch.manual_seed(0)
    return nn.Linear(IN_FEATURES, OUT_FEATURES, dtype=torch.float64)


@pytest.fixture
def adapter(base: nn.Linear) -> LoRALinear:
    return LoRALinear(base, AdapterConfig(rank=RANK, alpha=2.0 * RANK))


def test_output_shape(adapter: LoRALinear) -> None:
    x = torch.randn(3, 5, IN_FEATURES, dtype=torch.float64)
    assert adapter(x).shape == (3, 5, OUT_FEATURES)


def test_zero_correction_at_initialization(base: nn.Linear, adapter: LoRALinear) -> None:
    """B = 0, поэтому модель с адаптером в точности равна модели без него."""
    x = torch.randn(16, IN_FEATURES, dtype=torch.float64)
    assert torch.allclose(adapter(x), base(x), atol=1e-12)


def test_matrix_a_is_not_zero(adapter: LoRALinear) -> None:
    """Обнулять надо ровно одну матрицу: при A = B = 0 градиент тождественно нулевой."""
    assert adapter.lora_a.abs().sum().item() > 0
    assert adapter.lora_b.abs().sum().item() == pytest.approx(0.0)


def test_delta_equals_scaled_low_rank_product(adapter: LoRALinear) -> None:
    with torch.no_grad():
        adapter.lora_b.normal_()

    x = torch.randn(9, IN_FEATURES, dtype=torch.float64)
    expected = adapter.scaling * (x @ adapter.lora_a.T @ adapter.lora_b.T)
    assert torch.allclose(adapter.delta(x), expected, atol=1e-12)


def test_parameter_count_matches_formula(adapter: LoRALinear) -> None:
    expected = RANK * (IN_FEATURES + OUT_FEATURES)
    assert adapter.analytic_parameter_count() == expected
    assert adapter.trainable_parameter_count() == expected


def test_merged_linear_reproduces_adapter_output(base: nn.Linear, adapter: LoRALinear) -> None:
    """LoRA сливается с весами и на инференсе не стоит ничего — в этом её отличие."""
    with torch.no_grad():
        adapter.lora_b.normal_()

    assert adapter.can_merge is True
    merged = adapter.merge()
    x = torch.randn(16, IN_FEATURES, dtype=torch.float64)
    assert torch.allclose(merged(x), adapter(x), atol=1e-10)


def test_merge_does_not_touch_the_original_base(base: nn.Linear, adapter: LoRALinear) -> None:
    original = base.weight.detach().clone()
    with torch.no_grad():
        adapter.lora_b.normal_()
    adapter.merge()
    assert torch.equal(base.weight, original)


def test_gradients_reach_both_matrices(adapter: LoRALinear) -> None:
    with torch.no_grad():
        adapter.lora_b.normal_()

    adapter(torch.randn(4, IN_FEATURES, dtype=torch.float64)).sum().backward()
    assert adapter.lora_a.grad is not None
    assert adapter.lora_b.grad is not None
    assert adapter.base.weight.grad is None
```

- [ ] **Шаг 2: Запустить тест, убедиться, что он падает**

Run: `python -m pytest tests/adapters/test_lora.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kanlora.adapters.lora'`.

- [ ] **Шаг 3: Написать минимальную реализацию**

```python
# src/kanlora/adapters/lora.py
"""Классическая низкоранговая адаптация.

Поправка к весу раскладывается в произведение двух матриц ранга r:
    delta_W * x = (alpha / r) * B (A x),   A: (r, in),   B: (out, r).
Матрица B инициализируется нулём, поэтому в начале обучения модель не изменена.
Обнуляется ровно одна из матриц: при A = B = 0 градиент по обеим тождественно
равен нулю и адаптер не сдвинулся бы с места.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from kanlora.adapters.base import AdapterConfig, AdapterLinear

__all__ = ["LoRALinear"]


class LoRALinear(AdapterLinear):
    def __init__(self, base: nn.Linear, config: AdapterConfig) -> None:
        super().__init__(base, config)
        factory = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_a = nn.Parameter(torch.empty(config.rank, base.in_features, **factory))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, config.rank, **factory))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        projected = torch.nn.functional.linear(x, self.lora_a)
        return self.scaling * torch.nn.functional.linear(projected, self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.delta(x)

    def analytic_parameter_count(self) -> int:
        return self.config.rank * (self.base.in_features + self.base.out_features)

    @property
    def can_merge(self) -> bool:
        return True

    def merge(self) -> nn.Linear:
        merged = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        with torch.no_grad():
            merged.weight.copy_(self.base.weight + self.scaling * (self.lora_b @ self.lora_a))
            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias)
        return merged
```

- [ ] **Шаг 4: Запустить тесты, убедиться, что они проходят**

Run: `python -m pytest -q`
Expected: PASS.

- [ ] **Шаг 5: Коммит**

```bash
git add src/kanlora/adapters/lora.py tests/adapters/test_lora.py
git commit -m "feat: add LoRA adapter"
```

---

### Задача 4: DoRA

Второй опорный метод. Ключевое свойство, которое надо уметь произнести на защите: DoRA раскладывает вес на **модуль и направление**, обучает направление через ту же низкоранговую поправку, а модуль — отдельным вектором.

**Files:**
- Create: `src/kanlora/adapters/dora.py`
- Create: `tests/adapters/test_dora.py`

**Interfaces:**
- Consumes: `AdapterConfig`, `AdapterLinear` из `kanlora.adapters.base`.
- Produces: `DoRALinear(AdapterLinear)` с полями `lora_a` формы `(rank, in_features)`, `lora_b` формы `(out_features, rank)`, `magnitude: nn.Parameter` формы `(out_features,)`; метод `effective_weight() -> Tensor` формы `(out_features, in_features)`; `can_merge` равно `True`; `merge() -> nn.Linear`.

- [ ] **Шаг 1: Написать падающий тест**

```python
# tests/adapters/test_dora.py
"""Проверки DoRA.

DoRA раскладывает вес на модуль и направление: W = m * V / ||V||, где норма
берётся построчно, по каждому выходному каналу. Тесты закрепляют два факта,
без которых сравнение теряет смысл: старт из немодифицированной модели и
сводимость к LoRA, когда модуль равен норме поправленного веса.
"""

import pytest
import torch
from torch import nn

from kanlora.adapters.base import AdapterConfig
from kanlora.adapters.dora import DoRALinear
from kanlora.adapters.lora import LoRALinear

IN_FEATURES, OUT_FEATURES, RANK = 12, 7, 4


@pytest.fixture
def base() -> nn.Linear:
    torch.manual_seed(0)
    return nn.Linear(IN_FEATURES, OUT_FEATURES, dtype=torch.float64)


@pytest.fixture
def config() -> AdapterConfig:
    return AdapterConfig(rank=RANK, alpha=2.0 * RANK)


@pytest.fixture
def adapter(base: nn.Linear, config: AdapterConfig) -> DoRALinear:
    return DoRALinear(base, config)


def test_output_shape(adapter: DoRALinear) -> None:
    x = torch.randn(3, 5, IN_FEATURES, dtype=torch.float64)
    assert adapter(x).shape == (3, 5, OUT_FEATURES)


def test_magnitude_is_initialized_to_row_norms(base: nn.Linear, adapter: DoRALinear) -> None:
    expected = base.weight.norm(dim=1)
    assert adapter.magnitude.shape == (OUT_FEATURES,)
    assert torch.allclose(adapter.magnitude, expected, atol=1e-12)


def test_zero_correction_at_initialization(base: nn.Linear, adapter: DoRALinear) -> None:
    """m равен норме W, а B = 0, поэтому m * W / ||W|| в точности равно W."""
    x = torch.randn(16, IN_FEATURES, dtype=torch.float64)
    assert torch.allclose(adapter(x), base(x), atol=1e-10)


def test_reduces_to_lora_when_magnitude_matches_direction_norm(
    base: nn.Linear, config: AdapterConfig
) -> None:
    """При m = ||W + s*BA|| нормировка сокращается и DoRA совпадает с LoRA.

    Это и есть смысл утверждения «DoRA обобщает LoRA»: без отдельного модуля
    остаётся ровно LoRA.
    """
    torch.manual_seed(1)
    dora = DoRALinear(base, config)
    lora = LoRALinear(base, config)
    with torch.no_grad():
        lora.lora_a.copy_(dora.lora_a)
        dora.lora_b.normal_()
        lora.lora_b.copy_(dora.lora_b)
        direction = base.weight + dora.scaling * (dora.lora_b @ dora.lora_a)
        dora.magnitude.copy_(direction.norm(dim=1))

    x = torch.randn(16, IN_FEATURES, dtype=torch.float64)
    assert torch.allclose(dora(x), lora(x), atol=1e-10)


def test_parameter_count_matches_formula(adapter: DoRALinear) -> None:
    expected = RANK * (IN_FEATURES + OUT_FEATURES) + OUT_FEATURES
    assert adapter.analytic_parameter_count() == expected
    assert adapter.trainable_parameter_count() == expected


def test_merged_linear_reproduces_adapter_output(adapter: DoRALinear) -> None:
    with torch.no_grad():
        adapter.lora_b.normal_()
        adapter.magnitude.mul_(1.3)

    assert adapter.can_merge is True
    merged = adapter.merge()
    x = torch.randn(16, IN_FEATURES, dtype=torch.float64)
    assert torch.allclose(merged(x), adapter(x), atol=1e-10)


def test_gradients_reach_magnitude_and_both_matrices(adapter: DoRALinear) -> None:
    with torch.no_grad():
        adapter.lora_b.normal_()

    adapter(torch.randn(4, IN_FEATURES, dtype=torch.float64)).sum().backward()
    for name in ("lora_a", "lora_b", "magnitude"):
        parameter = getattr(adapter, name)
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert adapter.base.weight.grad is None
```

- [ ] **Шаг 2: Запустить тест, убедиться, что он падает**

Run: `python -m pytest tests/adapters/test_dora.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kanlora.adapters.dora'`.

- [ ] **Шаг 3: Написать минимальную реализацию**

```python
# src/kanlora/adapters/dora.py
"""Адаптация с разложением веса на модуль и направление (DoRA).

Вес раскладывается как W = m * V / ||V||, где норма берётся по каждому
выходному каналу отдельно. Направление V получает ту же низкоранговую поправку,
что и в LoRA, а модуль m обучается отдельным вектором. Замысел метода: в ходе
полной тонкой настройки модуль и направление меняются по-разному, а LoRA
вынуждена менять их совместно.

При инициализации m равен норме исходного веса, а B = 0, поэтому
m * W / ||W|| в точности равно W и модель стартует немодифицированной —
как и два других сравниваемых метода.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from kanlora.adapters.base import AdapterConfig, AdapterLinear

__all__ = ["DoRALinear"]


class DoRALinear(AdapterLinear):
    def __init__(self, base: nn.Linear, config: AdapterConfig) -> None:
        super().__init__(base, config)
        factory = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_a = nn.Parameter(torch.empty(config.rank, base.in_features, **factory))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, config.rank, **factory))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        with torch.no_grad():
            self.magnitude = nn.Parameter(base.weight.norm(dim=1).clone())

    def effective_weight(self) -> torch.Tensor:
        direction = self.base.weight + self.scaling * (self.lora_b @ self.lora_a)
        norm = direction.norm(dim=1, keepdim=True)
        return self.magnitude.unsqueeze(1) * direction / norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.effective_weight(), self.base.bias)

    def analytic_parameter_count(self) -> int:
        return (
            self.config.rank * (self.base.in_features + self.base.out_features)
            + self.base.out_features
        )

    @property
    def can_merge(self) -> bool:
        return True

    def merge(self) -> nn.Linear:
        merged = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        with torch.no_grad():
            merged.weight.copy_(self.effective_weight())
            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias)
        return merged
```

- [ ] **Шаг 4: Запустить тесты, убедиться, что они проходят**

Run: `python -m pytest -q`
Expected: PASS.

- [ ] **Шаг 5: Коммит**

```bash
git add src/kanlora/adapters/dora.py tests/adapters/test_dora.py
git commit -m "feat: add DoRA adapter"
```

---

### Задача 5: KAN-LoRA

Исследуемый метод и центр всей работы. `KANLayer` уже готов; здесь он вставляется между низкоранговыми матрицами: `delta_W * x = B * phi(A x)`.

**Главный тест задачи** — численное совпадение с LoRA при замороженных тождественных сплайнах. Он и есть ответ на вопрос комиссии «почему вы уверены, что ваш KAN-LoRA действительно обобщает LoRA».

**Files:**
- Create: `src/kanlora/adapters/kan_lora.py`
- Create: `tests/adapters/test_kan_lora.py`

**Interfaces:**
- Consumes: `AdapterConfig`, `AdapterLinear` из `kanlora.adapters.base`; `LoRALinear` из `kanlora.adapters.lora` (только в тестах); `KANLayer` из `kanlora.adapters.kan_layer`.
- Produces: `KANLoRALinear(AdapterLinear)` с полями `lora_a` формы `(rank, in_features)`, `lora_b` формы `(out_features, rank)`, `kan: KANLayer`; методы `delta(x: Tensor) -> Tensor`, `last_fraction_inside_grid() -> float | None` (доля активаций внутри сетки на последнем проходе, `None` до первого прохода); `can_merge` равно `False`.

- [ ] **Шаг 1: Написать падающий тест**

```python
# tests/adapters/test_kan_lora.py
"""Проверки нелинейного адаптера B * phi(A x).

Замысел всей работы держится на одном свойстве: при тождественных сплайнах
KAN-LoRA численно совпадает с LoRA. Тогда любое расхождение в качестве после
обучения объясняется выученной нелинейностью, а не другой точкой старта или
другим масштабом градиентов. Это свойство проверяется здесь напрямую.
"""

import pytest
import torch
from torch import nn

from kanlora.adapters.base import AdapterConfig
from kanlora.adapters.kan_lora import KANLoRALinear
from kanlora.adapters.lora import LoRALinear

IN_FEATURES, OUT_FEATURES, RANK = 12, 7, 4
GRID_SIZE, SPLINE_ORDER = 5, 3


@pytest.fixture
def base() -> nn.Linear:
    torch.manual_seed(0)
    return nn.Linear(IN_FEATURES, OUT_FEATURES, dtype=torch.float64)


@pytest.fixture
def config() -> AdapterConfig:
    return AdapterConfig(
        rank=RANK, alpha=2.0 * RANK, grid_size=GRID_SIZE, spline_order=SPLINE_ORDER
    )


@pytest.fixture
def adapter(base: nn.Linear, config: AdapterConfig) -> KANLoRALinear:
    return KANLoRALinear(base, config)


def small_input(rows: int = 16) -> torch.Tensor:
    """Вход, при котором A x гарантированно попадает внутрь сетки [-1, 1].

    Масштаб подобран так, чтобы проверялось совпадение с LoRA, а не поведение
    сплайна за границами сетки, — за границы отвечает отдельный тест.
    """
    torch.manual_seed(7)
    return 0.02 * torch.randn(rows, IN_FEATURES, dtype=torch.float64)


def test_output_shape(adapter: KANLoRALinear) -> None:
    x = torch.randn(3, 5, IN_FEATURES, dtype=torch.float64)
    assert adapter(x).shape == (3, 5, OUT_FEATURES)


def test_zero_correction_at_initialization(base: nn.Linear, adapter: KANLoRALinear) -> None:
    x = torch.randn(16, IN_FEATURES, dtype=torch.float64)
    assert torch.allclose(adapter(x), base(x), atol=1e-12)


def test_matches_lora_with_identity_splines(base: nn.Linear, config: AdapterConfig) -> None:
    """ГЛАВНЫЙ ТЕСТ. При phi = тождество KAN-LoRA поэлементно совпадает с LoRA.

    Совпадение точное, а не приближённое: тождественность обеспечена свойством
    линейной точности B-сплайнов, а не подгонкой коэффициентов.
    """
    torch.manual_seed(1)
    kan_lora = KANLoRALinear(base, config)
    lora = LoRALinear(base, config)
    with torch.no_grad():
        lora.lora_a.copy_(kan_lora.lora_a)
        kan_lora.lora_b.normal_()
        lora.lora_b.copy_(kan_lora.lora_b)

    x = small_input()
    assert kan_lora.kan.fraction_inside_grid(x @ kan_lora.lora_a.T).item() == pytest.approx(1.0)
    assert torch.allclose(kan_lora(x), lora(x), atol=1e-10)


def test_learned_nonlinearity_moves_output_away_from_lora(
    base: nn.Linear, config: AdapterConfig
) -> None:
    """Обратная сторона главного теста: со сдвинутыми сплайнами выходы расходятся.

    Без неё главный тест проходил бы и на реализации, где сплайны ни на что
    не влияют, — то есть на скрытой LoRA.
    """
    torch.manual_seed(1)
    kan_lora = KANLoRALinear(base, config)
    lora = LoRALinear(base, config)
    with torch.no_grad():
        lora.lora_a.copy_(kan_lora.lora_a)
        kan_lora.lora_b.normal_()
        lora.lora_b.copy_(kan_lora.lora_b)
        kan_lora.kan.spline_coefficients.add_(0.3 * torch.randn_like(kan_lora.kan.spline_coefficients))

    x = small_input()
    assert not torch.allclose(kan_lora(x), lora(x), atol=1e-4)


def test_parameter_count_matches_formula(adapter: KANLoRALinear) -> None:
    """Формула: LoRA + слой KAN размера r x r.

    Слой KAN добавляет r^2 * (grid + order) коэффициентов сплайна,
    r^2 масштабов сплайна, r^2 весов базовой ветви и r масштабов входа.
    """
    basis = GRID_SIZE + SPLINE_ORDER
    expected = (
        RANK * (IN_FEATURES + OUT_FEATURES) + RANK * RANK * (basis + 2) + RANK
    )
    assert adapter.analytic_parameter_count() == expected
    assert adapter.trainable_parameter_count() == expected


def test_refuses_to_merge(adapter: KANLoRALinear) -> None:
    """Нелинейный адаптер не сливается с весами и платит задержкой всегда."""
    assert adapter.can_merge is False
    with pytest.raises(NotImplementedError):
        adapter.merge()


def test_reports_fraction_inside_grid_after_forward(adapter: KANLoRALinear) -> None:
    """Диагностика вырождения снимается на каждом проходе, а не отдельным прогоном."""
    assert adapter.last_fraction_inside_grid() is None
    adapter(small_input())
    assert adapter.last_fraction_inside_grid() == pytest.approx(1.0)

    adapter(1e4 * torch.randn(16, IN_FEATURES, dtype=torch.float64))
    assert adapter.last_fraction_inside_grid() < 0.5


def test_gradients_reach_spline_coefficients(adapter: KANLoRALinear) -> None:
    with torch.no_grad():
        adapter.lora_b.normal_()

    adapter(small_input(4)).sum().backward()
    for name in ("lora_a", "lora_b"):
        assert getattr(adapter, name).grad is not None, name
    assert adapter.kan.spline_coefficients.grad is not None
    assert torch.isfinite(adapter.kan.spline_coefficients.grad).all()
    assert adapter.base.weight.grad is None


def test_frozen_input_scale_is_excluded_from_training(base: nn.Linear) -> None:
    """Режим абляции: сетка не растягивается, масштаб входа заморожен на 1.0."""
    config = AdapterConfig(
        rank=RANK, alpha=2.0 * RANK, grid_size=GRID_SIZE,
        spline_order=SPLINE_ORDER, learn_input_scale=False,
    )
    adapter = KANLoRALinear(base, config)

    assert adapter.kan.input_scale.requires_grad is False
    assert torch.allclose(adapter.kan.input_scale, torch.ones_like(adapter.kan.input_scale))
    assert adapter.analytic_parameter_count() == adapter.trainable_parameter_count()
```

- [ ] **Шаг 2: Запустить тест, убедиться, что он падает**

Run: `python -m pytest tests/adapters/test_kan_lora.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kanlora.adapters.kan_lora'`.

- [ ] **Шаг 3: Написать минимальную реализацию**

```python
# src/kanlora/adapters/kan_lora.py
"""Нелинейная низкоранговая адаптация.

    delta_W * x = (alpha / r) * B * phi(A x),

где phi — слой Колмогорова — Арнольда: обучаемые одномерные функции на рёбрах,
заданные B-сплайнами. Отличие от LoRA ровно одно — между двумя низкоранговыми
матрицами появляется нелинейность.

Слой phi при инициализации тождественен, а B обнулена, поэтому KAN-LoRA
стартует численно совпадающим с LoRA. Это и делает сравнение осмысленным:
всё последующее расхождение приписывается выученной нелинейности.

Сливаться с весами адаптер не может в принципе: поправка зависит от входа
нелинейно, поэтому задержку на инференсе метод платит всегда. Это его цена,
и она замеряется отдельно.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from kanlora.adapters.base import AdapterConfig, AdapterLinear
from kanlora.adapters.kan_layer import KANLayer

__all__ = ["KANLoRALinear"]


class KANLoRALinear(AdapterLinear):
    def __init__(self, base: nn.Linear, config: AdapterConfig) -> None:
        super().__init__(base, config)
        factory = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_a = nn.Parameter(torch.empty(config.rank, base.in_features, **factory))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, config.rank, **factory))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

        self.kan = KANLayer(
            config.rank,
            config.rank,
            grid_size=config.grid_size,
            spline_order=config.spline_order,
            **factory,
        )
        if not config.learn_input_scale:
            self.kan.input_scale.requires_grad_(False)

        self._fraction_inside_grid: float | None = None

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        projected = torch.nn.functional.linear(x, self.lora_a)
        with torch.no_grad():
            self._fraction_inside_grid = self.kan.fraction_inside_grid(projected).item()
        return self.scaling * torch.nn.functional.linear(self.kan(projected), self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.delta(x)

    def last_fraction_inside_grid(self) -> float | None:
        """Доля входов сплайна, попавших в сетку на последнем проходе.

        Если она мала, базис обнуляется и адаптер вырождается в LoRA с
        неработающими параметрами — самая частая скрытая ошибка в реализациях
        KAN-адаптеров. Поэтому величина снимается на каждом проходе.
        """
        return self._fraction_inside_grid

    def analytic_parameter_count(self) -> int:
        rank = self.config.rank
        basis = self.config.grid_size + self.config.spline_order
        kan_parameters = rank * rank * (basis + 2) + (
            rank if self.config.learn_input_scale else 0
        )
        return rank * (self.base.in_features + self.base.out_features) + kan_parameters

    @property
    def can_merge(self) -> bool:
        return False
```

- [ ] **Шаг 4: Запустить тесты, убедиться, что они проходят**

Run: `python -m pytest -q`
Expected: PASS.

Если `test_parameter_count_matches_formula` упал, а `test_frozen_input_scale_is_excluded_from_training` — нет, значит `analytic_parameter_count` считает `input_scale` не так, как `trainable_parameter_count`. Это ровно та ошибка, ради которой тест написан: она не роняет обучение, но делает неверной колонку «число параметров» во всех таблицах.

- [ ] **Шаг 5: Коммит**

```bash
git add src/kanlora/adapters/kan_lora.py tests/adapters/test_kan_lora.py
git commit -m "feat: add KAN-LoRA adapter"
```

---

### Задача 6: Внедрение адаптеров в модель

Один внедритель на все три метода. Если бы каждый метод внедрялся своим кодом, различия в наборе затронутых слоёв просочились бы в результаты незаметно.

**Files:**
- Create: `src/kanlora/adapters/inject.py`
- Create: `tests/conftest.py`
- Create: `tests/adapters/test_inject.py`

**Interfaces:**
- Consumes: `AdapterConfig`, `AdapterLinear`, `freeze` из `kanlora.adapters.base`; `LoRALinear`, `DoRALinear`, `KANLoRALinear`.
- Produces:
  - `ADAPTER_TYPES: dict[str, type[AdapterLinear]]` с ключами `"lora"`, `"dora"`, `"kan_lora"`.
  - `DEFAULT_TARGET_MODULES: tuple[str, ...]` — семь проекций.
  - `InjectionReport(method: str, replaced: tuple[str, ...], trainable_parameters: int, total_parameters: int)`.
  - `inject_adapters(model: nn.Module, method: str, config: AdapterConfig, target_modules: Sequence[str] = DEFAULT_TARGET_MODULES) -> InjectionReport`.
  - `adapter_modules(model: nn.Module) -> list[tuple[str, AdapterLinear]]`.
  - `tests/conftest.py` даёт фикстуру `tiny_causal_lm` — настоящая архитектура Qwen2 в миниатюре, с теми же именами проекций.

- [ ] **Шаг 1: Написать общую фикстуру крошечной модели**

Она нужна не только здесь, но и в задачах 11, 12, 17. Модель настоящая — той же архитектуры Qwen2, только крошечная, поэтому тесты гоняют ровно тот код, который пойдёт в эксперименты.

```python
# tests/conftest.py
"""Общие фикстуры.

Крошечная модель той же архитектуры, что и рабочая: имена проекций совпадают
с настоящими, поэтому внедритель, цикл обучения и генерация проверяются на том
же коде, который пойдёт в эксперименты, но за секунды и на процессоре.
"""

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

VOCAB_SIZE = 64


@pytest.fixture
def tiny_causal_lm() -> Qwen2ForCausalLM:
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    model = Qwen2ForCausalLM(config)
    return model.to(torch.float32).eval()
```

- [ ] **Шаг 2: Написать падающий тест**

```python
# tests/adapters/test_inject.py
"""Проверки внедрения адаптеров.

Внедритель один на три метода. Тесты закрепляют то, от чего зависит честность
сравнения: одинаковый набор затронутых слоёв, замороженная основа и нулевая
поправка сразу после внедрения — модель обязана выдавать ровно то же, что и до.
"""

import pytest
import torch
from torch import nn

from kanlora.adapters.base import AdapterConfig, AdapterLinear
from kanlora.adapters.inject import (
    DEFAULT_TARGET_MODULES,
    adapter_modules,
    inject_adapters,
)

METHODS = ["lora", "dora", "kan_lora"]
LAYERS = 2


@pytest.mark.parametrize("method", METHODS)
def test_replaces_every_target_projection(tiny_causal_lm, method: str) -> None:
    report = inject_adapters(tiny_causal_lm, method, AdapterConfig(rank=4))

    assert len(report.replaced) == LAYERS * len(DEFAULT_TARGET_MODULES)
    for name in report.replaced:
        assert name.split(".")[-1] in DEFAULT_TARGET_MODULES


@pytest.mark.parametrize("method", METHODS)
def test_all_methods_touch_the_same_slots(tiny_causal_lm, method: str) -> None:
    """Разные методы обязаны править один и тот же список слоёв."""
    report = inject_adapters(tiny_causal_lm, method, AdapterConfig(rank=4))
    assert set(report.replaced) == {
        f"model.layers.{index}.{block}.{projection}"
        for index in range(LAYERS)
        for block, projection in (
            ("self_attn", "q_proj"), ("self_attn", "k_proj"),
            ("self_attn", "v_proj"), ("self_attn", "o_proj"),
            ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj"),
        )
    }


@pytest.mark.parametrize("method", METHODS)
def test_output_is_unchanged_right_after_injection(tiny_causal_lm, method: str) -> None:
    """Нулевая поправка на уровне всей модели, а не отдельного слоя."""
    tokens = torch.randint(0, 64, (2, 9))
    with torch.no_grad():
        before = tiny_causal_lm(tokens).logits.clone()
        inject_adapters(tiny_causal_lm, method, AdapterConfig(rank=4))
        after = tiny_causal_lm(tokens).logits

    assert torch.allclose(before, after, atol=1e-5)


@pytest.mark.parametrize("method", METHODS)
def test_only_adapter_parameters_are_trainable(tiny_causal_lm, method: str) -> None:
    report = inject_adapters(tiny_causal_lm, method, AdapterConfig(rank=4))

    trainable = {
        name for name, parameter in tiny_causal_lm.named_parameters() if parameter.requires_grad
    }
    assert trainable, "адаптеры обязаны быть обучаемыми"
    for name in trainable:
        assert any(name.startswith(f"{slot}.") for slot in report.replaced), name
        assert ".base." not in name, name


@pytest.mark.parametrize("method", METHODS)
def test_report_counts_match_the_model(tiny_causal_lm, method: str) -> None:
    report = inject_adapters(tiny_causal_lm, method, AdapterConfig(rank=4))

    assert report.trainable_parameters == sum(
        p.numel() for p in tiny_causal_lm.parameters() if p.requires_grad
    )
    assert report.total_parameters == sum(p.numel() for p in tiny_causal_lm.parameters())
    assert report.method == method


@pytest.mark.parametrize("method", METHODS)
def test_adapter_modules_enumerates_what_was_injected(tiny_causal_lm, method: str) -> None:
    report = inject_adapters(tiny_causal_lm, method, AdapterConfig(rank=4))
    found = adapter_modules(tiny_causal_lm)

    assert [name for name, _ in found] == sorted(report.replaced)
    assert all(isinstance(module, AdapterLinear) for _, module in found)


def test_kan_lora_costs_a_couple_of_percent_more_parameters(tiny_causal_lm) -> None:
    """Число, которое идёт в таблицу вместо отдельных прогонов LoRA с большим рангом."""
    import copy

    reference = copy.deepcopy(tiny_causal_lm)
    lora = inject_adapters(reference, "lora", AdapterConfig(rank=4))
    kan = inject_adapters(tiny_causal_lm, "kan_lora", AdapterConfig(rank=4))

    assert kan.trainable_parameters > lora.trainable_parameters


def test_unknown_method_is_rejected(tiny_causal_lm) -> None:
    with pytest.raises(KeyError):
        inject_adapters(tiny_causal_lm, "qlora", AdapterConfig())


def test_missing_target_module_is_rejected(tiny_causal_lm) -> None:
    """Опечатка в имени проекции обязана падать, а не тихо внедрять меньше слоёв."""
    with pytest.raises(ValueError, match="не найден"):
        inject_adapters(tiny_causal_lm, "lora", AdapterConfig(), target_modules=("qkv_proj",))
```

- [ ] **Шаг 3: Запустить тест, убедиться, что он падает**

Run: `python -m pytest tests/adapters/test_inject.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kanlora.adapters.inject'`.

- [ ] **Шаг 4: Написать минимальную реализацию**

```python
# src/kanlora/adapters/inject.py
"""Замена nn.Linear на адаптер. Единая для всех трёх методов.

Один внедритель на LoRA, DoRA и KAN-LoRA существует ради сопоставимости:
если бы каждый метод правил модель своим кодом, различие в наборе затронутых
слоёв просочилось бы в результаты, не выдав ошибки.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from torch import nn

from kanlora.adapters.base import AdapterConfig, AdapterLinear, freeze
from kanlora.adapters.dora import DoRALinear
from kanlora.adapters.kan_lora import KANLoRALinear
from kanlora.adapters.lora import LoRALinear

__all__ = [
    "ADAPTER_TYPES",
    "DEFAULT_TARGET_MODULES",
    "InjectionReport",
    "adapter_modules",
    "inject_adapters",
]

ADAPTER_TYPES: dict[str, type[AdapterLinear]] = {
    "lora": LoRALinear,
    "dora": DoRALinear,
    "kan_lora": KANLoRALinear,
}

# Все семь проекций блока Qwen2: внимание целиком и все три матрицы MLP.
DEFAULT_TARGET_MODULES: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)


@dataclass(frozen=True)
class InjectionReport:
    method: str
    replaced: tuple[str, ...]
    trainable_parameters: int
    total_parameters: int


def inject_adapters(
    model: nn.Module,
    method: str,
    config: AdapterConfig,
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
) -> InjectionReport:
    """Замораживает модель и подменяет целевые nn.Linear адаптерами. Правит на месте."""
    adapter_type = ADAPTER_TYPES[method]
    freeze(model)

    wanted = set(target_modules)
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.split(".")[-1] in wanted
    ]
    found = {name.split(".")[-1] for name, _ in targets}
    missing = wanted - found
    if missing:
        raise ValueError(f"целевой слой не найден в модели: {', '.join(sorted(missing))}")

    for name, module in targets:
        parent_name, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attribute, adapter_type(module, config))

    return InjectionReport(
        method=method,
        replaced=tuple(sorted(name for name, _ in targets)),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_parameters=sum(p.numel() for p in model.parameters()),
    )


def adapter_modules(model: nn.Module) -> list[tuple[str, AdapterLinear]]:
    """Все внедрённые адаптеры в устойчивом порядке.

    Порядок фиксирован, потому что по нему собирается диагностика сплайнов:
    строки в таблицах анализа обязаны совпадать между прогонами.
    """
    return sorted(
        (
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, AdapterLinear)
        ),
        key=lambda item: item[0],
    )
```

- [ ] **Шаг 5: Запустить тесты, убедиться, что они проходят**

Run: `python -m pytest -q`
Expected: PASS.

- [ ] **Шаг 6: Коммит**

```bash
git add src/kanlora/adapters/inject.py src/kanlora/adapters/__init__.py tests/conftest.py tests/adapters/test_inject.py
git commit -m "feat: add single adapter injector for all three methods"
```

---

### Задача 7: Сериализация схемы базы данных

Схема попадает в промпт текстом, и этот текст обязан быть одинаковым во всех прогонах. Изменение формата задним числом обесценивает уже полученные результаты, не выдав ошибки, — поэтому формат фиксируется тестом.

**Files:**
- Create: `src/kanlora/data/__init__.py`
- Create: `src/kanlora/data/schema.py`
- Create: `tests/data/test_schema.py`
- Create: `tests/data/fixtures/tables_mini.json`

**Interfaces:**
- Consumes: ничего.
- Produces:
  - `Table(name: str, columns: tuple[str, ...])` — замороженный датакласс.
  - `DatabaseSchema(db_id: str, tables: tuple[Table, ...])` — замороженный датакласс.
  - `load_schemas(tables_json: Path) -> dict[str, DatabaseSchema]`.
  - `serialize_schema(schema: DatabaseSchema) -> str`.

- [ ] **Шаг 1: Создать образец файла схем**

Формат повторяет `tables.json` из Spider. Служебный столбец `*` (индекс таблицы −1) обязан отбрасываться.

```json
[
  {
    "db_id": "concert_singer",
    "table_names_original": ["stadium", "singer"],
    "column_names_original": [
      [-1, "*"],
      [0, "Stadium_ID"],
      [0, "Location"],
      [0, "Name"],
      [1, "Singer_ID"],
      [1, "Name"]
    ],
    "column_types": ["text", "number", "text", "text", "number", "text"]
  },
  {
    "db_id": "pets_1",
    "table_names_original": ["Student"],
    "column_names_original": [[-1, "*"], [0, "StuID"], [0, "LName"]],
    "column_types": ["text", "number", "text"]
  }
]
```

- [ ] **Шаг 2: Написать падающий тест**

```python
# tests/data/test_schema.py
"""Проверки сериализации схемы.

Текст схемы попадает в промпт и обязан быть одинаковым во всех прогонах всех
трёх методов. Изменение формата задним числом молча обесценило бы уже
полученные результаты, поэтому формат закреплён строкой в тесте.
"""

from pathlib import Path

import pytest

from kanlora.data.schema import DatabaseSchema, Table, load_schemas, serialize_schema

FIXTURE = Path(__file__).parent / "fixtures" / "tables_mini.json"


@pytest.fixture
def schemas() -> dict[str, DatabaseSchema]:
    return load_schemas(FIXTURE)


def test_loads_every_database(schemas: dict[str, DatabaseSchema]) -> None:
    assert set(schemas) == {"concert_singer", "pets_1"}


def test_columns_are_grouped_by_table(schemas: dict[str, DatabaseSchema]) -> None:
    tables = schemas["concert_singer"].tables
    assert tables == (
        Table("stadium", ("Stadium_ID", "Location", "Name")),
        Table("singer", ("Singer_ID", "Name")),
    )


def test_service_column_is_dropped(schemas: dict[str, DatabaseSchema]) -> None:
    """Столбец * относится ко всей базе, а не к таблице, и в промпт не идёт."""
    for schema in schemas.values():
        for table in schema.tables:
            assert "*" not in table.columns


def test_serialized_format_is_frozen(schemas: dict[str, DatabaseSchema]) -> None:
    assert serialize_schema(schemas["concert_singer"]) == (
        "stadium: Stadium_ID, Location, Name | singer: Singer_ID, Name"
    )


def test_serialization_is_deterministic(schemas: dict[str, DatabaseSchema]) -> None:
    """Порядок берётся из файла и не зависит от порядка обхода словарей."""
    first = serialize_schema(schemas["concert_singer"])
    reloaded = load_schemas(FIXTURE)["concert_singer"]
    assert serialize_schema(reloaded) == first


def test_schema_is_hashable(schemas: dict[str, DatabaseSchema]) -> None:
    """Замороженность — техническая защита от правки схемы в середине прогона."""
    assert len({schemas["pets_1"], schemas["pets_1"]}) == 1
```

- [ ] **Шаг 3: Запустить тест, убедиться, что он падает**

Run: `python -m pytest tests/data/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kanlora.data'`.

- [ ] **Шаг 4: Написать минимальную реализацию**

```python
# src/kanlora/data/schema.py
"""Схема базы данных и её текстовое представление для промпта.

Формат зафиксирован и одинаков для Spider и PAUQ, для всех трёх методов и для
всех зёрен. Столбец `*` относится ко всей базе, а не к таблице, и отбрасывается.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = ["DatabaseSchema", "Table", "load_schemas", "serialize_schema"]


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[str, ...]


@dataclass(frozen=True)
class DatabaseSchema:
    db_id: str
    tables: tuple[Table, ...]


def load_schemas(tables_json: Path) -> dict[str, DatabaseSchema]:
    """Читает tables.json формата Spider."""
    entries = json.loads(Path(tables_json).read_text(encoding="utf-8"))

    schemas: dict[str, DatabaseSchema] = {}
    for entry in entries:
        names = entry["table_names_original"]
        columns: list[list[str]] = [[] for _ in names]
        for table_index, column_name in entry["column_names_original"]:
            if table_index >= 0:
                columns[table_index].append(column_name)

        schemas[entry["db_id"]] = DatabaseSchema(
            db_id=entry["db_id"],
            tables=tuple(
                Table(name, tuple(table_columns))
                for name, table_columns in zip(names, columns, strict=True)
            ),
        )
    return schemas


def serialize_schema(schema: DatabaseSchema) -> str:
    """Одна строка вида `table: col, col | table: col`."""
    return " | ".join(
        f"{table.name}: {', '.join(table.columns)}" for table in schema.tables
    )
```

Создать пустой `src/kanlora/data/__init__.py`.

- [ ] **Шаг 5: Запустить тесты, убедиться, что они проходят**

Run: `python -m pytest -q`
Expected: PASS.

- [ ] **Шаг 6: Коммит**

```bash
git add src/kanlora/data tests/data
git commit -m "feat: add deterministic database schema serialization"
```

---

### Задача 8: Загрузка Spider и PAUQ

Оба набора хранятся в одном формате JSON и различаются только именами файлов, поэтому читатель один. Второй читатель означал бы второе место, где предобработка может незаметно разойтись между наборами, — а именно это расхождение и обесценило бы сравнение Spider с PAUQ.

**Files:**
- Create: `src/kanlora/data/loaders.py`
- Create: `src/kanlora/data/spider.py`
- Create: `src/kanlora/data/pauq.py`
- Create: `tests/data/test_loaders.py`
- Create: `tests/data/fixtures/spider_mini/train_spider.json`, `.../dev.json`, `.../tables.json`

**Interfaces:**
- Consumes: `DatabaseSchema`, `load_schemas` из `kanlora.data.schema`.
- Produces:
  - `Text2SqlExample(db_id: str, question: str, query: str)` — замороженный датакласс.
  - `DatasetLayout(train_file: str, eval_file: str, tables_file: str, database_dir: str)` — замороженный датакласс.
  - `SPIDER_LAYOUT: DatasetLayout`, `PAUQ_LAYOUT: DatasetLayout`.
  - `LAYOUTS: dict[str, DatasetLayout]` с ключами `"spider"`, `"pauq"`.
  - `load_examples(root: Path, layout: DatasetLayout, split: Literal["train", "eval"]) -> list[Text2SqlExample]`.
  - `load_dataset_schemas(root: Path, layout: DatasetLayout) -> dict[str, DatabaseSchema]`.
  - `database_path(root: Path, layout: DatasetLayout, db_id: str) -> Path`.
  - `subsample(examples: list[Text2SqlExample], size: int, seed: int) -> list[Text2SqlExample]`.

- [ ] **Шаг 1: Создать образцы файлов**

`tests/data/fixtures/spider_mini/tables.json` — скопировать содержимое `tests/data/fixtures/tables_mini.json` из задачи 7.

`tests/data/fixtures/spider_mini/train_spider.json`:

```json
[
  {"db_id": "concert_singer", "question": "How many singers are there?", "query": "SELECT count(*) FROM singer"},
  {"db_id": "concert_singer", "question": "List all stadium names.", "query": "SELECT Name FROM stadium"},
  {"db_id": "pets_1", "question": "How many students?", "query": "SELECT count(*) FROM Student"},
  {"db_id": "pets_1", "question": "Last names of students.", "query": "SELECT LName FROM Student"}
]
```

`tests/data/fixtures/spider_mini/dev.json`:

```json
[
  {"db_id": "concert_singer", "question": "What is the largest stadium?", "query": "SELECT Name FROM stadium ORDER BY Stadium_ID DESC LIMIT 1"}
]
```

- [ ] **Шаг 2: Написать падающий тест**

```python
# tests/data/test_loaders.py
"""Проверки загрузки наборов.

Spider и PAUQ читаются одним кодом. Здесь закрепляется, что этот код
преобразует примеры одинаково и что урезание выборки воспроизводимо: размер
обучающей выборки урезан до 2500, и все три метода обязаны видеть ровно одни
и те же примеры.
"""

from pathlib import Path

import pytest

from kanlora.data.loaders import (
    LAYOUTS,
    DatasetLayout,
    Text2SqlExample,
    database_path,
    load_dataset_schemas,
    load_examples,
    subsample,
)
from kanlora.data.pauq import PAUQ_LAYOUT
from kanlora.data.spider import SPIDER_LAYOUT

ROOT = Path(__file__).parent / "fixtures" / "spider_mini"


def test_train_split_is_read(tmp_path: Path) -> None:
    examples = load_examples(ROOT, SPIDER_LAYOUT, "train")
    assert len(examples) == 4
    assert examples[0] == Text2SqlExample(
        db_id="concert_singer",
        question="How many singers are there?",
        query="SELECT count(*) FROM singer",
    )


def test_eval_split_is_read() -> None:
    assert len(load_examples(ROOT, SPIDER_LAYOUT, "eval")) == 1


def test_unknown_split_is_rejected() -> None:
    with pytest.raises(KeyError):
        load_examples(ROOT, SPIDER_LAYOUT, "test")


def test_missing_file_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="train_spider.json"):
        load_examples(tmp_path, SPIDER_LAYOUT, "train")


def test_schemas_come_from_the_same_root() -> None:
    schemas = load_dataset_schemas(ROOT, SPIDER_LAYOUT)
    assert set(schemas) == {"concert_singer", "pets_1"}


def test_every_example_has_a_schema() -> None:
    """Пример без схемы дал бы пустой промпт и тихо испортил бы метрику."""
    schemas = load_dataset_schemas(ROOT, SPIDER_LAYOUT)
    for split in ("train", "eval"):
        for example in load_examples(ROOT, SPIDER_LAYOUT, split):
            assert example.db_id in schemas


def test_database_path_follows_the_layout() -> None:
    path = database_path(ROOT, SPIDER_LAYOUT, "pets_1")
    assert path == ROOT / "database" / "pets_1" / "pets_1.sqlite"


def test_layouts_are_registered_under_dataset_names() -> None:
    assert LAYOUTS["spider"] is SPIDER_LAYOUT
    assert LAYOUTS["pauq"] is PAUQ_LAYOUT


def test_layouts_differ_only_in_file_names() -> None:
    """Оба набора читаются одним кодом — различие сведено к именам файлов."""
    assert isinstance(PAUQ_LAYOUT, DatasetLayout)
    assert SPIDER_LAYOUT.database_dir == PAUQ_LAYOUT.database_dir == "database"
    assert SPIDER_LAYOUT.train_file != PAUQ_LAYOUT.train_file


def test_subsample_is_reproducible() -> None:
    examples = load_examples(ROOT, SPIDER_LAYOUT, "train")
    assert subsample(examples, 3, seed=0) == subsample(examples, 3, seed=0)


def test_subsample_does_not_depend_on_the_training_seed() -> None:
    """Все три метода и все зёрна обучения видят одну и ту же выборку.

    Иначе разница между методами частично объяснялась бы разной выборкой,
    и вывод о методе стал бы недоказуемым.
    """
    examples = load_examples(ROOT, SPIDER_LAYOUT, "train")
    assert subsample(examples, 3, seed=0) != subsample(examples, 3, seed=1)


def test_subsample_keeps_everything_when_size_exceeds_the_set() -> None:
    examples = load_examples(ROOT, SPIDER_LAYOUT, "train")
    assert subsample(examples, 100, seed=0) == examples
```

- [ ] **Шаг 3: Запустить тест, убедиться, что он падает**

Run: `python -m pytest tests/data/test_loaders.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kanlora.data.loaders'`.

- [ ] **Шаг 4: Написать минимальную реализацию**

```python
# src/kanlora/data/loaders.py
"""Чтение наборов Spider и PAUQ.

Оба набора хранятся в одном формате JSON и различаются только именами файлов,
поэтому читатель один. Второй читатель означал бы второе место, где
предобработка способна незаметно разойтись между наборами, а сравнение
Spider с PAUQ строится ровно на том, что предобработка у них одна.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from kanlora.data.schema import DatabaseSchema, load_schemas

__all__ = [
    "LAYOUTS",
    "DatasetLayout",
    "Text2SqlExample",
    "database_path",
    "load_dataset_schemas",
    "load_examples",
    "subsample",
]

Split = Literal["train", "eval"]


@dataclass(frozen=True)
class Text2SqlExample:
    db_id: str
    question: str
    query: str


@dataclass(frozen=True)
class DatasetLayout:
    train_file: str
    eval_file: str
    tables_file: str
    database_dir: str


def _split_file(layout: DatasetLayout, split: Split) -> str:
    return {"train": layout.train_file, "eval": layout.eval_file}[split]


def load_examples(root: Path, layout: DatasetLayout, split: Split) -> list[Text2SqlExample]:
    path = Path(root) / _split_file(layout, split)
    if not path.is_file():
        raise FileNotFoundError(f"файл разбиения не найден: {path}")

    entries = json.loads(path.read_text(encoding="utf-8"))
    return [
        Text2SqlExample(
            db_id=entry["db_id"],
            question=entry["question"].strip(),
            query=" ".join(entry["query"].split()),
        )
        for entry in entries
    ]


def load_dataset_schemas(root: Path, layout: DatasetLayout) -> dict[str, DatabaseSchema]:
    return load_schemas(Path(root) / layout.tables_file)


def database_path(root: Path, layout: DatasetLayout, db_id: str) -> Path:
    return Path(root) / layout.database_dir / db_id / f"{db_id}.sqlite"


def subsample(examples: list[Text2SqlExample], size: int, seed: int) -> list[Text2SqlExample]:
    """Урезание обучающей выборки с сохранением исходного порядка.

    Зерно здесь — зерно выборки, а не зерно обучения: выборка обязана быть
    одной и той же для всех трёх методов и всех трёх зёрен обучения, иначе
    разница между методами частично объяснялась бы разными данными.
    """
    if size >= len(examples):
        return list(examples)

    indices = sorted(random.Random(seed).sample(range(len(examples)), size))
    return [examples[index] for index in indices]


# Раскладки регистрируются здесь, а определяются в spider.py и pauq.py,
# чтобы имя набора из конфигурации отображалось в раскладку одним словарём.
from kanlora.data.pauq import PAUQ_LAYOUT  # noqa: E402
from kanlora.data.spider import SPIDER_LAYOUT  # noqa: E402

LAYOUTS: dict[str, DatasetLayout] = {"spider": SPIDER_LAYOUT, "pauq": PAUQ_LAYOUT}
```

```python
# src/kanlora/data/spider.py
"""Раскладка файлов Spider.

Ожидается распакованный официальный архив:
    data/spider/train_spider.json
    data/spider/dev.json
    data/spider/tables.json
    data/spider/database/<db_id>/<db_id>.sqlite
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SPIDER_LAYOUT"]


@dataclass(frozen=True)
class _Layout:
    train_file: str
    eval_file: str
    tables_file: str
    database_dir: str


SPIDER_LAYOUT = _Layout(
    train_file="train_spider.json",
    eval_file="dev.json",
    tables_file="tables.json",
    database_dir="database",
)
```

```python
# src/kanlora/data/pauq.py
"""Раскладка файлов PAUQ, разбиение pauq_xsp.

Берётся одно разбиение — по базам данных, как в Spider. Второе разбиение
не используется, чтобы не плодить вариантов.

Ожидается:
    data/pauq/pauq_xsp_train.json
    data/pauq/pauq_xsp_test.json
    data/pauq/tables.json
    data/pauq/database/<db_id>/<db_id>.sqlite

Если фактические имена файлов в выкачанном PAUQ отличаются, поправить их
здесь и синхронно в DATASET_FILES в scripts/check_environment.py.
"""

from __future__ import annotations

from kanlora.data.spider import _Layout

__all__ = ["PAUQ_LAYOUT"]

PAUQ_LAYOUT = _Layout(
    train_file="pauq_xsp_train.json",
    eval_file="pauq_xsp_test.json",
    tables_file="tables.json",
    database_dir="database",
)
```

**Замечание по устройству.** `DatasetLayout` объявлен в `loaders.py`, а `_Layout` — в `spider.py`; это один и тот же датакласс под двумя именами, что породило бы путаницу. Правильно: объявить датакласс один раз в `loaders.py`, а `spider.py` и `pauq.py` импортировать его оттуда. Но `loaders.py` импортирует раскладки из них — получается цикл. Разрубается так: **датакласс `DatasetLayout` объявляется в `spider.py`** (он же первый по смыслу), `loaders.py` импортирует его оттуда, `pauq.py` тоже. Реализуя шаг, сделать именно так: убрать `DatasetLayout` из `loaders.py`, оставить `from kanlora.data.spider import DatasetLayout, SPIDER_LAYOUT`, а в `spider.py` переименовать `_Layout` в `DatasetLayout`. Тест `test_layouts_differ_only_in_file_names` проверяет `isinstance(PAUQ_LAYOUT, DatasetLayout)` и упадёт, если этого не сделать.

- [ ] **Шаг 5: Запустить тесты, убедиться, что они проходят**

Run: `python -m pytest -q`
Expected: PASS.

- [ ] **Шаг 6: Коммит**

```bash
git add src/kanlora/data tests/data
git commit -m "feat: add Spider and PAUQ loaders with reproducible subsampling"
```

---

### Задача 9: Промпт, токенизация и маска функции потерь

Ошибка в маске — самая коварная в этой работе: обучение пойдёт, метрики посчитаются, но модель будет учиться воспроизводить схему базы вместо SQL. Маска покрывает только токены запроса.

**Files:**
- Create: `src/kanlora/data/collate.py`
- Create: `tests/data/test_collate.py`

**Interfaces:**
- Consumes: `Text2SqlExample` из `kanlora.data.loaders`; `DatabaseSchema`, `serialize_schema` из `kanlora.data.schema`.
- Produces:
  - `PROMPT_TEMPLATE: str`.
  - `IGNORE_INDEX: int` (равен −100).
  - `build_prompt(question: str, schema_text: str) -> str`.
  - `encode(tokenizer, prompt: str, target: str, max_length: int) -> dict[str, list[int]]` — ключи `input_ids`, `labels`.
  - `EncodedDataset` — `list[dict[str, list[int]]]`.
  - `build_dataset(examples, schemas, tokenizer, max_length) -> tuple[EncodedDataset, int]`, второй элемент — число усечённых примеров.
  - `build_prompts(examples, schemas) -> list[str]`.
  - `Collator(pad_token_id: int)` с `__call__(batch) -> dict[str, Tensor]` — ключи `input_ids`, `attention_mask`, `labels`.

- [ ] **Шаг 1: Написать падающий тест**

```python
# tests/data/test_collate.py
"""Проверки промпта, токенизации и маски функции потерь.

Ошибка в маске не роняет обучение: функция потерь считается, графики рисуются,
а модель учится воспроизводить схему базы вместо SQL. Поэтому маска
проверяется напрямую, а не через качество.
"""

import pytest
import torch

from kanlora.data.collate import (
    IGNORE_INDEX,
    Collator,
    build_dataset,
    build_prompt,
    build_prompts,
    encode,
)
from kanlora.data.loaders import Text2SqlExample
from kanlora.data.schema import DatabaseSchema, Table


class FakeTokenizer:
    """Посимвольный токенизатор: сравнивать удобно, поведение то же."""

    eos_token_id = 1
    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": [ord(character) for character in text]}


@pytest.fixture
def tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def schemas() -> dict[str, DatabaseSchema]:
    return {"db": DatabaseSchema("db", (Table("t", ("a", "b")),))}


@pytest.fixture
def examples() -> list[Text2SqlExample]:
    return [
        Text2SqlExample("db", "q1", "SELECT a FROM t"),
        Text2SqlExample("db", "q2", "SELECT b FROM t"),
    ]


def test_prompt_contains_schema_and_question() -> None:
    prompt = build_prompt("How many?", "t: a, b")
    assert "t: a, b" in prompt
    assert "How many?" in prompt
    assert prompt.endswith("SQL:\n")


def test_prompt_frame_is_identical_for_both_datasets() -> None:
    """Рамка одна, чтобы различие Spider и PAUQ сводилось к языку вопроса."""
    russian = build_prompt("Сколько певцов?", "t: a")
    english = build_prompt("How many singers?", "t: a")
    assert russian.replace("Сколько певцов?", "X") == english.replace("How many singers?", "X")


def test_labels_mask_the_prompt(tokenizer: FakeTokenizer) -> None:
    encoded = encode(tokenizer, "PROMPT", "SQL", max_length=64)
    prompt_length = len("PROMPT")

    assert encoded["labels"][:prompt_length] == [IGNORE_INDEX] * prompt_length
    assert all(label != IGNORE_INDEX for label in encoded["labels"][prompt_length:])


def test_labels_align_with_input_ids(tokenizer: FakeTokenizer) -> None:
    encoded = encode(tokenizer, "PROMPT", "SQL", max_length=64)
    assert len(encoded["input_ids"]) == len(encoded["labels"])

    target_part = encoded["input_ids"][len("PROMPT") :]
    assert encoded["labels"][len("PROMPT") :] == target_part


def test_sequence_ends_with_eos(tokenizer: FakeTokenizer) -> None:
    """Без конца последовательности генерация не остановится."""
    encoded = encode(tokenizer, "PROMPT", "SQL", max_length=64)
    assert encoded["input_ids"][-1] == tokenizer.eos_token_id
    assert encoded["labels"][-1] == tokenizer.eos_token_id


def test_truncation_cuts_the_prompt_and_keeps_the_target(tokenizer: FakeTokenizer) -> None:
    """Усечение режет голову промпта: запрос обязан уцелеть целиком."""
    encoded = encode(tokenizer, "P" * 100, "SELECT", max_length=20)

    assert len(encoded["input_ids"]) == 20
    target_ids = [ord(character) for character in "SELECT"] + [tokenizer.eos_token_id]
    assert encoded["input_ids"][-len(target_ids) :] == target_ids
    assert encoded["labels"][-len(target_ids) :] == target_ids


def test_target_longer_than_limit_raises(tokenizer: FakeTokenizer) -> None:
    """Молча выбросить часть запроса нельзя — это тихая порча обучающих данных."""
    with pytest.raises(ValueError, match="не помещается"):
        encode(tokenizer, "P", "S" * 100, max_length=10)


def test_build_dataset_reports_truncations(
    tokenizer: FakeTokenizer, examples, schemas
) -> None:
    dataset, truncated = build_dataset(examples, schemas, tokenizer, max_length=1024)
    assert len(dataset) == 2
    assert truncated == 0

    _, truncated_many = build_dataset(examples, schemas, tokenizer, max_length=30)
    assert truncated_many == 2


def test_build_prompts_matches_dataset_order(examples, schemas) -> None:
    prompts = build_prompts(examples, schemas)
    assert len(prompts) == 2
    assert "q1" in prompts[0] and "q2" in prompts[1]


def test_collator_pads_inputs_and_masks_padding(tokenizer: FakeTokenizer) -> None:
    batch = [
        {"input_ids": [5, 6, 7], "labels": [IGNORE_INDEX, 6, 7]},
        {"input_ids": [8], "labels": [8]},
    ]
    collated = Collator(pad_token_id=tokenizer.pad_token_id)(batch)

    assert collated["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
    assert collated["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert collated["labels"].tolist() == [[IGNORE_INDEX, 6, 7], [8, IGNORE_INDEX, IGNORE_INDEX]]
    assert collated["input_ids"].dtype == torch.long
```

- [ ] **Шаг 2: Запустить тест, убедиться, что он падает**

Run: `python -m pytest tests/data/test_collate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kanlora.data.collate'`.

- [ ] **Шаг 3: Написать минимальную реализацию**

```python
# src/kanlora/data/collate.py
"""Промпт, токенизация, маска функции потерь и сборка батча.

Функция потерь считается только по токенам SQL-запроса: токены промпта
получают IGNORE_INDEX. Без этого модель училась бы воспроизводить схему базы,
и ошибка не выдала бы себя ничем — обучение бы шло, метрики бы считались.

Рамка промпта одна и та же для Spider и для PAUQ, чтобы различие между
наборами сводилось к языку вопроса, а не к оформлению.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from kanlora.data.loaders import Text2SqlExample
from kanlora.data.schema import DatabaseSchema, serialize_schema

__all__ = [
    "IGNORE_INDEX",
    "PROMPT_TEMPLATE",
    "Collator",
    "build_dataset",
    "build_prompt",
    "build_prompts",
    "encode",
]

IGNORE_INDEX = -100

PROMPT_TEMPLATE = "Database schema:\n{schema}\n\nQuestion: {question}\n\nSQL:\n"


def build_prompt(question: str, schema_text: str) -> str:
    return PROMPT_TEMPLATE.format(schema=schema_text, question=question)


def encode(tokenizer, prompt: str, target: str, max_length: int) -> dict[str, list[int]]:
    """Токенизирует пару и маскирует токены промпта.

    При переполнении режется голова промпта: запрос обязан уцелеть целиком,
    иначе обучающая пара становится неверной, не вызвав ошибки.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"] + [
        tokenizer.eos_token_id
    ]

    if len(target_ids) > max_length:
        raise ValueError(
            f"SQL-запрос не помещается в {max_length} токенов "
            f"(требуется {len(target_ids)}); увеличьте max_length"
        )

    room = max_length - len(target_ids)
    prompt_ids = prompt_ids[-room:] if room > 0 else []

    return {
        "input_ids": prompt_ids + target_ids,
        "labels": [IGNORE_INDEX] * len(prompt_ids) + target_ids,
    }


def build_prompts(
    examples: Sequence[Text2SqlExample], schemas: dict[str, DatabaseSchema]
) -> list[str]:
    return [
        build_prompt(example.question, serialize_schema(schemas[example.db_id]))
        for example in examples
    ]


def build_dataset(
    examples: Sequence[Text2SqlExample],
    schemas: dict[str, DatabaseSchema],
    tokenizer,
    max_length: int,
) -> tuple[list[dict[str, list[int]]], int]:
    """Кодирует все примеры и сообщает, сколько промптов было усечено.

    Число усечений идёт в карточку результата: если оно велико, часть схем
    не доехала до модели, и это влияет на метрику сильнее, чем метод адаптации.
    """
    prompts = build_prompts(examples, schemas)

    dataset: list[dict[str, list[int]]] = []
    truncated = 0
    for prompt, example in zip(prompts, examples, strict=True):
        full = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        encoded = encode(tokenizer, prompt, example.query, max_length)
        if len(encoded["input_ids"]) - full < len(encoded["labels"]) - full:
            pass
        if full + 1 > max_length or len(encoded["input_ids"]) == max_length:
            truncated += 1
        dataset.append(encoded)

    return dataset, truncated


class Collator:
    """Дополняет батч до самой длинной последовательности справа."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: Sequence[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(item["input_ids"]) for item in batch)

        input_ids, attention_mask, labels = [], [], []
        for item in batch:
            padding = width - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append([1] * len(item["input_ids"]) + [0] * padding)
            labels.append(item["labels"] + [IGNORE_INDEX] * padding)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
```

**Замечание по реализации.** Подсчёт усечений выше написан запутанно. Сделать проще и честнее: считать усечённым пример, у которого `len(prompt_ids) + len(target_ids) > max_length`. То есть внутри `build_dataset` посчитать обе длины напрямую и сравнить с `max_length`, а мёртвый `if ... pass` убрать. Тест `test_build_dataset_reports_truncations` проверяет именно это поведение.

- [ ] **Шаг 4: Запустить тесты, убедиться, что они проходят**

Run: `python -m pytest -q`
Expected: PASS.

- [ ] **Шаг 5: Коммит**

```bash
git add src/kanlora/data/collate.py tests/data/test_collate.py
git commit -m "feat: add prompt building, tokenization and SQL-only loss mask"
```

---

### Задача 10: Оптимизаторы и разводка параметров

`torch.optim.Muon` принимает **только двумерные** параметры и отвергает всё прочее жёсткой ошибкой `ValueError: Muon only supports 2D parameters`. Отсюда получается градиент покрытия: LoRA покрыта Muon полностью, DoRA частично (вектор модуля одномерен), KAN-LoRA частично (коэффициенты сплайнов трёхмерны). Это самостоятельный результат работы, и здесь он получает код и тест.

**Files:**
- Create: `src/kanlora/train/__init__.py`
- Create: `src/kanlora/train/optimizers.py`
- Create: `tests/train/test_optimizers.py`

**Interfaces:**
- Consumes: ничего.
- Produces:
  - `OptimizerConfig(name: Literal["adamw", "muon"] = "adamw", learning_rate: float = 2.0e-4, weight_decay: float = 0.0, max_grad_norm: float = 1.0)` — замороженный датакласс.
  - `split_by_dimension(parameters: Iterable[nn.Parameter]) -> tuple[list, list]` — возвращает `(двумерные, остальные)`, только обучаемые.
  - `OptimizerBundle` с полями `optimizers: list[Optimizer]`, `parameters: list[nn.Parameter]` и методами `zero_grad()`, `step()`, `clip_grad_norm_(max_norm: float) -> Tensor`, `coverage() -> dict[str, int]`.
  - `build_optimizer(parameters: Iterable[nn.Parameter], config: OptimizerConfig) -> OptimizerBundle`.

- [ ] **Шаг 1: Написать падающий тест**

```python
# tests/train/test_optimizers.py
"""Проверки разводки параметров по оптимизаторам.

Muon принимает только двумерные параметры и отвергает остальные жёсткой
ошибкой. Отсюда разное покрытие у трёх методов — это результат работы, а не
техническая деталь, поэтому разводка проверяется поимённо: ни один параметр
не потерян и ни один не продублирован.
"""

import pytest
import torch
from torch import nn

from kanlora.adapters.base import AdapterConfig
from kanlora.adapters.dora import DoRALinear
from kanlora.adapters.kan_lora import KANLoRALinear
from kanlora.adapters.lora import LoRALinear
from kanlora.train.optimizers import OptimizerConfig, build_optimizer, split_by_dimension

IN_FEATURES, OUT_FEATURES, RANK = 12, 7, 4


def trainable(adapter) -> list[nn.Parameter]:
    return [p for p in adapter.parameters() if p.requires_grad]


@pytest.fixture(params=["lora", "dora", "kan_lora"])
def adapter(request):
    torch.manual_seed(0)
    base = nn.Linear(IN_FEATURES, OUT_FEATURES)
    config = AdapterConfig(rank=RANK)
    return {"lora": LoRALinear, "dora": DoRALinear, "kan_lora": KANLoRALinear}[
        request.param
    ](base, config)


def test_split_loses_and_duplicates_nothing(adapter) -> None:
    parameters = trainable(adapter)
    two_dimensional, other = split_by_dimension(parameters)

    identifiers = [id(p) for p in two_dimensional + other]
    assert sorted(identifiers) == sorted(id(p) for p in parameters)
    assert len(identifiers) == len(set(identifiers))


def test_split_puts_only_matrices_in_the_first_group(adapter) -> None:
    two_dimensional, other = split_by_dimension(trainable(adapter))
    assert all(p.dim() == 2 for p in two_dimensional)
    assert all(p.dim() != 2 for p in other)


def test_lora_is_fully_covered_by_muon() -> None:
    """Обе матрицы LoRA двумерны — Muon покрывает метод целиком."""
    adapter = LoRALinear(nn.Linear(IN_FEATURES, OUT_FEATURES), AdapterConfig(rank=RANK))
    bundle = build_optimizer(trainable(adapter), OptimizerConfig(name="muon"))

    coverage = bundle.coverage()
    assert coverage["muon"] == RANK * (IN_FEATURES + OUT_FEATURES)
    assert coverage["adamw"] == 0


def test_dora_magnitude_falls_back_to_adamw() -> None:
    """Вектор модуля одномерен, Muon его не принимает."""
    adapter = DoRALinear(nn.Linear(IN_FEATURES, OUT_FEATURES), AdapterConfig(rank=RANK))
    coverage = build_optimizer(trainable(adapter), OptimizerConfig(name="muon")).coverage()
    assert coverage["adamw"] == OUT_FEATURES


def test_kan_spline_coefficients_fall_back_to_adamw() -> None:
    """Коэффициенты сплайнов трёхмерны: индуктивное смещение Muon их не покрывает."""
    adapter = KANLoRALinear(nn.Linear(IN_FEATURES, OUT_FEATURES), AdapterConfig(rank=RANK))
    coverage = build_optimizer(trainable(adapter), OptimizerConfig(name="muon")).coverage()

    basis = adapter.config.grid_size + adapter.config.spline_order
    assert coverage["adamw"] == RANK * RANK * basis + RANK


def test_adamw_mode_uses_a_single_optimizer(adapter) -> None:
    bundle = build_optimizer(trainable(adapter), OptimizerConfig(name="adamw"))
    assert len(bundle.optimizers) == 1
    assert bundle.coverage()["muon"] == 0


def test_muon_mode_creates_two_optimizers_when_needed() -> None:
    adapter = KANLoRALinear(nn.Linear(IN_FEATURES, OUT_FEATURES), AdapterConfig(rank=RANK))
    bundle = build_optimizer(trainable(adapter), OptimizerConfig(name="muon"))
    assert len(bundle.optimizers) == 2


def test_step_changes_every_trainable_parameter(adapter) -> None:
    """Разводка обязана двигать всё: параметр без оптимизатора остался бы мёртвым."""
    parameters = trainable(adapter)
    bundle = build_optimizer(parameters, OptimizerConfig(name="muon", learning_rate=0.1))

    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    before = [p.detach().clone() for p in parameters]
    bundle.step()

    for index, (old, new) in enumerate(zip(before, parameters, strict=True)):
        assert not torch.equal(old, new), f"параметр {index} не сдвинулся"


def test_clipping_bounds_the_gradient_norm(adapter) -> None:
    parameters = trainable(adapter)
    bundle = build_optimizer(parameters, OptimizerConfig())
    for parameter in parameters:
        parameter.grad = torch.full_like(parameter, 100.0)

    bundle.clip_grad_norm_(1.0)
    total = torch.norm(torch.stack([p.grad.norm() for p in parameters]))
    assert total.item() == pytest.approx(1.0, abs=1e-4)


def test_zero_grad_clears_gradients(adapter) -> None:
    parameters = trainable(adapter)
    bundle = build_optimizer(parameters, OptimizerConfig())
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)

    bundle.zero_grad()
    assert all(p.grad is None for p in parameters)


def test_unknown_optimizer_is_rejected(adapter) -> None:
    with pytest.raises(KeyError):
        build_optimizer(trainable(adapter), OptimizerConfig(name="lion"))
```

- [ ] **Шаг 2: Запустить тест, убедиться, что он падает**

Run: `python -m pytest tests/train/test_optimizers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kanlora.train'`.

- [ ] **Шаг 3: Написать минимальную реализацию**

```python
# src/kanlora/train/optimizers.py
"""AdamW и Muon за единым интерфейсом.

Muon ортогонализует обновления и применим только к двумерным матрицам —
всё остальное он отвергает жёсткой ошибкой. Поэтому три сравниваемых метода
покрываются им по-разному: LoRA целиком (обе матрицы двумерны), DoRA частично
(вектор модуля одномерен), KAN-LoRA частично (коэффициенты сплайнов
трёхмерны). Непокрытые параметры идут в запасной AdamW, и прогон с Muon для
KAN-LoRA всегда гибридный. Это самостоятельный результат работы: индуктивное
смещение современного оптимизатора не распространяется на нелинейную часть
адаптера.

Оба оптимизатора видят одну скорость обучения — как и все три метода.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.optim import Optimizer

__all__ = ["OptimizerBundle", "OptimizerConfig", "build_optimizer", "split_by_dimension"]


@dataclass(frozen=True)
class OptimizerConfig:
    name: Literal["adamw", "muon"] = "adamw"
    learning_rate: float = 2.0e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0


def split_by_dimension(
    parameters: Iterable[nn.Parameter],
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Делит обучаемые параметры на двумерные и все прочие."""
    trainable = [p for p in parameters if p.requires_grad]
    return (
        [p for p in trainable if p.dim() == 2],
        [p for p in trainable if p.dim() != 2],
    )


class OptimizerBundle:
    """Один или два оптимизатора, ведущие себя как один."""

    def __init__(self, optimizers: list[Optimizer], groups: dict[str, list[nn.Parameter]]) -> None:
        self.optimizers = optimizers
        self._groups = groups
        self.parameters = [p for group in groups.values() for p in group]

    def zero_grad(self) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def clip_grad_norm_(self, max_norm: float) -> torch.Tensor:
        return torch.nn.utils.clip_grad_norm_(self.parameters, max_norm)

    def coverage(self) -> dict[str, int]:
        """Сколько параметров досталось каждому оптимизатору. Идёт в карточку результата."""
        return {name: sum(p.numel() for p in group) for name, group in self._groups.items()}


def build_optimizer(
    parameters: Iterable[nn.Parameter], config: OptimizerConfig
) -> OptimizerBundle:
    builders = {"adamw": _build_adamw, "muon": _build_muon}
    return builders[config.name](list(parameters), config)


def _build_adamw(parameters: list[nn.Parameter], config: OptimizerConfig) -> OptimizerBundle:
    trainable = [p for p in parameters if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    return OptimizerBundle([optimizer], {"adamw": trainable, "muon": []})


def _build_muon(parameters: list[nn.Parameter], config: OptimizerConfig) -> OptimizerBundle:
    matrices, rest = split_by_dimension(parameters)

    optimizers: list[Optimizer] = []
    if matrices:
        optimizers.append(
            torch.optim.Muon(
                matrices, lr=config.learning_rate, weight_decay=config.weight_decay
            )
        )
    if rest:
        optimizers.append(
            torch.optim.AdamW(rest, lr=config.learning_rate, weight_decay=config.weight_decay)
        )

    return OptimizerBundle(optimizers, {"muon": matrices, "adamw": rest})
```

Создать пустой `src/kanlora/train/__init__.py` и `tests/train/` (без `__init__.py`, как в остальных каталогах тестов).

- [ ] **Шаг 4: Запустить тесты, убедиться, что они проходят**

Run: `python -m pytest -q`
Expected: PASS.

Если `test_step_changes_every_trainable_parameter` упал для `kan_lora` — значит трёхмерные коэффициенты не попали ни в один оптимизатор. Это ровно та ошибка, которая не даёт исключения: обучение идёт, но нелинейность не обучается вообще, и вывод «сплайны остались линейными» окажется артефактом разводки, а не свойством метода.

- [ ] **Шаг 5: Коммит**

```bash
git add src/kanlora/train tests/train
git commit -m "feat: route 2D parameters to Muon and the rest to AdamW"
```

---

### Задача 11: Цикл обучения и замер видеопамяти

Сходимость юнит-тестом не проверяется. Вместо этого — **тест на переобучение 20 примеров**: функция потерь обязана упасть почти до нуля. Он ловит подавляющее большинство ошибок цикла обучения — неверную маску, оторванный граф градиентов, замороженные не те параметры, — и делает это на крошечной модели за секунды.

**Files:**
- Create: `src/kanlora/train/memory.py`
- Create: `src/kanlora/train/loop.py`
- Create: `tests/train/test_memory.py`
- Create: `tests/train/test_loop.py`
- Modify: `pyproject.toml` (маркер `slow`)

**Interfaces:**
- Consumes: `Collator`, `IGNORE_INDEX` из `kanlora.data.collate`; `OptimizerConfig`, `build_optimizer` из `kanlora.train.optimizers`; `adapter_modules` из `kanlora.adapters.inject`; `KANLoRALinear` из `kanlora.adapters.kan_lora`.
- Produces:
  - `PeakMemoryTracker(device: torch.device)` с методами `reset() -> None`, `peak_bytes() -> int`.
  - `TrainConfig(epochs: int = 2, batch_size: int = 1, gradient_accumulation: int = 8, seed: int = 0, gradient_checkpointing: bool = True, log_every: int = 20)` — замороженный датакласс. Нормы обрезания градиента в нём нет: она живёт только в `OptimizerConfig`, чтобы у одного значения не было двух источников.
  - `TrainReport(step_losses: list[float], epoch_losses: list[float], fraction_inside_grid: list[float], peak_memory_bytes: int, seconds: float, optimizer_coverage: dict[str, int])`.
  - `set_seed(seed: int) -> None`.
  - `train(model, dataset, collator, optimizer_config, train_config, device) -> TrainReport`.

- [ ] **Шаг 1: Написать падающий тест замера памяти**

```python
# tests/train/test_memory.py
"""Проверки замера пиковой видеопамяти.

Сравнение методов по памяти осмысленно только при одинаковых длине
последовательности, размере батча и точности — а сам замер обязан быть
одинаковым куском кода для всех трёх методов и не падать там, где CUDA нет.
"""

import torch

from kanlora.train.memory import PeakMemoryTracker


def test_reports_zero_on_cpu() -> None:
    tracker = PeakMemoryTracker(torch.device("cpu"))
    tracker.reset()
    assert tracker.peak_bytes() == 0


def test_reset_is_idempotent_on_cpu() -> None:
    tracker = PeakMemoryTracker(torch.device("cpu"))
    tracker.reset()
    tracker.reset()
    assert tracker.peak_bytes() == 0


@torch.inference_mode()
def test_tracks_allocation_growth_on_cuda() -> None:
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA недоступна")

    device = torch.device("cuda")
    tracker = PeakMemoryTracker(device)
    tracker.reset()
    baseline = tracker.peak_bytes()

    held = torch.empty(1024, 1024, dtype=torch.float32, device=device)
    assert tracker.peak_bytes() >= baseline + held.numel() * 4
```

- [ ] **Шаг 2: Запустить, убедиться в падении**

Run: `python -m pytest tests/train/test_memory.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kanlora.train.memory'`.

- [ ] **Шаг 3: Реализовать замер памяти**

```python
# src/kanlora/train/memory.py
"""Замер пиковой видеопамяти.

Величина сравнима между методами только при жёстко одинаковых длине
последовательности, размере батча и точности. Замер вынесен сюда одним куском
кода именно для того, чтобы «удобная» правка под один метод не разошлась
с остальными.
"""

from __future__ import annotations

import torch

__all__ = ["PeakMemoryTracker"]


class PeakMemoryTracker:
    def __init__(self, device: torch.device) -> None:
        self.device = torch.device(device)
        self.enabled = self.device.type == "cuda"

    def reset(self) -> None:
        if self.enabled:
            torch.cuda.reset_peak_memory_stats(self.device)

    def peak_bytes(self) -> int:
        """Ноль на процессоре: карточка результата тогда честно скажет, что замера не было."""
        if not self.enabled:
            return 0
        return int(torch.cuda.max_memory_allocated(self.device))
```

- [ ] **Шаг 4: Добавить маркер `slow` в `pyproject.toml`**

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "-q"
markers = [
    "slow: тесты, которые обучают крошечную модель (единицы десятков секунд)",
]
```

- [ ] **Шаг 5: Написать падающий тест цикла обучения**

```python
# tests/train/test_loop.py
"""Проверки цикла обучения.

Сходимость юнит-тестом не проверяется. Вместо неё — переобучение двадцати
примеров: функция потерь обязана упасть почти до нуля. Этот тест ловит
подавляющее большинство ошибок цикла — неверную маску, оторванный граф
градиентов, замороженные не те параметры, — ни одна из которых не даёт
исключения сама по себе.
"""

import pytest
import torch

from kanlora.adapters.base import AdapterConfig
from kanlora.adapters.inject import inject_adapters
from kanlora.data.collate import IGNORE_INDEX, Collator
from kanlora.train.loop import TrainConfig, set_seed, train
from kanlora.train.optimizers import OptimizerConfig

VOCAB_SIZE = 64
DEVICE = torch.device("cpu")


def twenty_examples(seed: int = 0) -> list[dict[str, list[int]]]:
    """Двадцать пар «промпт -> цель» с маской по цели, как в настоящих данных."""
    generator = torch.Generator().manual_seed(seed)
    dataset = []
    for _ in range(20):
        prompt = torch.randint(2, VOCAB_SIZE, (6,), generator=generator).tolist()
        target = torch.randint(2, VOCAB_SIZE, (4,), generator=generator).tolist()
        dataset.append(
            {
                "input_ids": prompt + target,
                "labels": [IGNORE_INDEX] * len(prompt) + target,
            }
        )
    return dataset


def test_set_seed_makes_initialization_reproducible() -> None:
    set_seed(3)
    first = torch.randn(5)
    set_seed(3)
    assert torch.equal(first, torch.randn(5))


@pytest.mark.slow
@pytest.mark.parametrize("method", ["lora", "dora", "kan_lora"])
def test_overfits_twenty_examples(tiny_causal_lm, method: str) -> None:
    """ГЛАВНЫЙ ТЕСТ ЦИКЛА. Все три метода обязаны выучить двадцать примеров наизусть."""
    inject_adapters(tiny_causal_lm, method, AdapterConfig(rank=4))

    report = train(
        model=tiny_causal_lm,
        dataset=twenty_examples(),
        collator=Collator(pad_token_id=0),
        optimizer_config=OptimizerConfig(name="adamw", learning_rate=0.01),
        train_config=TrainConfig(
            epochs=60, batch_size=4, gradient_accumulation=1,
            gradient_checkpointing=False, seed=0,
        ),
        device=DEVICE,
    )

    assert report.epoch_losses[-1] < 0.1, f"функция потерь застряла: {report.epoch_losses[-1]}"
    assert report.epoch_losses[-1] < report.epoch_losses[0]


@pytest.mark.slow
def test_only_adapter_parameters_change(tiny_causal_lm) -> None:
    """Сверка того, что обучаются ровно адаптеры: основа обязана остаться прежней."""
    inject_adapters(tiny_causal_lm, "lora", AdapterConfig(rank=4))
    frozen = {
        name: parameter.detach().clone()
        for name, parameter in tiny_causal_lm.named_parameters()
        if not parameter.requires_grad
    }

    train(
        model=tiny_causal_lm,
        dataset=twenty_examples(),
        collator=Collator(pad_token_id=0),
        optimizer_config=OptimizerConfig(learning_rate=0.01),
        train_config=TrainConfig(epochs=2, batch_size=4, gradient_accumulation=1,
                                 gradient_checkpointing=False),
        device=DEVICE,
    )

    for name, before in frozen.items():
        after = dict(tiny_causal_lm.named_parameters())[name]
        assert torch.equal(before, after), f"замороженный параметр изменился: {name}"


@pytest.mark.slow
def test_reports_fraction_inside_grid_per_epoch(tiny_causal_lm) -> None:
    """Доля активаций внутри сетки логируется каждую эпоху — это готовый график."""
    inject_adapters(tiny_causal_lm, "kan_lora", AdapterConfig(rank=4))

    report = train(
        model=tiny_causal_lm,
        dataset=twenty_examples(),
        collator=Collator(pad_token_id=0),
        optimizer_config=OptimizerConfig(learning_rate=0.01),
        train_config=TrainConfig(epochs=3, batch_size=4, gradient_accumulation=1,
                                 gradient_checkpointing=False),
        device=DEVICE,
    )

    assert len(report.fraction_inside_grid) == 3
    assert all(0.0 <= value <= 1.0 for value in report.fraction_inside_grid)


@pytest.mark.slow
def test_linear_methods_report_no_grid_statistics(tiny_causal_lm) -> None:
    inject_adapters(tiny_causal_lm, "lora", AdapterConfig(rank=4))

    report = train(
        model=tiny_causal_lm,
        dataset=twenty_examples(),
        collator=Collator(pad_token_id=0),
        optimizer_config=OptimizerConfig(learning_rate=0.01),
        train_config=TrainConfig(epochs=2, batch_size=4, gradient_accumulation=1,
                                 gradient_checkpointing=False),
        device=DEVICE,
    )

    assert report.fraction_inside_grid == []


@pytest.mark.slow
def test_gradient_accumulation_matches_the_larger_batch(tiny_causal_lm) -> None:
    """Накопление градиента обязано быть эквивалентно большему батчу.

    Забытое деление на число накоплений завышает шаг ровно во столько же раз
    и не выдаёт себя ничем, кроме худшего качества.
    """
    import copy

    reference = copy.deepcopy(tiny_causal_lm)
    inject_adapters(tiny_causal_lm, "lora", AdapterConfig(rank=4))
    inject_adapters(reference, "lora", AdapterConfig(rank=4))
    reference.load_state_dict(tiny_causal_lm.state_dict())

    common = {
        "dataset": twenty_examples(),
        "collator": Collator(pad_token_id=0),
        "optimizer_config": OptimizerConfig(learning_rate=0.01),
        "device": DEVICE,
    }
    big = train(model=tiny_causal_lm, train_config=TrainConfig(
        epochs=1, batch_size=4, gradient_accumulation=1, gradient_checkpointing=False), **common)
    split = train(model=reference, train_config=TrainConfig(
        epochs=1, batch_size=2, gradient_accumulation=2, gradient_checkpointing=False), **common)

    assert big.epoch_losses[0] == pytest.approx(split.epoch_losses[0], rel=1e-3)


@pytest.mark.slow
def test_report_carries_timing_memory_and_coverage(tiny_causal_lm) -> None:
    inject_adapters(tiny_causal_lm, "lora", AdapterConfig(rank=4))

    report = train(
        model=tiny_causal_lm,
        dataset=twenty_examples(),
        collator=Collator(pad_token_id=0),
        optimizer_config=OptimizerConfig(name="muon", learning_rate=0.01),
        train_config=TrainConfig(epochs=1, batch_size=4, gradient_accumulation=1,
                                 gradient_checkpointing=False),
        device=DEVICE,
    )

    assert report.seconds > 0
    assert report.peak_memory_bytes == 0  # на процессоре замера нет, и это честно видно
    assert report.optimizer_coverage["muon"] > 0
    assert len(report.step_losses) == 5
```

- [ ] **Шаг 6: Запустить, убедиться в падении**

Run: `python -m pytest tests/train/test_loop.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kanlora.train.loop'`.

- [ ] **Шаг 7: Реализовать цикл обучения**

```python
# src/kanlora/train/loop.py
"""Цикл обучения, общий для всех трёх методов.

Один цикл на LoRA, DoRA и KAN-LoRA — по той же причине, что и один внедритель:
различие в коде обучения просочилось бы в сравнение методов, не выдав ошибки.

Накопление градиента делит функцию потерь на число накоплений, поэтому шаг
эквивалентен обучению большим батчем. Градиентный чекпоинтинг включён, чтобы
модель в fp32 помещалась в 11 ГБ.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader

from kanlora.adapters.inject import adapter_modules
from kanlora.adapters.kan_lora import KANLoRALinear
from kanlora.train.memory import PeakMemoryTracker
from kanlora.train.optimizers import OptimizerConfig, build_optimizer

__all__ = ["TrainConfig", "TrainReport", "set_seed", "train"]


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 2
    batch_size: int = 1
    gradient_accumulation: int = 8
    seed: int = 0
    gradient_checkpointing: bool = True
    log_every: int = 20
    # Нормы обрезания градиента здесь намеренно нет: она живёт в OptimizerConfig
    # и берётся циклом оттуда. Два источника одного значения разошлись бы молча,
    # изменив методологию для части прогонов и не выдав ошибки.


@dataclass
class TrainReport:
    step_losses: list[float] = field(default_factory=list)
    epoch_losses: list[float] = field(default_factory=list)
    fraction_inside_grid: list[float] = field(default_factory=list)
    peak_memory_bytes: int = 0
    seconds: float = 0.0
    optimizer_coverage: dict[str, int] = field(default_factory=dict)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _mean_fraction_inside_grid(model: torch.nn.Module) -> float | None:
    values = [
        module.last_fraction_inside_grid()
        for _, module in adapter_modules(model)
        if isinstance(module, KANLoRALinear)
    ]
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def train(
    model: torch.nn.Module,
    dataset: list[dict[str, list[int]]],
    collator,
    optimizer_config: OptimizerConfig,
    train_config: TrainConfig,
    device: torch.device,
) -> TrainReport:
    set_seed(train_config.seed)
    model.to(device)
    model.train()

    if train_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    loader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(train_config.seed),
    )
    bundle = build_optimizer(model.parameters(), optimizer_config)
    tracker = PeakMemoryTracker(device)
    tracker.reset()

    report = TrainReport(optimizer_coverage=bundle.coverage())
    started = time.perf_counter()

    for epoch in range(train_config.epochs):
        epoch_total, epoch_batches = 0.0, 0
        bundle.zero_grad()

        for index, batch in enumerate(loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            (loss / train_config.gradient_accumulation).backward()

            epoch_total += loss.item()
            epoch_batches += 1
            report.step_losses.append(loss.item())

            if index % train_config.gradient_accumulation == 0 or index == len(loader):
                bundle.clip_grad_norm_(optimizer_config.max_grad_norm)
                bundle.step()
                bundle.zero_grad()

        report.epoch_losses.append(epoch_total / max(epoch_batches, 1))
        fraction = _mean_fraction_inside_grid(model)
        if fraction is not None:
            report.fraction_inside_grid.append(fraction)

        print(
            f"эпоха {epoch + 1}/{train_config.epochs}: "
            f"функция потерь {report.epoch_losses[-1]:.4f}"
            + (f", доля активаций в сетке {fraction:.3f}" if fraction is not None else "")
        )

    report.seconds = time.perf_counter() - started
    report.peak_memory_bytes = tracker.peak_bytes()
    return report
```

- [ ] **Шаг 8: Запустить тесты, убедиться, что они проходят**

Run: `python -m pytest -q`
Expected: PASS.

Если `test_overfits_twenty_examples` не сходится для какого-то метода — **не увеличивать число эпох и не поднимать скорость обучения.** Тест поставлен ровно для того, чтобы поймать ошибку, и подстройка его параметров эту ошибку спрячет. Разбираться по порядку: маска (`labels` покрывает только цель), градиенты (все обучаемые параметры получили `grad`), разводка по оптимизаторам (Задача 10), тождественная инициализация адаптера (Задачи 3–5).

- [ ] **Шаг 9: Коммит**

```bash
git add pyproject.toml src/kanlora/train tests/train
git commit -m "feat: add training loop with peak memory tracking"
```

---

### Задача 12: Генерация SQL

Жадная генерация, одинаковая для всех прогонов. Отбор с температурой сделал бы метрику случайной величиной сверх той случайности, которую вносят зёрна.

**Files:**
- Create: `src/kanlora/eval/__init__.py`
- Create: `src/kanlora/eval/generate.py`
- Create: `tests/eval/test_generate.py`

**Interfaces:**
- Consumes: ничего из проекта.
- Produces:
  - `normalize_sql(text: str) -> str` — обрезает по первому переводу строки или `;`, схлопывает пробелы.
  - `generate_sql(model, tokenizer, prompts: list[str], *, device, batch_size: int = 8, max_new_tokens: int = 128) -> list[str]`.

- [ ] **Шаг 1: Написать падающий тест**

```python
# tests/eval/test_generate.py
"""Проверки генерации.

Генерация жадная во всех прогонах: отбор с температурой добавил бы к метрике
случайность сверх той, что уже вносят три зерна, и различия между методами
утонули бы окончательно.
"""

import pytest
import torch
from transformers import AutoTokenizer

from kanlora.eval.generate import generate_sql, normalize_sql


def test_cuts_at_the_first_newline() -> None:
    assert normalize_sql("SELECT a FROM t\nQuestion: next") == "SELECT a FROM t"


def test_cuts_at_the_semicolon() -> None:
    assert normalize_sql("SELECT a FROM t; SELECT b") == "SELECT a FROM t"


def test_collapses_whitespace() -> None:
    assert normalize_sql("  SELECT   a\tFROM  t  ") == "SELECT a FROM t"


def test_empty_generation_stays_empty() -> None:
    assert normalize_sql("   ") == ""


@pytest.mark.slow
def test_generates_one_string_per_prompt(tiny_causal_lm) -> None:
    tokenizer = _tiny_tokenizer()
    prompts = ["SELECT", "FROM", "WHERE"]

    generated = generate_sql(
        tiny_causal_lm, tokenizer, prompts,
        device=torch.device("cpu"), batch_size=2, max_new_tokens=4,
    )

    assert len(generated) == len(prompts)
    assert all(isinstance(item, str) for item in generated)


@pytest.mark.slow
def test_batching_does_not_change_the_result(tiny_causal_lm) -> None:
    """Дополнение слева не должно влиять на выход — иначе метрика зависела бы от размера батча."""
    tokenizer = _tiny_tokenizer()
    prompts = ["SELECT a", "FROM long table name here", "WHERE"]
    common = {"device": torch.device("cpu"), "max_new_tokens": 4}

    one_by_one = generate_sql(tiny_causal_lm, tokenizer, prompts, batch_size=1, **common)
    batched = generate_sql(tiny_causal_lm, tokenizer, prompts, batch_size=3, **common)
    assert one_by_one == batched


def _tiny_tokenizer():
    """Настоящий токенизатор Qwen2.5, урезанный до словаря крошечной модели."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
    tokenizer.padding_side = "left"
    return tokenizer
```

**Замечание.** `_tiny_tokenizer` тянет токенизатор из сети и не совместим со словарём в 64 токена из фикстуры. Реализуя тест, заменить его на локальный поддельный токенизатор: класс с методами `__call__(texts, return_tensors="pt", padding=True)`, `batch_decode`, полями `pad_token_id = 0`, `eos_token_id = 1`, `padding_side`, отображающий символы в коды по модулю 64 (как `FakeTokenizer` в `tests/data/test_collate.py`, но с дополнением слева и пакетным декодированием). Вынести его в `tests/conftest.py` как фикстуру `tiny_tokenizer`, чтобы им же пользовалась Задача 17. Тест `test_batching_does_not_change_the_result` тогда проверяет ровно то, ради чего написан, — независимость выхода от дополнения.

- [ ] **Шаг 2: Запустить, убедиться в падении**

Run: `python -m pytest tests/eval/test_generate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kanlora.eval'`.

- [ ] **Шаг 3: Написать минимальную реализацию**

```python
# src/kanlora/eval/generate.py
"""Пакетная жадная генерация SQL.

Жадная во всех прогонах без исключения: отбор с температурой сделал бы
метрику случайной величиной сверх той случайности, которую уже вносят три
зерна, и различия между методами перестали бы быть различимы.

Дополнение слева обязательно: при дополнении справа модель продолжала бы
последовательность от заполнителя, а не от промпта.
"""

from __future__ import annotations

import re

import torch

__all__ = ["generate_sql", "normalize_sql"]

_WHITESPACE = re.compile(r"\s+")


def normalize_sql(text: str) -> str:
    """Обрезает продолжение после запроса и приводит пробелы к одному виду.

    Модель охотно пишет дальше собственный следующий вопрос; всё после первого
    перевода строки или точки с запятой к ответу не относится.
    """
    head = text.split("\n", 1)[0].split(";", 1)[0]
    return _WHITESPACE.sub(" ", head).strip()


@torch.inference_mode()
def generate_sql(
    model,
    tokenizer,
    prompts: list[str],
    *,
    device: torch.device,
    batch_size: int = 8,
    max_new_tokens: int = 128,
) -> list[str]:
    model.to(device)
    model.eval()
    model.config.use_cache = True

    previous_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        generated: list[str] = []
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
            encoded = tokenizer(chunk, return_tensors="pt", padding=True).to(device)

            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            continuation = output[:, encoded["input_ids"].shape[1] :]
            generated.extend(
                normalize_sql(text)
                for text in tokenizer.batch_decode(continuation, skip_special_tokens=True)
            )
        return generated
    finally:
        tokenizer.padding_side = previous_side
```

- [ ] **Шаг 4: Запустить тесты, убедиться, что они проходят**

Run: `python -m pytest -q`
Expected: PASS.

- [ ] **Шаг 5: Коммит**

```bash
git add src/kanlora/eval tests/eval
git commit -m "feat: add batched greedy SQL generation"
```

---

### Задача 13: Exact Match по официальной методике Spider

В Spider Exact Match — это **покомпонентное сравнение разобранного запроса**, а не посимвольное равенство строк. Считать его самому нельзя: собственная реализация даст числа, несравнимые с литературой, и это типичный вопрос комиссии. Официальный сценарий вендорится и оборачивается.

**Files:**
- Create: `third_party/spider_eval/` (вендорится, не пишется)
- Create: `third_party/README.md`
- Create: `src/kanlora/eval/exact_match.py`
- Create: `tests/eval/test_exact_match.py`

**Interfaces:**
- Consumes: `tables.json` набора (путь передаётся снаружи).
- Produces:
  - `exact_match(gold: list[str], predicted: list[str], db_ids: list[str], tables_json: Path) -> float` — доля в `[0, 1]`.
  - `exact_match_by_hardness(gold, predicted, db_ids, tables_json) -> dict[str, float]` — разбивка по сложности запроса, как в официальном сценарии.

- [ ] **Шаг 1: Вендорить официальный сценарий**

Взять из репозитория `taoyds/spider` (каталог `evaluation/`) файлы `evaluation.py` и `process_sql.py`, положить в `third_party/spider_eval/`, добавить пустой `__init__.py`. Не править их содержимое: смысл вендоринга в том, что метрика считается ровно тем кодом, которым её считают в литературе.

`third_party/README.md`:

```markdown
# Сторонний код

`spider_eval/` — официальный сценарий оценки Spider (`evaluation.py`, `process_sql.py`)
из репозитория taoyds/spider, взят без изменений.

Причина вендоринга: Exact Match в Spider — покомпонентное сравнение разобранного
запроса, а не равенство строк. Собственная реализация дала бы числа, несравнимые
с опубликованными. Файлы не правятся; при обновлении заменять целиком и указывать
коммит-источник здесь.

Коммит-источник: <заполнить хешем при вендоринге>
```

- [ ] **Шаг 2: Написать падающий тест**

Пары взяты из официального набора Spider, поэтому эталон известен независимо от нашей реализации.

```python
# tests/eval/test_exact_match.py
"""Проверки Exact Match.

В Spider это покомпонентное сравнение разобранного запроса, а не равенство
строк: запросы, различающиеся регистром, лишними скобками или порядком
условий в WHERE, считаются совпавшими. Тесты закрепляют именно это поведение
на парах из официального набора.
"""

from pathlib import Path

import pytest

from kanlora.eval.exact_match import exact_match, exact_match_by_hardness

TABLES = Path(__file__).parents[1] / "data" / "fixtures" / "spider_mini" / "tables.json"
DB = "concert_singer"


def test_identical_queries_match() -> None:
    query = "SELECT count(*) FROM singer"
    assert exact_match([query], [query], [DB], TABLES) == pytest.approx(1.0)


def test_case_and_spacing_do_not_matter() -> None:
    """Разбор запроса, а не строки: регистр ключевых слов не влияет."""
    score = exact_match(
        ["SELECT count(*) FROM singer"],
        ["select  COUNT(*)   from   singer"],
        [DB],
        TABLES,
    )
    assert score == pytest.approx(1.0)


def test_different_queries_do_not_match() -> None:
    score = exact_match(
        ["SELECT count(*) FROM singer"], ["SELECT Name FROM singer"], [DB], TABLES
    )
    assert score == pytest.approx(0.0)


def test_unparsable_prediction_counts_as_a_miss() -> None:
    """Модель часто выдаёт мусор; это промах, а не исключение на весь прогон."""
    score = exact_match(["SELECT count(*) FROM singer"], ["не знаю"], [DB], TABLES)
    assert score == pytest.approx(0.0)


def test_score_is_the_share_of_matches() -> None:
    gold = ["SELECT count(*) FROM singer", "SELECT Name FROM stadium"]
    predicted = ["SELECT count(*) FROM singer", "SELECT Location FROM stadium"]
    assert exact_match(gold, predicted, [DB, DB], TABLES) == pytest.approx(0.5)


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="длины"):
        exact_match(["SELECT 1"], ["SELECT 1", "SELECT 2"], [DB], TABLES)


def test_hardness_breakdown_covers_all_levels() -> None:
    """Разбивка по сложности — та же, что в официальном сценарии."""
    breakdown = exact_match_by_hardness(
        ["SELECT count(*) FROM singer"], ["SELECT count(*) FROM singer"], [DB], TABLES
    )
    assert set(breakdown) == {"easy", "medium", "hard", "extra", "all"}
```

- [ ] **Шаг 3: Запустить, убедиться в падении**

Run: `python -m pytest tests/eval/test_exact_match.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kanlora.eval.exact_match'`.

- [ ] **Шаг 4: Написать обёртку**

```python
# src/kanlora/eval/exact_match.py
"""Обёртка над официальной метрикой Spider.

Exact Match в Spider — покомпонентное сравнение разобранного запроса:
запрос разбирается на части (SELECT, WHERE, GROUP BY и так далее), и части
сравниваются как множества. Поэтому регистр, лишние пробелы и порядок условий
в WHERE не влияют, а перестановка столбцов в SELECT — влияет.

Метрика считается вендоренным официальным кодом, а не своей реализацией:
собственная дала бы числа, несравнимые с опубликованными.
"""

from __future__ import annotations

import sys
from pathlib import Path

__all__ = ["exact_match", "exact_match_by_hardness"]

_THIRD_PARTY = Path(__file__).resolve().parents[3] / "third_party"
if str(_THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY))

HARDNESS_LEVELS = ("easy", "medium", "hard", "extra", "all")


def _build_evaluator(tables_json: Path):
    from spider_eval.evaluation import Evaluator, build_foreign_key_map_from_json

    return Evaluator(), build_foreign_key_map_from_json(str(tables_json))


def exact_match_by_hardness(
    gold: list[str],
    predicted: list[str],
    db_ids: list[str],
    tables_json: Path,
) -> dict[str, float]:
    """Доля точных совпадений, всего и в разбивке по сложности запроса."""
    if not (len(gold) == len(predicted) == len(db_ids)):
        raise ValueError(
            f"длины не совпадают: эталонов {len(gold)}, предсказаний {len(predicted)}, "
            f"баз {len(db_ids)}"
        )

    from spider_eval.evaluation import rebuild_sql_val, rebuild_sql_col  # noqa: F401
    from spider_eval.process_sql import Schema, get_schema_from_json, get_sql

    evaluator, foreign_keys = _build_evaluator(tables_json)
    schemas = get_schema_from_json(str(tables_json))

    scores = {level: [0, 0] for level in HARDNESS_LEVELS}  # [совпало, всего]

    for gold_query, predicted_query, db_id in zip(gold, predicted, db_ids, strict=True):
        schema = Schema(schemas[db_id])
        gold_sql = get_sql(schema, gold_query)
        hardness = evaluator.eval_hardness(gold_sql)

        try:
            predicted_sql = get_sql(schema, predicted_query)
            matched = bool(evaluator.eval_exact_match(predicted_sql, gold_sql))
        except Exception:
            # Модель регулярно выдаёт неразбираемый текст. Это промах,
            # а не повод уронить оценку всего прогона.
            matched = False

        for level in (hardness, "all"):
            scores[level][1] += 1
            scores[level][0] += int(matched)

    return {
        level: (matched / total if total else 0.0) for level, (matched, total) in scores.items()
    }


def exact_match(
    gold: list[str], predicted: list[str], db_ids: list[str], tables_json: Path
) -> float:
    return exact_match_by_hardness(gold, predicted, db_ids, tables_json)["all"]
```

**Замечание по реализации.** Точные имена функций вендоренного сценария (`build_foreign_key_map_from_json`, `get_schema_from_json`, сигнатура `Evaluator.eval_exact_match`) надо сверить с фактическим содержимым `third_party/spider_eval/evaluation.py` **сразу после вендоринга** и поправить обёртку под них. Официальный сценарий писался как программа командной строки, а не как библиотека, поэтому расхождения ожидаемы. Тесты выше проверяют поведение, а не имена, и останутся верными при любой правке импортов.

- [ ] **Шаг 5: Запустить тесты, убедиться, что они проходят**

Run: `python -m pytest tests/eval/test_exact_match.py -v && python -m pytest -q`
Expected: PASS.

- [ ] **Шаг 6: Коммит**

```bash
git add third_party src/kanlora/eval/exact_match.py tests/eval/test_exact_match.py
git commit -m "feat: wrap official Spider exact match evaluation"
```

---

### Задача 14: Execution Accuracy

Второй метрический столб: запрос исполняется на настоящей базе, и сравниваются **результаты**, а не тексты. Запрос, написанный иначе, но дающий тот же ответ, здесь засчитывается — в этом и разница с Exact Match.

**Files:**
- Create: `src/kanlora/eval/execution.py`
- Create: `tests/eval/test_execution.py`

**Interfaces:**
- Consumes: `database_path`, `DatasetLayout` из `kanlora.data.loaders`.
- Produces:
  - `ExecutionOutcome(rows: list[tuple] | None, failed: bool)`.
  - `execute_query(db_path: Path, query: str, timeout: float = 30.0) -> ExecutionOutcome`.
  - `results_match(gold: ExecutionOutcome, predicted: ExecutionOutcome, order_matters: bool) -> bool`.
  - `execution_accuracy(gold: list[str], predicted: list[str], db_ids: list[str], root: Path, layout: DatasetLayout) -> float`.

- [ ] **Шаг 1: Написать падающий тест**

```python
# tests/eval/test_execution.py
"""Проверки Execution Accuracy.

Сравниваются результаты исполнения, а не тексты запросов: запрос, написанный
иначе, но дающий тот же ответ, засчитывается. Именно этим метрика отличается
от Exact Match, и именно поэтому она нужна отдельно.
"""

import sqlite3
from pathlib import Path

import pytest

from kanlora.data.spider import SPIDER_LAYOUT
from kanlora.eval.execution import execute_query, execution_accuracy, results_match

DB = "shop"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / "database" / DB
    path.mkdir(parents=True)
    connection = sqlite3.connect(path / f"{DB}.sqlite")
    connection.executescript(
        """
        CREATE TABLE item (id INTEGER, name TEXT, price INTEGER);
        INSERT INTO item VALUES (1, 'apple', 30), (2, 'pear', 10), (3, 'plum', 20);
        """
    )
    connection.commit()
    connection.close()
    return tmp_path


def test_executes_and_returns_rows(root: Path) -> None:
    outcome = execute_query(root / "database" / DB / f"{DB}.sqlite", "SELECT count(*) FROM item")
    assert outcome.failed is False
    assert outcome.rows == [(3,)]


def test_broken_query_is_reported_not_raised(root: Path) -> None:
    outcome = execute_query(root / "database" / DB / f"{DB}.sqlite", "SELECT FROM nowhere")
    assert outcome.failed is True
    assert outcome.rows is None


def test_missing_database_is_reported_not_raised(tmp_path: Path) -> None:
    outcome = execute_query(tmp_path / "absent.sqlite", "SELECT 1")
    assert outcome.failed is True


def test_row_order_is_ignored_without_order_by(root: Path) -> None:
    """Без ORDER BY порядок строк в SQL не определён, и требовать его нельзя."""
    database = root / "database" / DB / f"{DB}.sqlite"
    ascending = execute_query(database, "SELECT name FROM item ORDER BY price")
    descending = execute_query(database, "SELECT name FROM item ORDER BY price DESC")

    assert results_match(ascending, descending, order_matters=False) is True
    assert results_match(ascending, descending, order_matters=True) is False


def test_failed_execution_never_matches(root: Path) -> None:
    database = root / "database" / DB / f"{DB}.sqlite"
    good = execute_query(database, "SELECT 1")
    bad = execute_query(database, "SELECT FROM")
    assert results_match(good, bad, order_matters=False) is False
    assert results_match(bad, bad, order_matters=False) is False


def test_accuracy_counts_equivalent_rewrites_as_correct(root: Path) -> None:
    """Главное свойство метрики: другой текст, тот же ответ — засчитано."""
    score = execution_accuracy(
        gold=["SELECT name FROM item WHERE price > 15"],
        predicted=["SELECT name FROM item WHERE NOT price <= 15"],
        db_ids=[DB],
        root=root,
        layout=SPIDER_LAYOUT,
    )
    assert score == pytest.approx(1.0)


def test_accuracy_respects_order_by(root: Path) -> None:
    score = execution_accuracy(
        gold=["SELECT name FROM item ORDER BY price"],
        predicted=["SELECT name FROM item ORDER BY price DESC"],
        db_ids=[DB],
        root=root,
        layout=SPIDER_LAYOUT,
    )
    assert score == pytest.approx(0.0)


def test_mismatched_lengths_are_rejected(root: Path) -> None:
    with pytest.raises(ValueError, match="длины"):
        execution_accuracy(["SELECT 1"], [], [DB], root, SPIDER_LAYOUT)
```

- [ ] **Шаг 2: Запустить, убедиться в падении**

Run: `python -m pytest tests/eval/test_execution.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kanlora.eval.execution'`.

- [ ] **Шаг 3: Написать минимальную реализацию**

```python
# src/kanlora/eval/execution.py
"""Исполнение SQL на sqlite и сравнение результатов.

Сравниваются результаты, а не тексты: запрос, написанный иначе, но дающий тот
же ответ, засчитывается. Порядок строк учитывается только при наличии ORDER BY
в эталонном запросе — без него порядок в SQL не определён, и требовать его
означало бы занижать метрику по случайному признаку.

Любая ошибка исполнения считается промахом, а не аварией: модель регулярно
выдаёт неисполнимый текст, и это нормальная часть измерения.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from kanlora.data.loaders import DatasetLayout, database_path

__all__ = ["ExecutionOutcome", "execute_query", "execution_accuracy", "results_match"]


@dataclass(frozen=True)
class ExecutionOutcome:
    rows: list[tuple] | None
    failed: bool


def execute_query(db_path: Path, query: str, timeout: float = 30.0) -> ExecutionOutcome:
    if not Path(db_path).is_file():
        return ExecutionOutcome(None, True)

    connection = None
    try:
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout)
        connection.text_factory = lambda value: value.decode("utf-8", errors="replace")
        return ExecutionOutcome(connection.execute(query).fetchall(), False)
    except Exception:
        return ExecutionOutcome(None, True)
    finally:
        if connection is not None:
            connection.close()


def results_match(
    gold: ExecutionOutcome, predicted: ExecutionOutcome, order_matters: bool
) -> bool:
    if gold.failed or predicted.failed:
        return False
    if order_matters:
        return gold.rows == predicted.rows
    return sorted(map(repr, gold.rows)) == sorted(map(repr, predicted.rows))


def execution_accuracy(
    gold: list[str],
    predicted: list[str],
    db_ids: list[str],
    root: Path,
    layout: DatasetLayout,
) -> float:
    if not (len(gold) == len(predicted) == len(db_ids)):
        raise ValueError(
            f"длины не совпадают: эталонов {len(gold)}, предсказаний {len(predicted)}, "
            f"баз {len(db_ids)}"
        )
    if not gold:
        return 0.0

    matched = 0
    for gold_query, predicted_query, db_id in zip(gold, predicted, db_ids, strict=True):
        database = database_path(root, layout, db_id)
        order_matters = "order by" in gold_query.lower()
        matched += int(
            results_match(
                execute_query(database, gold_query),
                execute_query(database, predicted_query),
                order_matters,
            )
        )
    return matched / len(gold)
```

- [ ] **Шаг 4: Запустить тесты, убедиться, что они проходят**

Run: `python -m pytest -q`
Expected: PASS.

- [ ] **Шаг 5: Коммит**

```bash
git add src/kanlora/eval/execution.py tests/eval/test_execution.py
git commit -m "feat: add execution accuracy over sqlite databases"
```

---

### Задача 15: Мера выученной нелинейности

Это исследовательская часть работы, а не техническая. Вопрос, ради которого всё делается: **используется ли нелинейность в адаптере на самом деле, и какой ценой?** Здесь появляется число, которым на него отвечают.

Измерение: берём выученную одномерную функцию, подгоняем к ней прямую методом наименьших квадратов на том отрезке, куда реально попадают активации, и смотрим долю, которую прямая не объясняет. Ноль по всем слоям — нелинейность не используется, KAN-LoRA сводится к LoRA с накладными расходами. Это доказанный отрицательный результат, а не «не получилось».

**Files:**
- Create: `src/kanlora/analysis/__init__.py`
- Create: `src/kanlora/analysis/spline_stats.py`
- Create: `tests/analysis/test_spline_stats.py`

**Interfaces:**
- Consumes: `KANLayer` из `kanlora.adapters.kan_layer`; `KANLoRALinear` из `kanlora.adapters.kan_lora`; `adapter_modules` из `kanlora.adapters.inject`.
- Produces:
  - `nonlinearity_index(inputs: Tensor, outputs: Tensor) -> float` — доля дисперсии, не объяснённая прямой; `0.0` для прямой, `1.0` для функции, ортогональной прямой.
  - `EdgeStats(source: int, target: int, nonlinearity: float)`.
  - `layer_nonlinearity(layer: KANLayer, lo: float, hi: float, samples: int = 257) -> list[EdgeStats]`.
  - `model_nonlinearity(model) -> dict[str, dict[str, float]]` — по каждому адаптеру `{"mean": ..., "max": ..., "fraction_inside_grid": ...}`.

- [ ] **Шаг 1: Написать падающий тест**

```python
# tests/analysis/test_spline_stats.py
"""Проверки меры выученной нелинейности.

Мера отвечает на главный вопрос работы: используется ли нелинейность на самом
деле. Ошибка здесь не роняет ничего — она просто выдаёт неверное число,
на котором строится вывод. Поэтому мера проверяется на функциях с заранее
известным ответом.
"""

import math

import pytest
import torch

from kanlora.adapters.base import AdapterConfig
from kanlora.adapters.inject import inject_adapters
from kanlora.adapters.kan_layer import KANLayer
from kanlora.analysis.spline_stats import (
    layer_nonlinearity,
    model_nonlinearity,
    nonlinearity_index,
)

GRID = torch.linspace(-1.0, 1.0, 401, dtype=torch.float64)


def test_straight_line_has_zero_nonlinearity() -> None:
    assert nonlinearity_index(GRID, 3.0 * GRID + 1.5) == pytest.approx(0.0, abs=1e-12)


def test_identity_has_zero_nonlinearity() -> None:
    assert nonlinearity_index(GRID, GRID) == pytest.approx(0.0, abs=1e-12)


def test_symmetric_parabola_is_fully_unexplained_by_a_line() -> None:
    """У x^2 на симметричном отрезке лучшая прямая — константа: прямая не объясняет ничего."""
    assert nonlinearity_index(GRID, GRID**2) == pytest.approx(1.0, abs=1e-6)


def test_line_plus_bump_is_between_zero_and_one() -> None:
    outputs = 2.0 * GRID + 0.1 * torch.sin(4.0 * math.pi * GRID)
    index = nonlinearity_index(GRID, outputs)
    assert 0.0 < index < 1.0


def test_constant_function_is_reported_as_linear() -> None:
    """Вырожденный случай: постоянная функция — частный случай прямой, а не деление на ноль."""
    assert nonlinearity_index(GRID, torch.zeros_like(GRID)) == pytest.approx(0.0)


def test_identity_layer_shows_no_nonlinearity() -> None:
    """Слой при инициализации тождественен, значит мера обязана дать ноль на всех рёбрах."""
    layer = KANLayer(3, 3, grid_size=5, spline_order=3, dtype=torch.float64)
    stats = layer_nonlinearity(layer, lo=-0.99, hi=0.99)

    assert len(stats) == 9
    assert max(item.nonlinearity for item in stats) == pytest.approx(0.0, abs=1e-8)


def test_perturbed_layer_shows_nonlinearity() -> None:
    layer = KANLayer(3, 3, grid_size=5, spline_order=3, dtype=torch.float64)
    torch.manual_seed(0)
    with torch.no_grad():
        layer.spline_coefficients.add_(0.5 * torch.randn_like(layer.spline_coefficients))

    stats = layer_nonlinearity(layer, lo=-0.99, hi=0.99)
    assert max(item.nonlinearity for item in stats) > 0.01


def test_edges_are_identified_by_index() -> None:
    layer = KANLayer(2, 3, grid_size=5, spline_order=3, dtype=torch.float64)
    stats = layer_nonlinearity(layer, lo=-0.9, hi=0.9)
    assert {(item.source, item.target) for item in stats} == {
        (source, target) for source in range(2) for target in range(3)
    }


def test_model_summary_covers_every_kan_adapter(tiny_causal_lm) -> None:
    inject_adapters(tiny_causal_lm, "kan_lora", AdapterConfig(rank=4))
    summary = model_nonlinearity(tiny_causal_lm)

    assert len(summary) == 14  # 2 слоя * 7 проекций
    for values in summary.values():
        assert set(values) == {"mean", "max", "fraction_inside_grid"}
        assert values["mean"] == pytest.approx(0.0, abs=1e-8)


def test_model_summary_is_empty_for_linear_methods(tiny_causal_lm) -> None:
    inject_adapters(tiny_causal_lm, "lora", AdapterConfig(rank=4))
    assert model_nonlinearity(tiny_causal_lm) == {}
```

- [ ] **Шаг 2: Запустить, убедиться в падении**

Run: `python -m pytest tests/analysis/test_spline_stats.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kanlora.analysis'`.

- [ ] **Шаг 3: Написать минимальную реализацию**

```python
# src/kanlora/analysis/spline_stats.py
"""Мера выученной нелинейности.

Отвечает на главный вопрос работы: используется ли нелинейность в адаптере
на самом деле. К каждой выученной одномерной функции подгоняется прямая
методом наименьших квадратов на том отрезке, куда реально попадают активации,
и берётся доля дисперсии, которую прямая не объясняет:

    index = RSS / TSS,   RSS — остаток после подгонки, TSS — полная дисперсия.

Ноль означает, что функция — прямая, единица — что прямая не объясняет
ничего. Если индекс близок к нулю по всем слоям, нелинейность не используется
и KAN-LoRA сводится к LoRA с накладными расходами. Это доказанный
отрицательный результат, а не неудача, и на защите он весит больше, чем
прибавка в доли процента к Exact Match.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from kanlora.adapters.inject import adapter_modules
from kanlora.adapters.kan_layer import KANLayer
from kanlora.adapters.kan_lora import KANLoRALinear

__all__ = ["EdgeStats", "layer_nonlinearity", "model_nonlinearity", "nonlinearity_index"]


@dataclass(frozen=True)
class EdgeStats:
    source: int
    target: int
    nonlinearity: float


def nonlinearity_index(inputs: torch.Tensor, outputs: torch.Tensor) -> float:
    """Доля дисперсии функции, не объяснённая наилучшей прямой."""
    x = inputs.to(torch.float64).flatten()
    y = outputs.to(torch.float64).flatten()

    total = ((y - y.mean()) ** 2).sum()
    if total.item() <= 1e-18:
        # Постоянная функция — частный случай прямой, а не деление на ноль.
        return 0.0

    design = torch.stack([x, torch.ones_like(x)], dim=1)
    coefficients = torch.linalg.lstsq(design, y.unsqueeze(1)).solution
    residual = ((y - (design @ coefficients).squeeze(1)) ** 2).sum()
    return float((residual / total).clamp(min=0.0, max=1.0))


def layer_nonlinearity(
    layer: KANLayer, lo: float, hi: float, samples: int = 257
) -> list[EdgeStats]:
    """Индекс нелинейности каждого ребра слоя на отрезке [lo, hi].

    Отрезок задаётся снаружи: мерить надо там, куда попадают активации,
    а не на всей области определения сетки.
    """
    device = layer.spline_coefficients.device
    grid = torch.linspace(lo, hi, samples, dtype=torch.float64, device=device)

    stats: list[EdgeStats] = []
    with torch.no_grad():
        for source in range(layer.in_features):
            probe = torch.zeros(samples, layer.in_features, dtype=layer.spline_coefficients.dtype,
                                device=device)
            probe[:, source] = grid.to(probe.dtype)
            response = layer(probe)
            for target in range(layer.out_features):
                stats.append(
                    EdgeStats(source, target, nonlinearity_index(grid, response[:, target]))
                )
    return stats


def model_nonlinearity(model: torch.nn.Module) -> dict[str, dict[str, float]]:
    """Сводка по всем KAN-адаптерам модели. Для LoRA и DoRA возвращает пустой словарь."""
    summary: dict[str, dict[str, float]] = {}

    for name, module in adapter_modules(model):
        if not isinstance(module, KANLoRALinear):
            continue

        lo, hi = module.kan.grid_range
        values = [item.nonlinearity for item in layer_nonlinearity(module.kan, 0.99 * lo, 0.99 * hi)]
        fraction = module.last_fraction_inside_grid()
        summary[name] = {
            "mean": sum(values) / len(values),
            "max": max(values),
            "fraction_inside_grid": float("nan") if fraction is None else fraction,
        }
    return summary
```

**Замечание про отрезок измерения.** `model_nonlinearity` мерит на всей сетке, а спецификация требует мерить там, куда реально попадают активации. Правильный вариант приходит в Задаче 16: перед вызовом `model_nonlinearity` прогнать через модель батч отложенного набора, попутно собрав фактический размах входов сплайна по каждому адаптеру, и передать его сюда. На этом шаге достаточно сетки; расширение сигнатуры до `model_nonlinearity(model, ranges: dict[str, tuple[float, float]] | None = None)` делается в Задаче 16 вместе с тестом на то, что переданный отрезок действительно используется.

- [ ] **Шаг 4: Запустить тесты, убедиться, что они проходят**

Run: `python -m pytest -q`
Expected: PASS.

- [ ] **Шаг 5: Коммит**

```bash
git add src/kanlora/analysis tests/analysis
git commit -m "feat: add learned nonlinearity index for KAN splines"
```

---

### Задача 16: Конфигурация, карточка результата и сквозной прогон

Здесь всё собирается в одну команду. Требование спецификации: **та же самая команда без изменений запускает и проверочный прогон на 200 примерах, и полный** — отдельного «настоящего» пути запуска не существует, иначе проверенным окажется не тот код, который считает результаты.

Карточка результата — единица поставки. В ней лежит всё, что нужно для восстановления прогона: полная конфигурация, версии библиотек, коммит, метрики, пик видеопамяти, время, покрытие оптимизаторами и статистика сплайнов.

**Files:**
- Create: `src/kanlora/config.py`
- Create: `src/kanlora/results.py`
- Create: `configs/base.yaml`
- Create: `configs/smoke.yaml`
- Create: `scripts/run_experiment.py`
- Create: `results/.gitkeep`
- Create: `tests/test_config.py`
- Create: `tests/test_results.py`
- Modify: `src/kanlora/analysis/spline_stats.py` (отрезок измерения снаружи)
- Modify: `tests/analysis/test_spline_stats.py` (тест на этот отрезок)

**Interfaces:**
- Consumes: всё построенное в задачах 2–15.
- Produces:
  - `ExperimentConfig` — замороженный датакласс с полями `run_id: str`, `method: str`, `dataset: str`, `seed: int`, `model_name: str`, `data_root: str`, `train_subset: int`, `subsample_seed: int`, `max_length: int`, `eval_limit: int | None`, `eval_batch_size: int`, `max_new_tokens: int`, `adapter: AdapterConfig`, `train: TrainConfig`, `optimizer: OptimizerConfig`.
  - `load_config(path: Path) -> ExperimentConfig`.
  - `apply_overrides(config: ExperimentConfig, **overrides) -> ExperimentConfig`.
  - `config_to_dict(config: ExperimentConfig) -> dict`.
  - `collect_versions() -> dict[str, str]`, `git_commit() -> str`.
  - `build_result_card(config, injection, train_report, metrics, spline_stats) -> dict`.
  - `write_result_card(card: dict, directory: Path) -> Path`.
  - `model_nonlinearity(model, ranges: dict[str, tuple[float, float]] | None = None)` — расширенная сигнатура.

- [ ] **Шаг 1: Написать падающий тест конфигурации**

```python
# tests/test_config.py
"""Проверки конфигурации прогона.

Конфигурация обязана быть неизменяемой и целиком попадать в карточку
результата: воспроизводимость держится на том, что по карточке можно
восстановить прогон, не заглядывая в историю команд.
"""

from pathlib import Path

import pytest
import yaml

from kanlora.config import ExperimentConfig, apply_overrides, config_to_dict, load_config

BASE = Path(__file__).parents[1] / "configs" / "base.yaml"


@pytest.fixture
def config() -> ExperimentConfig:
    return load_config(BASE)


def test_fixed_hyperparameters_match_the_plan(config: ExperimentConfig) -> None:
    """Гиперпараметры зафиксированы один раз; тест не даёт им уехать незаметно."""
    assert config.adapter.rank == 8
    assert config.adapter.alpha == 16.0
    assert config.adapter.grid_size == 5
    assert config.adapter.spline_order == 3
    assert config.max_length == 768
    assert config.train.epochs == 2
    assert config.train_subset == 2500
    assert config.optimizer.learning_rate == pytest.approx(2.0e-4)
    assert config.optimizer.max_grad_norm == 1.0


def test_configuration_is_immutable(config: ExperimentConfig) -> None:
    with pytest.raises(Exception):
        config.seed = 5


def test_overrides_produce_a_new_configuration(config: ExperimentConfig) -> None:
    changed = apply_overrides(config, method="kan_lora", dataset="pauq", seed=2)

    assert (changed.method, changed.dataset, changed.seed) == ("kan_lora", "pauq", 2)
    assert (config.method, config.dataset, config.seed) != (changed.method, changed.dataset, changed.seed)


def test_override_of_none_is_ignored(config: ExperimentConfig) -> None:
    """Неуказанный ключ командной строки не должен затирать значение из файла."""
    assert apply_overrides(config, method=None).method == config.method


def test_train_seed_follows_the_run_seed(config: ExperimentConfig) -> None:
    assert apply_overrides(config, seed=7).train.seed == 7


def test_run_id_encodes_the_whole_experiment_cell(config: ExperimentConfig) -> None:
    """Имя прогона обязано различать все 22 ячейки матрицы."""
    changed = apply_overrides(config, method="kan_lora", dataset="pauq", seed=2)
    assert changed.run_id == "kan_lora-pauq-adamw-seed2"

    muon = apply_overrides(config, method="lora", dataset="spider", optimizer_name="muon")
    assert muon.run_id == "lora-spider-muon-seed0"

    ablation = apply_overrides(config, method="kan_lora", learn_input_scale=False)
    assert ablation.run_id.endswith("-no-input-scale")


def test_serialization_round_trips(config: ExperimentConfig, tmp_path: Path) -> None:
    path = tmp_path / "copy.yaml"
    path.write_text(yaml.safe_dump(config_to_dict(config)), encoding="utf-8")
    assert load_config(path) == config


def test_smoke_configuration_is_small_but_shaped_like_the_real_one() -> None:
    """Проверочная конфигурация отличается только размерами, не устройством."""
    smoke = load_config(Path(__file__).parents[1] / "configs" / "smoke.yaml")
    base = load_config(BASE)

    assert smoke.train_subset == 200
    assert smoke.train.epochs == 1
    assert smoke.eval_limit == 50
    assert smoke.adapter == base.adapter
    assert smoke.max_length == base.max_length
```

- [ ] **Шаг 2: Написать падающий тест карточки результата**

```python
# tests/test_results.py
"""Проверки карточки результата.

Карточка — единица поставки: по ней восстанавливается прогон и из неё
собираются все таблицы. Поля, которых в ней нет, в работу не попадут.
"""

import json
from pathlib import Path

import pytest

from kanlora.config import load_config
from kanlora.results import build_result_card, collect_versions, git_commit, write_result_card
from kanlora.train.loop import TrainReport

BASE = Path(__file__).parents[1] / "configs" / "base.yaml"


@pytest.fixture
def card() -> dict:
    return build_result_card(
        config=load_config(BASE),
        injection={"trainable_parameters": 1234, "total_parameters": 500000,
                   "replaced": ["model.layers.0.self_attn.q_proj"]},
        train_report=TrainReport(
            step_losses=[2.0, 1.0], epoch_losses=[1.5], fraction_inside_grid=[0.87],
            peak_memory_bytes=4_000_000_000, seconds=4200.0,
            optimizer_coverage={"adamw": 1234, "muon": 0},
        ),
        metrics={"exact_match": 0.41, "execution_accuracy": 0.55,
                 "exact_match_by_hardness": {"easy": 0.7, "all": 0.41}},
        spline_stats={"model.layers.0.self_attn.q_proj": {"mean": 0.03, "max": 0.11,
                                                          "fraction_inside_grid": 0.87}},
    )


def test_card_carries_everything_needed_to_reproduce(card: dict) -> None:
    assert set(card) >= {
        "run_id", "config", "versions", "git_commit", "metrics",
        "peak_memory_bytes", "train_seconds", "trainable_parameters",
        "optimizer_coverage", "fraction_inside_grid", "spline_stats",
    }


def test_versions_include_the_libraries_that_affect_numbers() -> None:
    versions = collect_versions()
    assert set(versions) >= {"python", "torch", "transformers"}
    assert versions["torch"].startswith("2.")


def test_git_commit_is_recorded(card: dict) -> None:
    assert isinstance(card["git_commit"], str)
    assert card["git_commit"]


def test_absent_execution_accuracy_is_null_not_zero() -> None:
    """Отличие «не мерили» от «ноль» принципиально: PAUQ может остаться без баз данных."""
    card = build_result_card(
        config=load_config(BASE), injection={"trainable_parameters": 1, "total_parameters": 2,
                                             "replaced": []},
        train_report=TrainReport(), metrics={"exact_match": 0.1, "execution_accuracy": None},
        spline_stats=None,
    )
    assert card["metrics"]["execution_accuracy"] is None


def test_written_card_is_valid_json_named_by_run(card: dict, tmp_path: Path) -> None:
    path = write_result_card(card, tmp_path)

    assert path.name == f"{card['run_id']}.json"
    assert json.loads(path.read_text(encoding="utf-8")) == card


def test_card_is_written_in_readable_utf8(card: dict, tmp_path: Path) -> None:
    """Карточки лежат в git и читаются глазами — экранированной кириллицы там быть не должно."""
    card["note"] = "проверка"
    text = write_result_card(card, tmp_path).read_text(encoding="utf-8")
    assert "проверка" in text
```

- [ ] **Шаг 3: Запустить оба, убедиться в падении**

Run: `python -m pytest tests/test_config.py tests/test_results.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kanlora.config'`.

- [ ] **Шаг 4: Написать `configs/base.yaml`**

```yaml
# Основная конфигурация. Варианты матрицы задаются ключами командной строки,
# а не отдельными файлами: 22 почти одинаковых файла — это 22 места, где
# гиперпараметр может незаметно разойтись между методами.
model_name: Qwen/Qwen2.5-Coder-0.5B-Instruct
data_root: data

method: lora
dataset: spider
seed: 0

# Урезание обучающей выборки — единственный параметр, который допускается
# менять после старта, и менять его надо один раз для всех прогонов сразу.
train_subset: 2500
# Зерно выборки отдельно от зерна обучения: выборка обязана быть одной и той
# же для всех трёх методов и всех трёх зёрен.
subsample_seed: 12345

max_length: 768
eval_limit: null
eval_batch_size: 8
max_new_tokens: 128

adapter:
  rank: 8
  alpha: 16.0
  grid_size: 5
  spline_order: 3
  learn_input_scale: true

train:
  epochs: 2
  batch_size: 1
  gradient_accumulation: 8
  gradient_checkpointing: true
  log_every: 20

optimizer:
  name: adamw
  learning_rate: 2.0e-4
  weight_decay: 0.0
  max_grad_norm: 1.0
```

`configs/smoke.yaml` — то же самое, но с `train_subset: 200`, `epochs: 1`, `eval_limit: 50`. Отличается только размерами: устройство прогона обязано быть тем же, иначе проверенным окажется не тот путь.

- [ ] **Шаг 5: Написать `src/kanlora/config.py`**

```python
# src/kanlora/config.py
"""Конфигурация одного прогона.

Заморожена целиком и попадает в карточку результата дословно: воспроизводимость
держится на том, что по карточке прогон восстанавливается без обращения
к истории команд.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import yaml

from kanlora.adapters.base import AdapterConfig
from kanlora.train.loop import TrainConfig
from kanlora.train.optimizers import OptimizerConfig

__all__ = ["ExperimentConfig", "apply_overrides", "config_to_dict", "load_config"]


@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str
    data_root: str
    method: str
    dataset: str
    seed: int
    train_subset: int
    subsample_seed: int
    max_length: int
    eval_limit: int | None
    eval_batch_size: int
    max_new_tokens: int
    adapter: AdapterConfig
    train: TrainConfig
    optimizer: OptimizerConfig

    @property
    def run_id(self) -> str:
        """Имя прогона, различающее все ячейки матрицы экспериментов."""
        name = f"{self.method}-{self.dataset}-{self.optimizer.name}-seed{self.seed}"
        if self.method == "kan_lora" and not self.adapter.learn_input_scale:
            name += "-no-input-scale"
        return name


def load_config(path: Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ExperimentConfig(
        model_name=raw["model_name"],
        data_root=raw["data_root"],
        method=raw["method"],
        dataset=raw["dataset"],
        seed=raw["seed"],
        train_subset=raw["train_subset"],
        subsample_seed=raw["subsample_seed"],
        max_length=raw["max_length"],
        eval_limit=raw["eval_limit"],
        eval_batch_size=raw["eval_batch_size"],
        max_new_tokens=raw["max_new_tokens"],
        adapter=AdapterConfig(**raw["adapter"]),
        train=TrainConfig(seed=raw["seed"], **raw["train"]),
        optimizer=OptimizerConfig(**raw["optimizer"]),
    )


def apply_overrides(
    config: ExperimentConfig,
    *,
    method: str | None = None,
    dataset: str | None = None,
    seed: int | None = None,
    optimizer_name: str | None = None,
    learn_input_scale: bool | None = None,
    train_subset: int | None = None,
    eval_limit: int | None = None,
) -> ExperimentConfig:
    """Накладывает ключи командной строки. None означает «не трогать»."""
    updated = config
    if method is not None:
        updated = replace(updated, method=method)
    if dataset is not None:
        updated = replace(updated, dataset=dataset)
    if seed is not None:
        updated = replace(updated, seed=seed, train=replace(updated.train, seed=seed))
    if optimizer_name is not None:
        updated = replace(updated, optimizer=replace(updated.optimizer, name=optimizer_name))
    if learn_input_scale is not None:
        updated = replace(
            updated, adapter=replace(updated.adapter, learn_input_scale=learn_input_scale)
        )
    if train_subset is not None:
        updated = replace(updated, train_subset=train_subset)
    if eval_limit is not None:
        updated = replace(updated, eval_limit=eval_limit)
    return updated


def config_to_dict(config: ExperimentConfig) -> dict:
    data = asdict(config)
    data["train"].pop("seed", None)  # зерно живёт на верхнем уровне
    return data
```

- [ ] **Шаг 6: Написать `src/kanlora/results.py`**

```python
# src/kanlora/results.py
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
```

- [ ] **Шаг 7: Расширить меру нелинейности отрезком измерения**

Спецификация требует подгонять прямую **на том отрезке, куда реально попадают активации**, а не на всей сетке. Добавить в `KANLoRALinear` накопление фактического размаха и передать его в анализ.

Сначала тест (дописать в `tests/analysis/test_spline_stats.py`):

```python
def test_measurement_interval_is_taken_from_the_caller() -> None:
    """Мерить надо там, куда попадают активации, а не по всей области определения."""
    from kanlora.adapters.kan_layer import KANLayer

    layer = KANLayer(2, 2, grid_size=5, spline_order=3, dtype=torch.float64)
    torch.manual_seed(0)
    with torch.no_grad():
        layer.spline_coefficients.add_(0.5 * torch.randn_like(layer.spline_coefficients))

    wide = max(item.nonlinearity for item in layer_nonlinearity(layer, -0.99, 0.99))
    narrow = max(item.nonlinearity for item in layer_nonlinearity(layer, -0.05, 0.05))
    assert narrow < wide, "на узком отрезке любая гладкая функция ближе к прямой"


def test_model_summary_uses_supplied_ranges(tiny_causal_lm) -> None:
    from kanlora.adapters.base import AdapterConfig
    from kanlora.adapters.inject import adapter_modules, inject_adapters

    inject_adapters(tiny_causal_lm, "kan_lora", AdapterConfig(rank=4))
    for _, module in adapter_modules(tiny_causal_lm):
        with torch.no_grad():
            module.kan.spline_coefficients.add_(
                0.5 * torch.randn_like(module.kan.spline_coefficients)
            )

    names = [name for name, _ in adapter_modules(tiny_causal_lm)]
    wide = model_nonlinearity(tiny_causal_lm, {name: (-0.99, 0.99) for name in names})
    narrow = model_nonlinearity(tiny_causal_lm, {name: (-0.05, 0.05) for name in names})

    assert narrow[names[0]]["max"] < wide[names[0]]["max"]
```

Затем реализация: в `KANLoRALinear` завести буферы `observed_lo`, `observed_hi` (регистрируются как `register_buffer`, обновляются в `delta` под `torch.no_grad()` минимумом и максимумом `projected` за проход, стартовое значение — `+inf` и `-inf`), метод `observed_range() -> tuple[float, float]`. В `spline_stats.model_nonlinearity` добавить параметр `ranges: dict[str, tuple[float, float]] | None = None`: при `None` брать `module.observed_range()`, а если размах ещё не наблюдался — сетку. Дописать в `tests/adapters/test_kan_lora.py` тест на то, что `observed_range` расширяется после прохода.

- [ ] **Шаг 8: Написать `scripts/run_experiment.py`**

```python
# scripts/run_experiment.py
"""Один прогон целиком: обучение -> оценка -> карточка результата.

Эта же команда без изменений запускает и проверку на 200 примерах, и полный
прогон. Отдельного «настоящего» пути запуска не существует: иначе проверенным
оказался бы не тот код, который считает результаты.

Примеры:
    python scripts/run_experiment.py --config configs/smoke.yaml
    python scripts/run_experiment.py --config configs/base.yaml --method kan_lora \
        --dataset pauq --seed 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kanlora.adapters.inject import inject_adapters  # noqa: E402
from kanlora.analysis.spline_stats import model_nonlinearity  # noqa: E402
from kanlora.config import apply_overrides, load_config  # noqa: E402
from kanlora.data.collate import Collator, build_dataset, build_prompts  # noqa: E402
from kanlora.data.loaders import (  # noqa: E402
    LAYOUTS, database_path, load_dataset_schemas, load_examples, subsample,
)
from kanlora.eval.exact_match import exact_match_by_hardness  # noqa: E402
from kanlora.eval.execution import execution_accuracy  # noqa: E402
from kanlora.eval.generate import generate_sql  # noqa: E402
from kanlora.results import build_result_card, write_result_card  # noqa: E402
from kanlora.train.loop import set_seed, train  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Один прогон эксперимента")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--method", choices=["lora", "dora", "kan_lora"])
    parser.add_argument("--dataset", choices=["spider", "pauq"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--optimizer", dest="optimizer_name", choices=["adamw", "muon"])
    parser.add_argument(
        "--no-input-scale",
        dest="learn_input_scale",
        action="store_const",
        const=False,
        help="абляция: сетка сплайна не растягивается, масштаб входа заморожен на 1.0",
    )
    parser.add_argument("--train-subset", type=int)
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--output", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    config = apply_overrides(
        load_config(arguments.config),
        method=arguments.method,
        dataset=arguments.dataset,
        seed=arguments.seed,
        optimizer_name=arguments.optimizer_name,
        learn_input_scale=arguments.learn_input_scale,
        train_subset=arguments.train_subset,
        eval_limit=arguments.eval_limit,
    )
    print(f"прогон: {config.run_id}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config.seed)

    root = Path(config.data_root) / config.dataset
    layout = LAYOUTS[config.dataset]
    schemas = load_dataset_schemas(root, layout)

    train_examples = subsample(
        load_examples(root, layout, "train"), config.train_subset, config.subsample_seed
    )
    eval_examples = load_examples(root, layout, "eval")
    if config.eval_limit is not None:
        eval_examples = eval_examples[: config.eval_limit]

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # fp32 явно и одинаково для всех трёх методов: сравнение по видеопамяти
    # осмысленно только при одинаковой точности.
    model = AutoModelForCausalLM.from_pretrained(config.model_name, dtype=torch.float32)
    injection = inject_adapters(model, config.method, config.adapter)
    print(
        f"обучаемых параметров: {injection.trainable_parameters} "
        f"из {injection.total_parameters}, затронуто слоёв: {len(injection.replaced)}"
    )

    dataset, truncated = build_dataset(train_examples, schemas, tokenizer, config.max_length)
    if truncated:
        print(f"внимание: усечено промптов: {truncated} из {len(dataset)}")

    report = train(
        model=model,
        dataset=dataset,
        collator=Collator(pad_token_id=tokenizer.pad_token_id),
        optimizer_config=config.optimizer,
        train_config=config.train,
        device=device,
    )

    predicted = generate_sql(
        model, tokenizer, build_prompts(eval_examples, schemas),
        device=device, batch_size=config.eval_batch_size,
        max_new_tokens=config.max_new_tokens,
    )
    gold = [example.query for example in eval_examples]
    db_ids = [example.db_id for example in eval_examples]

    hardness = exact_match_by_hardness(gold, predicted, db_ids, root / layout.tables_file)
    metrics = {
        "exact_match": hardness["all"],
        "exact_match_by_hardness": hardness,
        "execution_accuracy": None,
        "truncated_prompts": truncated,
        "evaluated": len(gold),
    }

    # Execution Accuracy требует файлов баз. Их отсутствие — заранее оговорённое
    # отступление, а не сбой: метрика остаётся null, и это видно в карточке.
    if database_path(root, layout, db_ids[0]).is_file():
        metrics["execution_accuracy"] = execution_accuracy(
            gold, predicted, db_ids, root, layout
        )
    else:
        print("файлы баз данных не найдены: Execution Accuracy не считается")

    spline_stats = model_nonlinearity(model) or None
    card = build_result_card(
        config=config,
        injection={
            "trainable_parameters": injection.trainable_parameters,
            "total_parameters": injection.total_parameters,
            "replaced": list(injection.replaced),
        },
        train_report=report,
        metrics=metrics,
        spline_stats=spline_stats,
    )
    path = write_result_card(card, arguments.output)

    print(
        f"Exact Match {metrics['exact_match']:.4f}, "
        f"Execution Accuracy {metrics['execution_accuracy']}, "
        f"пик видеопамяти {report.peak_memory_bytes / 2**30:.2f} ГиБ, "
        f"время {report.seconds / 60:.1f} мин"
    )
    print(f"карточка: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Шаг 9: Запустить тесты, убедиться, что они проходят**

Run: `python -m pytest -q`
Expected: PASS.

- [ ] **Шаг 10: Сквозная проверка на 200 примерах**

Это требование верификации из спецификации: та же команда, урезанная конфигурация, на выходе — карточка с ненулевыми метриками.

```bash
python scripts/run_experiment.py --config configs/smoke.yaml --method lora --dataset spider
python scripts/run_experiment.py --config configs/smoke.yaml --method dora --dataset spider
python scripts/run_experiment.py --config configs/smoke.yaml --method kan_lora --dataset spider
```

Expected для каждого: печатается число обучаемых параметров, функция потерь по эпохе, для `kan_lora` — доля активаций в сетке, затем Exact Match и Execution Accuracy **строго больше нуля** и путь к карточке.

**Разбор возможных исходов:**

- **Exact Match ровно ноль** — модель не выдаёт разбираемый SQL. Смотреть на 3–5 сгенерированных строк глазами: скорее всего, промпт усечён (`внимание: усечено промптов`) или генерация обрывается раньше запроса (`max_new_tokens`).
- **Execution Accuracy ровно ноль при ненулевом Exact Match** — не находятся файлы баз или сравнение результатов сломано; проверить `scripts/check_environment.py`.
- **`доля активаций в сетке` близка к нулю** — сплайны вырождены, адаптер работает как LoRA с неработающими параметрами. Это ожидаемое наблюдение в начале обучения; смотреть, растёт ли доля к концу.
- **Пик видеопамяти = 0** — прогон идёт на процессоре, а не на карте. На машине обучения это ошибка.

- [ ] **Шаг 11: Убрать проверочные карточки и закоммитить**

```bash
rm -f results/lora-spider-adamw-seed0.json results/dora-spider-adamw-seed0.json \
      results/kan_lora-spider-adamw-seed0.json
git add configs scripts/run_experiment.py src/kanlora/config.py src/kanlora/results.py \
        src/kanlora/analysis/spline_stats.py src/kanlora/adapters/kan_lora.py \
        results/.gitkeep tests/
git commit -m "feat: add experiment runner producing reproducible result cards"
```

---

### Задача 17: Задержка генерации со слиянием и без

Цена нелинейного метода, которой нет в задании. LoRA и DoRA после обучения складываются с весами и на инференсе стоят ноль; **KAN-LoRA нелинеен и не сливается в принципе**, поэтому платит задержкой всегда. Здесь эта цена получает число.

**Files:**
- Create: `src/kanlora/eval/latency.py`
- Create: `tests/eval/test_latency.py`

**Interfaces:**
- Consumes: `AdapterLinear` из `kanlora.adapters.base`; `adapter_modules` из `kanlora.adapters.inject`.
- Produces:
  - `LatencyReport(seconds_per_token: float, total_seconds: float, tokens: int, merged: bool)`.
  - `merge_adapters(model) -> int` — сливает все сливаемые адаптеры на месте, возвращает их число; на несливаемом бросает `NotImplementedError`.
  - `measure_latency(model, tokenizer, prompt: str, *, device, max_new_tokens: int = 64, repeats: int = 3, warmup: int = 1, merged: bool = False) -> LatencyReport`.

- [ ] **Шаг 1: Написать падающий тест**

```python
# tests/eval/test_latency.py
"""Проверки замера задержки.

LoRA и DoRA после обучения сливаются с весами и на инференсе не стоят ничего.
KAN-LoRA не сливается в принципе: поправка нелинейно зависит от входа. Это его
цена, и здесь она получает число.
"""

import pytest
import torch

from kanlora.adapters.base import AdapterConfig
from kanlora.adapters.inject import adapter_modules, inject_adapters
from kanlora.eval.latency import LatencyReport, measure_latency, merge_adapters

DEVICE = torch.device("cpu")


@pytest.mark.parametrize("method", ["lora", "dora"])
def test_merging_replaces_adapters_with_plain_linear(tiny_causal_lm, method: str) -> None:
    inject_adapters(tiny_causal_lm, method, AdapterConfig(rank=4))
    merged = merge_adapters(tiny_causal_lm)

    assert merged == 14  # 2 слоя * 7 проекций
    assert adapter_modules(tiny_causal_lm) == []


@pytest.mark.parametrize("method", ["lora", "dora"])
def test_merging_does_not_change_the_output(tiny_causal_lm, method: str) -> None:
    """Слияние обязано быть тождественным преобразованием, иначе замер сравнивает разные модели."""
    inject_adapters(tiny_causal_lm, method, AdapterConfig(rank=4))
    for _, module in adapter_modules(tiny_causal_lm):
        with torch.no_grad():
            module.lora_b.normal_(std=0.01)

    tokens = torch.randint(0, 64, (2, 9))
    with torch.no_grad():
        before = tiny_causal_lm(tokens).logits.clone()
        merge_adapters(tiny_causal_lm)
        after = tiny_causal_lm(tokens).logits

    assert torch.allclose(before, after, atol=1e-5)


def test_kan_lora_refuses_to_merge(tiny_causal_lm) -> None:
    """Не «пока не реализовано», а свойство метода: поправка зависит от входа."""
    inject_adapters(tiny_causal_lm, "kan_lora", AdapterConfig(rank=4))
    with pytest.raises(NotImplementedError):
        merge_adapters(tiny_causal_lm)


@pytest.mark.slow
def test_latency_report_is_filled(tiny_causal_lm, tiny_tokenizer) -> None:
    inject_adapters(tiny_causal_lm, "lora", AdapterConfig(rank=4))
    report = measure_latency(
        tiny_causal_lm, tiny_tokenizer, "SELECT",
        device=DEVICE, max_new_tokens=8, repeats=2, warmup=1,
    )

    assert isinstance(report, LatencyReport)
    assert report.tokens == 8
    assert report.total_seconds > 0
    assert report.seconds_per_token == pytest.approx(report.total_seconds / report.tokens)
    assert report.merged is False
```

- [ ] **Шаг 2: Запустить, убедиться в падении**

Run: `python -m pytest tests/eval/test_latency.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kanlora.eval.latency'`.

- [ ] **Шаг 3: Написать минимальную реализацию**

```python
# src/kanlora/eval/latency.py
"""Задержка генерации со слиянием адаптера и без.

LoRA и DoRA после обучения складываются с весами: W + BA считается один раз,
и на инференсе метод не стоит ничего. KAN-LoRA не сливается в принципе —
поправка B * phi(A x) зависит от входа нелинейно, — поэтому платит задержкой
на каждом токене всегда. Этой цены нет в задании, и она измеряется здесь.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from kanlora.adapters.base import AdapterLinear
from kanlora.adapters.inject import adapter_modules

__all__ = ["LatencyReport", "measure_latency", "merge_adapters"]


@dataclass(frozen=True)
class LatencyReport:
    seconds_per_token: float
    total_seconds: float
    tokens: int
    merged: bool


def merge_adapters(model: torch.nn.Module) -> int:
    """Заменяет каждый адаптер обычным nn.Linear с поглощённой поправкой."""
    merged = 0
    for name, module in adapter_modules(model):
        if not isinstance(module, AdapterLinear):
            continue
        replacement = module.merge()  # на KAN-LoRA бросит NotImplementedError
        parent_name, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attribute, replacement)
        merged += 1
    return merged


@torch.inference_mode()
def measure_latency(
    model,
    tokenizer,
    prompt: str,
    *,
    device: torch.device,
    max_new_tokens: int = 64,
    repeats: int = 3,
    warmup: int = 1,
    merged: bool = False,
) -> LatencyReport:
    """Среднее время жадной генерации фиксированного числа токенов.

    Прогрев обязателен: первый вызов включает выделение памяти и подбор ядер,
    и без него замер сравнивал бы разогрев, а не метод.
    """
    model.to(device)
    model.eval()
    encoded = tokenizer([prompt], return_tensors="pt").to(device)

    def generate() -> None:
        model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,  # ровно столько токенов во всех замерах
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
        )

    for _ in range(warmup):
        generate()
    if device.type == "cuda":
        torch.cuda.synchronize()

    started = time.perf_counter()
    for _ in range(repeats):
        generate()
    if device.type == "cuda":
        torch.cuda.synchronize()
    total = (time.perf_counter() - started) / repeats

    return LatencyReport(
        seconds_per_token=total / max_new_tokens,
        total_seconds=total,
        tokens=max_new_tokens,
        merged=merged,
    )
```

- [ ] **Шаг 4: Запустить тесты, убедиться, что они проходят**

Run: `python -m pytest -q`
Expected: PASS.

- [ ] **Шаг 5: Коммит**

```bash
git add src/kanlora/eval/latency.py tests/eval/test_latency.py
git commit -m "feat: measure generation latency with and without adapter merging"
```

---

### Задача 18: Сборка таблиц и графиков

Таблицы собираются из карточек одной командой и руками не правятся. Это и есть то, что репозиторий обязан выдавать для текста РПЗ.

**Files:**
- Create: `scripts/make_tables.py`
- Create: `tests/test_make_tables.py`

**Interfaces:**
- Consumes: карточки результатов из `results/*.json`.
- Produces:
  - `load_cards(directory: Path) -> list[dict]`.
  - `aggregate(cards: list[dict]) -> dict[tuple[str, str], dict[str, tuple[float, float]]]` — по ключу «метод, набор» даёт `{метрика: (среднее, размах)}`.
  - `main_table(cards: list[dict]) -> str` — таблица в разметке Markdown.
  - `nonlinearity_table(cards: list[dict]) -> str`.
  - `optimizer_table(cards: list[dict]) -> str`.
  - `plot_fraction_inside_grid(cards, path: Path) -> None`.

- [ ] **Шаг 1: Написать падающий тест**

```python
# tests/test_make_tables.py
"""Проверки сборки таблиц.

Выводы формулируются только по различиям, выходящим за разброс по зерну,
поэтому таблица обязана показывать разброс рядом со средним. Таблица,
показывающая одно среднее, скрывала бы ровно то, ради чего заложены три зерна.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_tables.py"


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("make_tables", SCRIPT)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules["make_tables"] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def card(method: str, dataset: str, seed: int, exact: float, execution: float | None = 0.5) -> dict:
    return {
        "run_id": f"{method}-{dataset}-adamw-seed{seed}",
        "method": method, "dataset": dataset, "seed": seed, "optimizer": "adamw",
        "metrics": {"exact_match": exact, "execution_accuracy": execution},
        "peak_memory_bytes": 4 * 2**30, "train_seconds": 3600.0,
        "trainable_parameters": 100_000,
        "optimizer_coverage": {"adamw": 100_000, "muon": 0},
        "fraction_inside_grid": [0.8, 0.9],
        "spline_stats": None,
    }


@pytest.fixture
def cards() -> list[dict]:
    return [
        card("lora", "spider", 0, 0.40), card("lora", "spider", 1, 0.42),
        card("lora", "spider", 2, 0.44),
        card("kan_lora", "spider", 0, 0.45), card("kan_lora", "spider", 1, 0.43),
        card("kan_lora", "spider", 2, 0.47),
    ]


def test_loads_every_card(module, cards, tmp_path: Path) -> None:
    for item in cards:
        (tmp_path / f"{item['run_id']}.json").write_text(json.dumps(item), encoding="utf-8")
    assert len(module.load_cards(tmp_path)) == len(cards)


def test_aggregate_reports_mean_and_spread(module, cards) -> None:
    aggregated = module.aggregate(cards)
    mean, spread = aggregated[("lora", "spider")]["exact_match"]

    assert mean == pytest.approx(0.42)
    assert spread == pytest.approx(0.04)  # размах: 0.44 - 0.40


def test_single_seed_has_zero_spread(module) -> None:
    aggregated = module.aggregate([card("dora", "pauq", 0, 0.3)])
    assert aggregated[("dora", "pauq")]["exact_match"][1] == pytest.approx(0.0)


def test_absent_execution_accuracy_does_not_become_zero(module) -> None:
    """PAUQ может остаться без баз; нельзя усреднять «не мерили» с нулём."""
    aggregated = module.aggregate([card("lora", "pauq", 0, 0.3, execution=None)])
    assert aggregated[("lora", "pauq")]["execution_accuracy"] is None


def test_main_table_shows_spread_next_to_the_mean(module, cards) -> None:
    table = module.main_table(cards)
    assert "±" in table or "разброс" in table
    assert "lora" in table and "kan_lora" in table
    assert table.count("|") > 10


def test_main_table_includes_the_baseline_row_placeholder(module, cards) -> None:
    """В таблицах обязателен базовый уровень без адаптера: видно, что дала адаптация."""
    assert "без адаптера" in module.main_table(cards)


def test_optimizer_table_shows_muon_coverage(module, cards) -> None:
    table = module.optimizer_table(cards)
    assert "Muon" in table and "AdamW" in table


def test_plot_writes_a_file(module, cards, tmp_path: Path) -> None:
    path = tmp_path / "grid.png"
    module.plot_fraction_inside_grid(cards, path)
    assert path.is_file() and path.stat().st_size > 0
```

- [ ] **Шаг 2: Запустить, убедиться в падении**

Run: `python -m pytest tests/test_make_tables.py -v`
Expected: FAIL — файла `scripts/make_tables.py` нет.

- [ ] **Шаг 3: Написать минимальную реализацию**

```python
# scripts/make_tables.py
"""Сборка таблиц и графиков из карточек результатов.

Таблицы собираются одной командой и руками не правятся: правка руками
означала бы, что число в работе не восстанавливается из карточек.

Рядом со средним всегда стоит размах по зёрнам. Различие между методами по
Exact Match ожидается порядка одного-двух процентов, то есть внутри разброса
по случайному зерну, и выводы формулируются только по различиям, выходящим
за него. Таблица с одним средним скрывала бы ровно это.

Запуск: python scripts/make_tables.py [--results results] [--output results]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METHOD_NAMES = {"lora": "LoRA", "dora": "DoRA", "kan_lora": "KAN-LoRA"}
DATASET_NAMES = {"spider": "Spider", "pauq": "PAUQ"}
METRICS = ("exact_match", "execution_accuracy")


def load_cards(directory: Path) -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(Path(directory).glob("*.json"))
    ]


def aggregate(cards: list[dict]) -> dict[tuple[str, str], dict]:
    """Среднее и размах по зёрнам для каждой ячейки «метод, набор».

    Размах, а не стандартное отклонение: при трёх зёрнах отклонение оценивается
    плохо, а размах говорит ровно то, что нужно, — насколько далеко разъезжаются
    прогоны одного метода.
    """
    buckets: dict[tuple[str, str], dict[str, list[float | None]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for card in cards:
        key = (card["method"], card["dataset"])
        for metric in METRICS:
            buckets[key][metric].append(card["metrics"].get(metric))
        buckets[key]["peak_memory_bytes"].append(card["peak_memory_bytes"])
        buckets[key]["trainable_parameters"].append(card["trainable_parameters"])

    aggregated: dict[tuple[str, str], dict] = {}
    for key, metrics in buckets.items():
        summary: dict[str, tuple[float, float] | None] = {}
        for name, values in metrics.items():
            present = [value for value in values if value is not None]
            # «Не мерили» не усредняется с нулём: PAUQ может остаться без баз.
            summary[name] = (
                (sum(present) / len(present), max(present) - min(present)) if present else None
            )
        aggregated[key] = summary
    return aggregated


def _cell(value: tuple[float, float] | None, digits: int = 4) -> str:
    if value is None:
        return "не измерялась"
    mean, spread = value
    return f"{mean:.{digits}f} ± {spread / 2:.{digits}f}"


def main_table(cards: list[dict]) -> str:
    aggregated = aggregate(cards)
    lines = [
        "| Метод | Набор | Exact Match | Execution Accuracy | Пик видеопамяти, ГиБ | Обучаемых параметров |",
        "|---|---|---|---|---|---|",
        "| без адаптера (базовый уровень) | — | заполнить прогоном базовой модели | | | 0 |",
    ]
    for (method, dataset), summary in sorted(aggregated.items()):
        memory = summary["peak_memory_bytes"]
        parameters = summary["trainable_parameters"]
        lines.append(
            f"| {METHOD_NAMES.get(method, method)} | {DATASET_NAMES.get(dataset, dataset)} "
            f"| {_cell(summary['exact_match'])} | {_cell(summary['execution_accuracy'])} "
            f"| {memory[0] / 2**30:.2f} | {int(parameters[0])} |"
        )
    return "\n".join(lines)


def nonlinearity_table(cards: list[dict]) -> str:
    lines = [
        "| Прогон | Средний индекс нелинейности | Максимальный | Доля активаций в сетке |",
        "|---|---|---|---|",
    ]
    for card in sorted(cards, key=lambda item: item["run_id"]):
        stats = card.get("spline_stats")
        if not stats:
            continue
        means = [values["mean"] for values in stats.values()]
        maxima = [values["max"] for values in stats.values()]
        fractions = card.get("fraction_inside_grid") or [float("nan")]
        lines.append(
            f"| {card['run_id']} | {sum(means) / len(means):.4f} | {max(maxima):.4f} "
            f"| {fractions[-1]:.3f} |"
        )
    return "\n".join(lines)


def optimizer_table(cards: list[dict]) -> str:
    """Покрытие параметров оптимизатором Muon — самостоятельный результат работы."""
    lines = ["| Прогон | Параметров в Muon | Параметров в AdamW | Доля в Muon |", "|---|---|---|---|"]
    for card in sorted(cards, key=lambda item: item["run_id"]):
        coverage = card.get("optimizer_coverage") or {}
        muon, adamw = coverage.get("muon", 0), coverage.get("adamw", 0)
        total = muon + adamw
        lines.append(
            f"| {card['run_id']} | {muon} | {adamw} | "
            f"{muon / total:.3f} |" if total else f"| {card['run_id']} | 0 | 0 | — |"
        )
    return "\n".join(lines)


def plot_fraction_inside_grid(cards: list[dict], path: Path) -> None:
    figure, axes = plt.subplots(figsize=(7, 4))
    for card in sorted(cards, key=lambda item: item["run_id"]):
        values = card.get("fraction_inside_grid")
        if values:
            axes.plot(range(1, len(values) + 1), values, marker="o", label=card["run_id"])

    axes.set_xlabel("эпоха")
    axes.set_ylabel("доля активаций внутри сетки")
    axes.set_title("Вырождается ли сплайновый базис в ходе обучения")
    axes.set_ylim(0.0, 1.05)
    axes.grid(True, alpha=0.3)
    if axes.get_legend_handles_labels()[0]:
        axes.legend(fontsize="small")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description="Сборка таблиц и графиков из карточек")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    arguments = parser.parse_args()

    cards = load_cards(arguments.results)
    if not cards:
        print(f"карточек не найдено в {arguments.results}")
        return 1

    tables = arguments.output / "tables.md"
    tables.parent.mkdir(parents=True, exist_ok=True)
    tables.write_text(
        "# Результаты\n\n## Основная матрица\n\n"
        + main_table(cards)
        + "\n\n## Выученная нелинейность\n\n"
        + nonlinearity_table(cards)
        + "\n\n## Покрытие параметров оптимизатором Muon\n\n"
        + optimizer_table(cards)
        + "\n",
        encoding="utf-8",
    )
    plot_fraction_inside_grid(cards, arguments.output / "figures" / "fraction_inside_grid.png")

    print(f"собрано карточек: {len(cards)}; таблицы: {tables}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

**Замечание.** Строка с тернарным выражением в `optimizer_table` разобрана неверно: условие `if total else` относится ко всему f-выражению, из-за чего при `total == 0` выводится строка без числителей. Реализуя шаг, переписать это обычным `if/else` в две строки — тест `test_optimizer_table_shows_muon_coverage` проходит в обоих случаях, но читать надо однозначно.

- [ ] **Шаг 4: Запустить тесты, убедиться, что они проходят**

Run: `python -m pytest -q`
Expected: PASS.

- [ ] **Шаг 5: Коммит**

```bash
git add scripts/make_tables.py tests/test_make_tables.py
git commit -m "feat: build result tables and figures from result cards"
```

---

### Задача 19: Замер времени, проверка на расхождение и запуск матрицы

Последняя задача, и единственная, которой нужна GTX 1080 Ti. Кода здесь нет — есть порядок действий, отступление от которого стоит недели.

**Files:**
- Create: `docs/runbook.md`
- Modify: `results/*.json` (пополняется прогонами)
- Modify: `CLAUDE.md` (раздел «Key Commands» — команды наконец существуют)

- [ ] **Шаг 1: Проверить окружение на машине обучения**

```bash
python scripts/check_environment.py --data-root data
python -m pytest -q
```

Expected: `CUDA` — OK, `поддержка sm_61 в сборке` — OK, все наборы — OK, все тесты проходят.

**Если `sm_61` не поддержан** — работа блокирована на самом старте; см. разбор в Задаче 1, шаг 7. Сообщить преподавателю в тот же день.

- [ ] **Шаг 2: Замерить фактическое время одного прогона**

Это первое, что делается после сквозной проверки, и определяет весь бюджет. Оценка в 1.2 часа на прогон может ошибаться вдвое.

```bash
python scripts/run_experiment.py --config configs/base.yaml --method kan_lora \
    --dataset spider --seed 0 --train-subset 250 --eval-limit 100 --output results/timing
```

Взять `train_seconds` из карточки, умножить на 10 (полная выборка 2500) — это оценка времени обучения одного прогона; прибавить время оценки, отмасштабированное с 100 примеров на полный отложенный набор (Spider dev — 1034 примера, то есть примерно в 10 раз).

Замерить именно **KAN-LoRA**: он самый дорогой из трёх, и бюджет надо считать по нему.

- [ ] **Шаг 3: Подобрать размер обучающей выборки под замер**

22 прогона обязаны уложиться в 30 часов. Если по замеру `22 * оценка_на_прогон > 28 часов` — **уменьшить `train_subset` один раз для всех прогонов сразу** и записать новое значение в `configs/base.yaml`. Это единственный параметр, который допускается менять после старта.

Что менять нельзя ни при каких обстоятельствах: число зёрен (три), число эпох (два), длину последовательности (768), ранг, размер сетки, порядок сплайна, скорость обучения. Три зерна — не запас, а условие доказуемости: различие между методами ожидается порядка процента, то есть внутри разброса по зерну.

Порядок снятия прогонов при нехватке времени задан спецификацией: **первыми снимаются три прогона с Muon, затем абляция. Основная матрица с тремя зёрнами не режется.**

- [ ] **Шаг 4: Проверить, что ни один метод не расходится**

500 примеров, по одному прогону на метод. Проверяется не качество, а то, что функция потерь падает, а не растёт и не даёт `nan`.

```bash
for method in lora dora kan_lora; do
  python scripts/run_experiment.py --config configs/base.yaml --method $method \
      --dataset spider --seed 0 --train-subset 500 --eval-limit 100 --output results/sanity
done
```

Expected: у каждого метода функция потерь по второй эпохе меньше, чем по первой, ни одного `nan`. Скорость обучения одна на все три метода, и это оговорено как ограничение работы; если один метод расходится, а два других — нет, **не подбирать ему отдельную скорость обучения** (это сломало бы контролируемость сравнения), а разбираться в реализации.

- [ ] **Шаг 5: Запустить основную матрицу**

18 прогонов: 3 метода × 2 набора × 3 зерна. Идут фоном, пока пишется текст.

```bash
for dataset in spider pauq; do
  for method in lora dora kan_lora; do
    for seed in 0 1 2; do
      python scripts/run_experiment.py --config configs/base.yaml \
          --method $method --dataset $dataset --seed $seed
    done
  done
done
```

После каждых нескольких прогонов коммитить карточки:

```bash
git add results/*.json
git commit -m "chore: add result cards for <что именно прогнано>"
```

- [ ] **Шаг 6: Прогоны с Muon и абляция**

3 прогона с Muon (по одному на метод, Spider, одно зерно) и 1 абляционный.

```bash
for method in lora dora kan_lora; do
  python scripts/run_experiment.py --config configs/base.yaml \
      --method $method --dataset spider --seed 0 --optimizer muon
done

python scripts/run_experiment.py --config configs/base.yaml \
    --method kan_lora --dataset spider --seed 0 --no-input-scale
```

В абляционном прогоне смотреть на `fraction_inside_grid`: если без растяжения сетки доля активаций внутри неё резко ниже — это и есть готовый ответ на вопрос «а вы понимаете, что там внутри».

- [ ] **Шаг 7: Замерить задержку со слиянием и без**

Отдельный короткий сценарий поверх `measure_latency` и `merge_adapters` (Задача 17): загрузить модель, внедрить каждый из трёх методов, замерить задержку до слияния, затем — для LoRA и DoRA — после слияния. Для KAN-LoRA слияние невозможно, и это записывается как результат, а не как пропуск.

- [ ] **Шаг 8: Собрать таблицы**

```bash
python scripts/make_tables.py --results results --output results
```

Заполнить строку «без адаптера (базовый уровень)» отдельным прогоном модели без обучения (`--train-subset 0`, если поддерживается, либо оценкой исходной модели). Без этой строки непонятно, какая доля результата получена адаптацией, а Qwen2.5-Coder и без адаптера пишет SQL неплохо — это честный недостаток выбора модели, и он должен быть виден в таблице.

- [ ] **Шаг 9: Обновить `CLAUDE.md`**

Заменить раздел «Key Commands» на фактические команды:

```markdown
## Key Commands

- Тесты: `python -m pytest -q` (быстрые), `python -m pytest -q -m "not slow"` (без обучения).
- Проверка окружения и данных: `python scripts/check_environment.py`.
- Один прогон: `python scripts/run_experiment.py --config configs/base.yaml --method {lora|dora|kan_lora} --dataset {spider|pauq} --seed N`.
- Проверочный прогон: то же с `--config configs/smoke.yaml`.
- Таблицы и графики: `python scripts/make_tables.py`.
```

Убрать из `CLAUDE.md` фразу «This repo currently has no code» — она перестала быть верной.

- [ ] **Шаг 10: Коммит**

```bash
git add results CLAUDE.md docs/runbook.md
git commit -m "chore: add experiment matrix results and tables"
```

---

## Самопроверка плана против спецификации

**Покрытие требований спецификации:**

| Требование спецификации | Задача |
|---|---|
| `pyproject.toml`, фиксация версий | 1 (готово частично, дополняется) |
| Проверка Muon в `torch.optim` | 1; **эталонная реализация отменена** — Muon есть в torch 2.14.0 |
| Проверка доступности файлов баз для Execution Accuracy | 1, шаг 8 |
| `spline.py`, `kan_layer.py` | готовы до начала плана |
| `base.py`, `lora.py`, `dora.py`, `kan_lora.py`, `inject.py` | 2, 3, 4, 5, 6 |
| `schema.py`, `spider.py`, `pauq.py`, `collate.py` | 7, 8, 9 |
| `loop.py`, `optimizers.py`, `memory.py` | 10, 11 |
| `generate.py`, `exact_match.py`, `execution.py` | 12, 13, 14 |
| `spline_stats.py` | 15 |
| `configs/`, `run_experiment.py`, `make_tables.py`, `results/` | 16, 18 |
| Тест: базис B-сплайнов (сумма 1, носитель, эталон) | готов |
| Тест: нулевая поправка при инициализации | 3, 4, 5 (по адаптеру) и 6 (на уровне модели) |
| Тест: KAN-LoRA с тождественными сплайнами совпадает с LoRA | 5 |
| Тест: DoRA при единичном модуле сводится к LoRA | 4 |
| Тест: счётчики параметров совпадают с формулой | 3, 4, 5, 6 |
| Тест: сериализация схемы детерминирована | 7 |
| Тест: маска потерь покрывает только токены SQL | 9 |
| Тест: Exact Match на известных парах Spider | 13 |
| Тест: разводка по оптимизаторам, ничего не потеряно и не продублировано | 10 |
| Тест на переобучение 20 примеров | 11 |
| Сверка списка параметров с `requires_grad=True` | 6 и 11 |
| Сквозной прогон на 200 примерах | 16, шаг 10 |
| Замер фактического времени, подбор размера выборки | 19, шаги 2–3 |
| Проверка на 500 примерах, что ни один метод не расходится | 19, шаг 4 |
| Основная матрица 18 прогонов | 19, шаг 5 |
| Muon: 3 прогона; абляция: 1 прогон | 19, шаг 6 |
| Замер задержки со слиянием и без | 17 и 19, шаг 7 |
| Доля активаций в сетке логируется каждую эпоху | 11 (в цикле), 18 (график) |
| Подгонка прямой к выученным функциям | 15, 16 (шаг 7 — отрезок по фактическим активациям) |
| Базовый уровень без адаптера в таблицах | 18 (строка таблицы), 19, шаг 8 (заполнение) |
| Воспроизводимость: зерно, конфигурация, версии в карточке | 16 |
| Видеопамять через `max_memory_allocated` при одинаковых условиях | 11 |

**Расхождения, оставленные сознательно** (все три перечислены в разделе «Отклонения от спецификации» выше и утверждены): `input_scale` вместо `tanh` и соответствующее изменение абляции; отмена эталонной реализации Muon; чтение наборов с диска вместо `datasets`.

**Три места, где реализующему предписано отойти от буквы плана** (в тексте задач помечены как «Замечание»): устранение дублирования `DatasetLayout` в Задаче 8, упрощение подсчёта усечений в Задаче 9, поддельный токенизатор вместо сетевого в Задаче 12, разбор тернарного выражения в Задаче 18. Каждое из них закреплено тестом, поэтому отход проверяем.

**Согласованность типов и имён проверена.** Отдельно стоит отметить один разобранный при самопроверке случай: норма обрезания градиента объявлена **только** в `OptimizerConfig`, и цикл обучения берёт её оттуда. В `TrainConfig` её нет намеренно — два источника одного значения разошлись бы молча, изменив методологию для части прогонов и не выдав ошибки. `configs/base.yaml` и тест `test_fixed_hyperparameters_match_the_plan` написаны согласованно с этим.

Сквозные имена, совпадающие во всех задачах, где они встречаются: `AdapterConfig`, `AdapterLinear`, `analytic_parameter_count`, `trainable_parameter_count`, `can_merge`, `merge`, `lora_a` / `lora_b`, `KANLayer.reset_to_identity`, `fraction_inside_grid`, `last_fraction_inside_grid`, `inject_adapters`, `adapter_modules`, `InjectionReport`, `DatasetLayout`, `Text2SqlExample`, `serialize_schema`, `build_prompt` / `build_prompts`, `Collator`, `IGNORE_INDEX`, `OptimizerConfig`, `OptimizerBundle.coverage`, `TrainConfig`, `TrainReport`, `set_seed`, `train`, `generate_sql`, `normalize_sql`, `exact_match_by_hardness`, `execution_accuracy`, `nonlinearity_index`, `model_nonlinearity`, `ExperimentConfig.run_id`, `build_result_card`, `write_result_card`, `measure_latency`, `merge_adapters`.

---

## Порядок исполнения и риски

Задачи 1–18 не требуют GPU и делаются на Mac. Задача 19 — единственная, которой нужна GTX 1080 Ti, и она же самая длинная по календарю: обучение идёт фоном, пока пишется текст.

Соответствие неделям из спецификации:

- **Неделя 1:** задачи 1–9.
- **Неделя 2:** задачи 10–16, затем шаги 1–5 задачи 19 (замер времени, проверка на расхождение, запуск матрицы).
- **Неделя 3:** матрица доигрывает фоном; параллельно задачи 17–18 и шаг 6 задачи 19.
- **Неделя 4:** шаги 7–10 задачи 19, анализ, запас под замечания преподавателя.

Риски и реакции — по таблице спецификации, без изменений. Единственное дополнение: **проверку `sm_61` (Задача 1, шаг 7) надо выполнить на машине с картой в первый же день**, а не на неделе 2. Она способна обнулить весь план, и обнаружить это в конце месяца нельзя.
