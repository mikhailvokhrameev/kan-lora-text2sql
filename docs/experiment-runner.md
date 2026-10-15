# Конфигурация, карточка результата и сквозной прогон (`config.py`, `results.py`, `run_experiment.py`)

**Модули:** `src/kanlora/config.py`, `src/kanlora/results.py`,
`scripts/run_experiment.py`
**Конфигурации:** `configs/base.yaml`, `configs/smoke.yaml`
**Тесты:** `tests/test_config.py`, `tests/test_results.py`

## Зачем один скрипт на смоук-тест и на полный прогон

Требование методологии: **одна и та же команда**, различающаяся только
конфигурационным файлом и ключами командной строки, запускает и проверку на
200 примерах, и полный прогон на 2500. Отдельного «настоящего» пути запуска
нет: если бы смоук-тест и боевой прогон шли разным кодом, проверенным
оказался бы не тот код, который считает результаты.

Единица поставки — не число в блокноте, а JSON-карточка результата,
лежащая в `results/` вместе с git-историей проекта. По ней восстанавливается
весь прогон: конфигурация, версии библиотек, коммит, метрики, пиковая
видеопамять, время обучения и статистика сплайнов. Таблицы и графики
(задача 18 плана) собираются только из карточек и никогда не правятся
руками.

## `ExperimentConfig` (`config.py`)

```python
@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str
    data_root: str
    method: str
    dataset: str
    seed: int
    train_subset: int
    subsample_seed: int
    validation_size: int
    max_length: int
    eval_limit: int | None
    eval_batch_size: int
    max_new_tokens: int
    adapter: AdapterConfig
    train: TrainConfig
    optimizer: OptimizerConfig
```

Заморожен целиком, включая вложенные `AdapterConfig` (`adapters/base.py`),
`TrainConfig` (`train/loop.py`) и `OptimizerConfig` (`train/optimizers.py`).
Неизменяемость — не защита от опечаток, а гарантия для карточки результата:
если бы конфигурацию можно было поменять после чтения, карточка описывала бы
не тот прогон, который реально состоялся.

`subsample_seed` отделён от `seed` намеренно: `seed` — это зерно обучения,
одно из трёх (`0, 1, 2`) для усреднения по трём прогонам одного метода;
`subsample_seed` фиксирует, **какие именно** 2500 примеров попадут в
обучающую выборку, и обязан быть одним и тем же для всех методов и всех трёх
зёрен обучения — иначе разница в качестве между методами частично
объяснялась бы разными обучающими данными, а не самим методом.

`validation_size` (200) — размер валидационной выборки из train вне обучающей
подвыборки, по тому же `subsample_seed` (`validation_subsample`,
`docs/data.md`). На ней цикл обучения считает функцию потерь на шаге 0 и
после каждой эпохи (`docs/train-loop.md`).

### `method` и базовая модель без адаптера (`none`)

Допустимые методы перечислены в `METHODS = ("none", "lora", "dora",
"kan_lora")` (`config.py`); любое другое значение `method` роняет
`ValueError` уже при создании `ExperimentConfig` (`__post_init__`), так что
опечатка в YAML или в `--method` не доходит до загрузки модели. Тот же
кортеж задаёт `choices` ключа `--method` в `scripts/run_experiment.py`.

`method: none` — базовая точка: Qwen2.5-Coder-Instruct без адаптера,
zero-shot. Адаптеры не внедряются, обучения нет, только оценка. Без этой
точки таблица не показывает, дало ли дообучение что-то вообще. Оценка идёт
тем же путём, что и у обученных методов: промпты строит `build_prompts`
(шаблон диалога токенизатора, `docs/collate.md`), генерирует `generate_sql`
с тем же правилом усечения промпта и тем же `max_new_tokens`, ответ
нормализует `normalize_sql`, Exact Match и Execution Accuracy считаются теми
же функциями. Отдельных настроек для базовой модели нет: она запускается с
той же конфигурацией (`configs/base.yaml` или `configs/smoke.yaml`), секции
`adapter`, `train` и `optimizer` при этом просто не используются.

Instruct-модель без дообучения обычно отвечает пояснением и SQL в
Markdown-блоке ` ```sql `; из такого ответа `normalize_sql` берёт содержимое
первого блока — правило протокола оценки, общее для всех методов
(`docs/generate.md`, «Извлечение SQL из Markdown-блока»).

### `run_id`

```python
@property
def run_id(self) -> str:
```

Имя прогона вида `{method}-{dataset}-{optimizer}-seed{seed}`, например
`kan_lora-pauq-adamw-seed2`. Используется как имя файла карточки результата
и обязано различать все ячейки матрицы экспериментов: 3 метода × 2 набора ×
3 зерна × (обычно один оптимизатор, но `optimizer.name` в имени на случай
прогонов с `Muon`) = как минимум 18 разных строк, без коллизий.

У базовой модели без адаптера имя `none-{dataset}-seed{seed}`, например
`none-spider-seed0`: оптимизатора у прогона без обучения нет, и
`learn_input_scale` имя не меняет. Генерация жадная, поэтому зерно на
результат базовой модели не влияет; одного прогона на набор достаточно.

Абляционный режим (`adapter.learn_input_scale is False`, только для
`kan_lora`, см. `docs/adapters.md`) добавляет суффикс `-no-input-scale` —
это ровно тот случай из «Отклонений от спецификации» плана: не «прогон без
`tanh`», а «`input_scale`, заморожен на 1.0», и он обязан быть отличим по
имени файла от обычного прогона `kan_lora`.

### `load_config` и `apply_overrides`

```python
def load_config(path: Path) -> ExperimentConfig
def apply_overrides(config, *, method=None, dataset=None, seed=None,
                     optimizer_name=None, learn_input_scale=None,
                     train_subset=None, eval_limit=None) -> ExperimentConfig
```

`load_config` читает YAML один в один в поля `ExperimentConfig`;
`train.seed` берётся из того же `seed`, что и `config.seed`, — в файле
конфигурации зерно обучения указывается один раз, а не дважды в двух
местах, которые могли бы разойтись.

`apply_overrides` накладывает ключи командной строки на уже загруженную
конфигурацию через `dataclasses.replace`. Каждый параметр — либо явное
значение, либо `None`, и `None` означает «не трогать», а не «сбросить в
пусто»: непереданный ключ командной строки не должен затирать значение из
файла нулём или дефолтом. `seed` при переопределении синхронно обновляет и
`config.seed`, и `config.train.seed` — по той же причине, по которой их
нельзя было развести на два независимых поля при чтении.

### `config_to_dict`

```python
def config_to_dict(config: ExperimentConfig) -> dict
```

`dataclasses.asdict` рекурсивно превращает конфигурацию (включая вложенные
`AdapterConfig`/`TrainConfig`/`OptimizerConfig`) в обычный словарь для
сериализации в карточку результата и обратно в YAML. Поле `train.seed`
удаляется из результата: оно дублирует `config.seed` и существует только
как внутреннее удобство `TrainConfig`, а не как самостоятельная настройка —
хранить его дважды в сериализованном виде значило бы завести два места, где
зерно обучения может разойтись при ручном редактировании файла.

## `configs/base.yaml` и `configs/smoke.yaml`

`base.yaml` — гиперпараметры из «Global Constraints» плана дословно: ранг 8,
alpha 16.0, сетка 5, порядок сплайна 3, скорость обучения `2.0e-4`,
обрезание градиента по норме 1.0, батч 1, накопление 8, длина
последовательности 768, 2 эпохи, обучающая выборка 2500 примеров.

`smoke.yaml` отличается от `base.yaml` **только размерами**:
`train_subset: 200`, `train.epochs: 1`, `eval_limit: 50`. Адаптер,
`max_length` и устройство прогона — те же самые. Если бы смоук-конфигурация
отличалась ещё и устройством (например, другим набором целевых проекций),
проверенным оказался бы не тот путь кода, который идёт в боевой прогон.

## Карточка результата (`results.py`)

### `collect_versions` и `git_commit`

```python
def collect_versions() -> dict[str, str]  # python, torch, transformers, cuda
def git_commit() -> str
```

Версии библиотек, которые способны незаметно повлиять на числа
(округление, поведение генерации, реализация оптимизатора), и хеш коммита,
которым был получен результат. `git_commit()` не бросает исключение вне
git-репозитория или без доступного `git` — возвращает строку `"неизвестен"`:
диагностика окружения важнее аварийного завершения уже состоявшегося
прогона.

### `build_result_card`

```python
def build_result_card(config, injection, train_report, metrics, spline_stats) -> dict
```

Собирает карточку из результатов уже выполненного прогона: `injection` —
словарь с `trainable_parameters`, `total_parameters`, `replaced` (из
`InjectionReport`, `adapters/inject.py`); `train_report` — `TrainReport`
(`train/loop.py`) с историей функции потерь, долей активаций в сетке за
каждую эпоху, пиковой видеопамятью, временем и покрытием оптимизаторов;
`metrics` — словарь с Exact Match (в том числе по сложности запроса) и
Execution Accuracy; `spline_stats` — сводка `model_nonlinearity`
(`docs/nonlinearity.md`) или `None` для линейных методов.

Для `method: none` вызывается с `train_report=None` и
`injection={"trainable_parameters": None, "total_parameters": <число
параметров модели>, "replaced": []}`. Тогда в карточке `null`: `optimizer`
(хотя замороженная `config` по-прежнему содержит секцию `optimizer` из
YAML), `trainable_parameters`, `train_seconds`, `peak_memory_bytes` (пик
видеопамяти — величина обучения, на оценке не замеряется),
`optimizer_coverage`, `step_losses`, `epoch_losses`, `validation_losses`,
`grad_norms`, `fraction_inside_grid`, `spline_stats`; `adapted_modules` равно 0,
`metrics.truncated_prompts` — `null` (обучающая выборка не кодируется).
`null` здесь значит «не применимо: не обучалась», а не ноль — по тому же
принципу, что и для Execution Accuracy ниже. Читатели карточек это
переносят: `scripts/make_tables.py` показывает прочерк
(`docs/make-tables.md`).

Версии данных и модели (лекции 1 и 3 MLOps). `git_commit` фиксирует код, но
`data/` не в git, а модель грузится с Hugging Face по имени. Поэтому карточка
несёт ещё два поля:

- `data_files` — SHA-256 трёх файлов набора (`train_file`, `eval_file`,
  `tables_file` раскладки), посчитанный `file_hashes(root, names)` из
  `kanlora/results.py`; отсутствующий файл даёт `null`. Изменённый или другой
  файл Spider/PAUQ виден по расхождению хэша между карточками.
- `model_revision` — коммит репозитория модели на Hugging Face, который
  transformers записывает в `model.config._commit_hash` при загрузке
  (например, `ea3f2471…` для текущего кэша Qwen2.5-Coder-0.5B-Instruct);
  `null`, если его нет. Сравнимы только прогоны с одинаковой ревизией.

Хэши файлов баз SQLite не считаются: их сотни, и на EM они не влияют.

Карточка несёт из `train_report` оба ряда функции потерь: `step_losses` —
значение на каждом шаге оптимизатора (после накопления градиента), и
`epoch_losses` — среднее за эпоху. Третий ряд — `validation_losses`:
функция потерь на валидационной выборке на шаге 0 и после каждой эпохи.
Рядом — `grad_norms`: норма градиента до обрезания на каждом шаге
оптимизатора (`docs/train-loop.md`).
Без `step_losses` в карточке нельзя
отличить гладкое обучение от скачков внутри эпохи, усреднение до одного
числа на эпоху эту информацию стирает необратимо.

Принципиальный момент: `metrics["execution_accuracy"]` может быть `None`, а
не `0.0`. Отсутствие файлов баз данных PAUQ — заранее оговорённое в плане
отступление («Отклонения от спецификации»), и «метрику не мерили» обязано
быть отличимо в карточке от «метрика измерена и равна нулю». Смешать эти
два случая значило бы либо соврать про качество модели, либо потерять факт
того, что часть матрицы экспериментов не покрыта Execution Accuracy вообще.

### `write_result_card`

```python
def write_result_card(card: dict, directory: Path) -> Path
```

Пишет карточку в `{directory}/{card['run_id']}.json` с `ensure_ascii=False`
и `sort_keys=True`. `ensure_ascii=False` обязателен: карточки лежат в git и
читаются глазами, а русские подписи и сообщения об усечении промптов,
экранированные в `\uXXXX`, читать невозможно. `sort_keys=True` делает диффы
между повторными записями одной и той же карточки минимальными.

## `scripts/run_experiment.py`

Один прогон целиком, в порядке выполнения:

1. `apply_overrides(load_config(args.config), ...)` — конфигурация из файла
   плюс ключи командной строки (`--method`, `--dataset`, `--seed`,
   `--optimizer`, `--no-input-scale`, `--train-subset`, `--eval-limit`).
2. `device = "cuda"`, если доступна, иначе `"cpu"`; `set_seed(config.seed)`
   до чтения данных, чтобы `subsample` (см. `docs/data.md`) и последующая
   инициализация адаптеров были воспроизводимы по зерну прогона.
3. Схемы баз и примеры читаются `kanlora.data.loaders` по раскладке
   (`LAYOUTS[config.dataset]`); обучающая выборка урезается `subsample` до
   `config.train_subset` по `config.subsample_seed`, валидационная —
   `validation_subsample` до `config.validation_size` из оставшихся примеров
   train по тому же зерну (для `method: none` обе пусты); отложенная выборка
   урезается до `config.eval_limit`, если он задан (`None` — весь набор).
4. Модель грузится **явно в fp32** (`AutoModelForCausalLM.from_pretrained(...,
   dtype=torch.float32)`) — сравнение по видеопамяти между методами
   осмысленно только при одинаковой точности (см. «Global Constraints»
   плана), а не потому что fp32 быстрее или удобнее.
5–7. Шаги адаптации и обучения собраны в `adapt_and_train(config, model,
   tokenizer, train_examples, schemas, device, validation_examples=())`,
   которая возвращает
   `(injection, report, truncated)`:
   - `inject_adapters` подменяет семь целевых проекций и печатает число
     обучаемых параметров и число затронутых слоёв — первая проверка
     правдоподобия результата ещё до начала обучения;
   - `build_dataset` кодирует обучающую выборку и сообщает число усечённых
     промптов; при усечении печатается предупреждение — большая доля
     усечений искажает метрику сильнее, чем сам метод адаптации;
   - валидационные примеры кодируются тем же `build_dataset` (их усечения
     в счётчик `truncated` не входят);
   - `train()` (`docs/train-loop.md`) обучает модель, считает функцию потерь
     на валидации на шаге 0 и после каждой эпохи и возвращает `TrainReport`.

   Для `method: none` функция ничего из этого не делает: печатает число
   параметров модели и возвращает `({"trainable_parameters": None,
   "total_parameters": N, "replaced": []}, None, None)`. Обучающая выборка
   для базовой модели не читается (шаг 3 загружает только отложенную).
8. `generate_sql` (`docs/generate.md`) генерирует ответы на отложенной
   выборке, усекая промпты до `config.max_length - config.max_new_tokens`
   по тому же правилу, что и `build_dataset` при обучении; Exact Match считается официальной метрикой Spider
   (`exact_match_by_hardness`, `docs/exact_match.md`) по `tables.json`.
9. Execution Accuracy считается, если хотя бы для одной из уникальных
   `db_id` отложенной выборки на диске нашёлся файл базы
   (`count_available_databases(root, layout, unique_db_ids)`,
   `eval/execution.py`) — проверяется весь набор уникальных баз, а не
   только первый пример, потому что отложенная выборка обычно ссылается на
   много разных баз, и наличие файла для первой ничего не говорит об
   остальных. Если не найдено ни одной — печатается сообщение, и метрика
   остаётся `None` (то самое заранее оговорённое отступление для PAUQ без
   баз данных). Если найдены не все — метрика всё равно считается
   (`execute_query` уже трактует отсутствующий файл как промах), но
   печатается предупреждение с числом недостающих баз, чтобы частично
   измеренное значение не приняли за полностью чистое.
10. `model_nonlinearity(model) or None` — сводка по сплайнам для KAN-LoRA
    (`docs/nonlinearity.md`); пустой словарь `{}` для LoRA/DoRA превращается
    в `None`, чтобы в карточке линейного метода не лежало бессмысленное
    пустое поле, которое пришлось бы отдельно проверять при сборке таблиц.
11. `build_result_card` и `write_result_card` в каталог `--output`
    (по умолчанию `results/`), затем печать итоговых метрик, пиковой
    видеопамяти в ГиБ и времени обучения (для `method: none` — только
    метрик) и пути к карточке.

Запуск базовой точки:

```bash
python scripts/run_experiment.py --config configs/base.yaml --method none --dataset spider
python scripts/run_experiment.py --config configs/base.yaml --method none --dataset pauq
```

Сквозной путь скрипта модульными тестами не покрыт — он оркестрирует уже протестированные
по отдельности модули (данные, адаптеры, обучение, генерация, оценка,
анализ нелинейности) и требует настоящей модели `Qwen2.5-Coder-0.5B-Instruct`
и распакованных наборов Spider/PAUQ на диске (`scripts/check_environment.py`,
`docs/environment-check.md`). Верификация — сквозной прогон трёх методов на
`configs/smoke.yaml` (200 примеров); получившиеся карточки коммитятся в
`results/` (`lora-spider-adamw-seed0.json`, `dora-spider-adamw-seed0.json`,
`kan_lora-spider-adamw-seed0.json`), а не удаляются после проверки — это
единственный источник чисел, которые Таблица 1 записки
(`rpz/РПЗ_Вохрамеев_ИУ6-75Б.docx`) приводит для контрольного прогона, и
прослеживаемость этих чисел до карточки в git важнее, чем чистота `results/`
от проверочных прогонов.

Верификация проведена: все три метода на `configs/smoke.yaml` (набор
Spider, 200 обучающих примеров, 50 отложенных) дали ненулевые Exact Match и
Execution Accuracy без единой ошибки на всём пути обучение → генерация →
оценка → карточка результата — LoRA (EM 0.52, EA 0.62, 3.3 мин на CPU), DoRA
(EM 0.52, EA 0.64, 9.9 мин — дороже за счёт вычисления `effective_weight()`
на каждом шаге, см. `docs/adapters.md`), KAN-LoRA (EM 0.54, EA 0.64, 4.1 мин,
доля активаций в сетке 0.843 — сплайны не вырождены уже после одной эпохи на
200 примерах). Прогон выполнен на коде после исправления разводки
Muon/AdamW по роли параметра (`matrix_parameters()`, `docs/optimizers.md`),
но сам смок-тест использует `optimizer.name: adamw` — Muon-фикс на числа
этого прогона не влияет, только на код, который их посчитал.

## Проверенные инварианты

- Гиперпараметры `base.yaml` совпадают с зафиксированными в плане: ранг 8,
  alpha 16.0, сетка 5, порядок 3, `max_length` 768, 2 эпохи, обучающая
  выборка 2500, скорость обучения `2.0e-4`, обрезание градиента 1.0
  (`test_fixed_hyperparameters_match_the_plan`).
- Конфигурация неизменяема — присваивание полю после загрузки бросает
  исключение (`test_configuration_is_immutable`).
- `apply_overrides` создаёт новую конфигурацию, не трогая исходную
  (`test_overrides_produce_a_new_configuration`), а непереданный (`None`)
  параметр не затирает значение из файла
  (`test_override_of_none_is_ignored`); переопределение `seed` обновляет и
  `config.seed`, и `config.train.seed` синхронно
  (`test_train_seed_follows_the_run_seed`).
- `run_id` различает метод, набор, оптимизатор, зерно и абляционный режим
  (`test_run_id_encodes_the_whole_experiment_cell`).
- `none` входит в `METHODS`, `run_id` базовой модели —
  `none-{dataset}-seed{seed}` без оптимизатора и без суффикса абляции
  (`test_baseline_method_none_is_a_valid_method`,
  `test_baseline_run_id_does_not_name_an_optimizer`); неизвестный метод
  роняет `ValueError` (`test_unknown_method_is_rejected`).
- Карточка базовой модели пишет `null` в `optimizer` и во все поля обучения,
  `adapted_modules` — 0 (`test_baseline_card_leaves_training_fields_null`).
- `--method none` принимается командной строкой, а `adapt_and_train` для
  него не вызывает `inject_adapters`, `build_dataset` и `train` и не меняет
  веса модели (`tests/test_run_experiment.py`: заглушки этих функций
  падают при вызове, проверка на `tiny_causal_lm` без сети).
- Конфигурация переживает сериализацию в YAML и обратно без потерь
  (`test_serialization_round_trips`).
- `smoke.yaml` отличается от `base.yaml` только размерами выборки и числом
  эпох, а не устройством адаптера или длиной последовательности
  (`test_smoke_configuration_is_small_but_shaped_like_the_real_one`).
- Карточка результата несёт все поля, необходимые для восстановления
  прогона и сборки таблиц (`test_card_carries_everything_needed_to_reproduce`).
- Версии библиотек и хеш коммита действительно попадают в карточку
  (`test_versions_include_the_libraries_that_affect_numbers`,
  `test_git_commit_is_recorded`).
- Отсутствующая Execution Accuracy сериализуется как `null`, а не как `0.0`
  (`test_absent_execution_accuracy_is_null_not_zero`).
- Записанный файл — валидный JSON, названный по `run_id`, и не экранирует
  кириллицу (`test_written_card_is_valid_json_named_by_run`,
  `test_card_is_written_in_readable_utf8`).

## Не реализовано

Сборка итоговых таблиц и графиков из карточек результатов (задача 18), а
также замер фактического времени прогона и запуск полной матрицы
экспериментов на GTX 1080 Ti (задача 19) — ещё не сделаны. Замер задержки
генерации со слиянием адаптера и без (задача 17) реализован отдельно — см.
[`latency.md`](latency.md); карточка результата пока не несёт это поле,
так как оно не входит в матрицу метрик, зафиксированную в `results.py`.
Смоук-верификация (см. выше)
проведена на CPU и на наборе Spider; по PAUQ отдельно не прогонялась, но
использует тот же код и ту же раскладку `DatasetLayout` — специфичного для
PAUQ пути в `run_experiment.py` нет.
