# Данные: схема БД и загрузчики Spider/PAUQ

**Модули:** `src/kanlora/data/schema.py`, `src/kanlora/data/loaders.py`,
`src/kanlora/data/spider.py`, `src/kanlora/data/pauq.py`
**Тесты:** `tests/data/test_schema.py`, `tests/data/test_loaders.py`

## Зачем один загрузчик на оба набора

Spider и PAUQ хранятся в одном формате JSON (список объектов
`{db_id, question, query}` для примеров, `tables.json` формата Spider для
схем) и различаются только именами файлов разбиений. Поэтому вся логика
чтения и предобработки — одна функция `load_examples`, а `spider.py` и
`pauq.py` содержат только раскладку имён файлов. Второй, независимый
загрузчик означал бы второе место, где преобразование примера в
`Text2SqlExample` может незаметно разойтись между наборами, — а именно это
расхождение предобработки и обесценило бы сравнение Spider с PAUQ,
ради которого набор PAUQ вообще привлечён.

## Сериализация схемы (`schema.py`)

- `Table(name: str, columns: tuple[str, ...])`, `DatabaseSchema(db_id: str, tables: tuple[Table, ...])`
  — замороженные (`frozen=True`) датаклассы, поэтому хешируемы и не могут
  быть случайно изменены в середине прогона.
- `load_schemas(tables_json: Path) -> dict[str, DatabaseSchema]` — читает
  `tables.json` формата Spider. Столбец `*` (служебный, `table_index == -1`,
  относится ко всей базе, а не к конкретной таблице) отбрасывается при
  разборе — он не несёт информации для промпта. Порядок таблиц и столбцов
  внутри таблицы берётся из порядка в исходном JSON и не зависит от порядка
  обхода промежуточных словарей.
- `serialize_schema(schema: DatabaseSchema) -> str` — одна строка вида
  `table: col, col | table: col`. Формат зафиксирован и одинаков для всех
  прогонов всех трёх методов: он идёт прямо в промпт модели (реализация
  промпта — задача 9, ещё не сделана), и его изменение задним числом
  обесценило бы уже полученные результаты.

## Раскладки наборов (`spider.py`, `pauq.py`)

`DatasetLayout(train_file: str, eval_file: str, tables_file: str, database_dir: str)`
— замороженный датакласс с именами файлов одного набора. Объявлен в
`spider.py`, а не в `loaders.py`: `loaders.py` использует готовые раскладки
`SPIDER_LAYOUT`/`PAUQ_LAYOUT` для реестра `LAYOUTS`, а `pauq.py` использует
сам класс `DatasetLayout` — объявление датакласса в `loaders.py` создало бы
цикл импортов (`loaders → spider/pauq → loaders`). Решение — держать
`DatasetLayout` там, где он нужен раньше всего по смыслу (раскладка Spider
как опорная), и импортировать его оттуда в `pauq.py`.

```python
SPIDER_LAYOUT = DatasetLayout(
    train_file="train_spider.json", eval_file="dev.json",
    tables_file="tables.json", database_dir="database",
)
PAUQ_LAYOUT = DatasetLayout(
    train_file="pauq_xsp_train.json", eval_file="pauq_xsp_test.json",
    tables_file="tables.json", database_dir="database",
)
```

Для PAUQ используется только разбиение по базам данных `pauq_xsp` — второе
разбиение репозитория `ai-spiderweb/pauq` не подключается, чтобы не плодить
вариантов методологии. Если фактические имена файлов после распаковки
архива PAUQ отличаются от констант выше, их нужно поправить здесь
синхронно с `DATASET_FILES` в `scripts/check_environment.py` (см.
`docs/environment-check.md`).

## Чтение примеров (`loaders.py`)

`Text2SqlExample(db_id: str, question: str, query: str)` — замороженный
датакласс одного примера.

`load_examples(root: Path, layout: DatasetLayout, split: Literal["train", "eval"]) -> list[Text2SqlExample]`:

- путь к файлу — `root / layout.train_file` или `root / layout.eval_file`
  в зависимости от `split`; неизвестное значение `split` (не `"train"` и не
  `"eval"`) роняет `KeyError` при обращении к внутреннему словарю
  `_split_file`, а не проходит молча;
- отсутствующий файл роняет `FileNotFoundError` с именем файла в тексте
  ошибки (`match="train_spider.json"` и подобные проверяются в тестах) —
  так опечатка в пути к данным обнаруживается сразу, а не после часа
  обучения на пустой выборке;
- `question` обрезается по краям (`str.strip()`), `query` нормализуется
  схлопыванием любых внутренних пробелов/переносов строк в одиночный
  пробел (`" ".join(query.split())`) — исходные файлы Spider/PAUQ не
  гарантируют единообразного форматирования SQL, а несогласованные пробелы
  в целевой строке иначе просачивались бы в маску функции потерь при
  токенизации (задача 9).

`load_dataset_schemas(root: Path, layout: DatasetLayout) -> dict[str, DatabaseSchema]`
— тонкая обёртка над `schema.load_schemas(root / layout.tables_file)`;
существует, чтобы вызывающий код не собирал путь к `tables.json` вручную и
не дублировал раскладку.

`database_path(root: Path, layout: DatasetLayout, db_id: str) -> Path` —
`root / layout.database_dir / db_id / f"{db_id}.sqlite"`. Единая точка,
которой будет пользоваться Execution Accuracy (задача 14, ещё не
реализована), чтобы путь к базе не собирался наново в каждом месте, где он
нужен.

`LAYOUTS: dict[str, DatasetLayout] = {"spider": SPIDER_LAYOUT, "pauq": PAUQ_LAYOUT}`
— реестр по имени набора из конфигурации прогона.

## Воспроизводимое урезание выборки (`subsample`)

```python
def subsample(examples: list[Text2SqlExample], size: int, seed: int) -> list[Text2SqlExample]
```

Если `size >= len(examples)`, возвращается копия всей выборки без
изменений. Иначе выбирается `size` индексов через
`random.Random(seed).sample(...)`, индексы сортируются, и результат
собирается в исходном порядке файла.

Важное разграничение: `seed` здесь — **зерно выборки**, отдельное от
**зерна обучения** (`0, 1, 2` по методологии проекта). Обучающая выборка
урезается до фиксированного размера (2500 примеров по глобальным
ограничениям плана) один раз, одним и тем же `seed`, и это урезание обязано
быть одинаковым для LoRA, DoRA и KAN-LoRA и для всех трёх зёрен обучения —
иначе разница в качестве между методами частично объяснялась бы тем, что
они обучались на разных примерах, а не разными адаптерами, и сравнение
потеряло бы доказательную силу. Поэтому `subsample` детерминирована по
`(examples, size, seed)` и не зависит ни от какого глобального генератора
случайных чисел.

## Проверенные инварианты

- `load_schemas` группирует столбцы по таблицам в порядке исходного файла
  и отбрасывает служебный столбец `*` (`test_columns_are_grouped_by_table`,
  `test_service_column_is_dropped`).
- Формат `serialize_schema` закреплён буквально и детерминирован
  относительно повторной загрузки того же файла
  (`test_serialized_format_is_frozen`, `test_serialization_is_deterministic`).
- `DatabaseSchema` хешируем (`test_schema_is_hashable`) — защита от случайной
  правки схемы в середине прогона на уровне типов, а не только по конвенции.
- `load_examples` читает оба разбиения (`test_train_split_is_read`,
  `test_eval_split_is_read`), падает `KeyError` на неизвестном разбиении
  (`test_unknown_split_is_rejected`) и `FileNotFoundError` с именем файла в
  сообщении на отсутствующем файле (`test_missing_file_names_the_path`).
- У каждого примера есть схема соответствующей базы
  (`test_every_example_has_a_schema`) — пример без схемы дал бы пустой
  промпт и тихо испортил бы метрику ниже по пайплайну.
- `SPIDER_LAYOUT` и `PAUQ_LAYOUT` — раскладки одного типа `DatasetLayout`,
  различающиеся только именами файлов, оба зарегистрированы в `LAYOUTS` под
  своими именами (`test_layouts_are_registered_under_dataset_names`,
  `test_layouts_differ_only_in_file_names`).
- `subsample` детерминирована по `seed`
  (`test_subsample_is_reproducible`), разные `seed` дают разные подвыборки
  (`test_subsample_does_not_depend_on_the_training_seed`), а запрошенный
  размер, превышающий выборку, возвращает выборку целиком
  (`test_subsample_keeps_everything_when_size_exceeds_the_set`).

## Не реализовано

- Промпт, токенизация и маска функции потерь по SQL (`data/collate.py`,
  задача 9 плана) — соединяют `Text2SqlExample` и `DatabaseSchema` в вход
  модели; этот модуль ещё не написан.
- Само чтение баз данных SQLite (для Execution Accuracy, задача 14) —
  `database_path` только строит путь, ничего не открывает.

Эти пункты не описываются подробнее здесь, чтобы не документировать
несуществующее поведение; их интерфейсы зафиксированы в
`docs/superpowers/plans/2026-09-09-kan-lora-text2sql.md`.
