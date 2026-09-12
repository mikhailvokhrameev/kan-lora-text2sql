# Exact Match по официальной методике Spider (`eval/exact_match.py`)

**Модуль:** `src/kanlora/eval/exact_match.py`
**Тесты:** `tests/eval/test_exact_match.py`
**Вендоренный код:** `third_party/spider_eval/` (`evaluation.py`, `process_sql.py`
из `taoyds/spider`, коммит `b7b5b8c890cd30e35427348bb9eb8c6d1350ca7c`, без изменений)

## Зачем метрика вендорится, а не пишется своя

В Spider Exact Match — это покомпонентное сравнение разобранного запроса
(SELECT, WHERE, GROUP BY и так далее сравниваются как множества значений),
а не равенство строк или собственный SQL-парсер. Собственная реализация дала
бы числа, несравнимые с опубликованными в литературе результатами — а
именно такое сравнение и есть цель работы. Поэтому `evaluation.py` и
`process_sql.py` взяты из официального репозитория без единой правки;
`src/kanlora/eval/exact_match.py` — это только обёртка вокруг них под
интерфейс проекта.

## Скрытая зависимость на `nltk` (`punkt_tab`)

Вендоренный `process_sql.py` токенизирует сам текст SQL-запроса через
`nltk.word_tokenize` (а не только разбирает его как SQL) — это часть
оригинального кода Spider, а не что-то, что можно убрать при вендоринге
«без изменений». `word_tokenize` требует скачанных данных `punkt_tab`, иначе
падает `LookupError`.

Эта зависимость долгое время не была объявлена в `pyproject.toml`: тесты
случайно проходили, потому что `nltk` и данные `punkt_tab` уже были
установлены в интерпретаторе, которым по факту запускались тесты. Только
после перехода на честно изолированное окружение (`uv sync` в чистый `.venv`)
это вскрылось — `ModuleNotFoundError: No module named 'nltk'`. Теперь `nltk`
явно в зависимостях, а `scripts/check_environment.py` (`check_nltk_punkt`,
`docs/environment-check.md`) проверяет наличие данных `punkt_tab` до начала
эксперимента, а не посреди оценки Exact Match после уже состоявшегося
обучения. Данные скачиваются один раз командой:

```bash
python -m nltk.downloader punkt_tab
```

## Почему обёртка нужна вообще

Официальный `evaluation.py` написан как программа командной строки:
принимает пути к файлам с запросами и каталог с `.sqlite`-базами, и вместо
возврата числа печатает таблицу в stdout. Обёртка:

- принимает списки запросов и `db_id` в памяти (то, что уже есть после
  `generate_sql()`, `docs/generate.md`), а не пути к файлам на диске;
- строит схему каждой базы из `tables.json`, а не подключаясь к `.sqlite`
  напрямую — расположение файлов баз данных может отличаться между машиной
  разработки и GPU-машиной, а `tables.json` лежит в git вместе с набором;
- возвращает `float` или `dict[str, float]` вместо печати в stdout.

## Плоский импорт вендоренного кода

`evaluation.py` в оригинале — не модуль пакета, а самостоятельный скрипт: он
делает `from process_sql import ...` (без указания пакета). Такой импорт
работает только тогда, когда каталог с обоими файлами лежит прямо в
`sys.path`, а не когда он импортируется как подпакет (`spider_eval.
evaluation`). Поэтому обёртка добавляет в `sys.path` именно
`third_party/spider_eval/`, а не `third_party/`, и импортирует `evaluation`
и `process_sql` как модули верхнего уровня:

```python
_SPIDER_EVAL = Path(__file__).resolve().parents[3] / "third_party" / "spider_eval"
sys.path.insert(0, str(_SPIDER_EVAL))

from evaluation import Evaluator, build_foreign_key_map_from_json, ...
from process_sql import Schema, get_sql
```

Импорт сделан внутри функции (`exact_match_by_hardness`), а не на уровне
модуля: тяжёлые вычисления парсинга SQL не нужны до первого вызова, а
изменение `sys.path` побочным эффектом импорта `kanlora.eval.exact_match`
было бы неожиданным для остального кода.

## Схема базы из `tables.json`

`process_sql.get_schema_from_json` из вендоренного кода читает не тот
формат JSON, что `tables.json` Spider — он рассчитан на другой вспомогательный
файл (`{"table": ..., "col_data": [...]}` на запись) и не подходит. Официальный
`evaluate()` вместо этого строит схему подключением к `.sqlite`
(`Schema(get_schema(db))`), а `.sqlite`-путей у обёртки нет по интерфейсу.

`_load_schemas` строит тот же по форме словарь
(`{имя_таблицы_в_нижнем_регистре: [имена_столбцов_в_нижнем_регистре]}`),
что и `get_schema`, — но не разбирает `tables.json` заново, а вызывает
`kanlora.data.schema.load_schemas` (`docs/data.md`) и приводит уже
готовый `DatabaseSchema` к нижнему регистру через `_schema_dict_from_entry`.
До этого обёртка сама читала поля `table_names_original` /
`column_names_original` записи `tables.json` — тот же разбор, что и в
`data/schema.py`, но независимо реализованный, с риском незаметно
разойтись при изменении одного из двух мест. Отбрасывание служебного
столбца `*` (индекс таблицы -1) теперь происходит только внутри
`load_schemas`; `exact_match.py` про этот индекс больше не знает — он
получает уже отфильтрованный `DatabaseSchema.tables` и только меняет
регистр имён под ожидания `Schema` из `process_sql.py`.

## Учёт внешних ключей (`rebuild_sql_col`)

Официальный `evaluate()` перед сравнением приводит столбцы, связанные внешним
ключом, к общему представлению (`rebuild_sql_val` + `rebuild_sql_col` с картой
`build_foreign_key_map_from_json`) — иначе `JOIN` по внешнему ключу, где
эталон и предсказание ссылаются на семантически одну и ту же пару столбцов
из разных таблиц, засчитывался бы как расхождение. Обёртка повторяет этот шаг
дословно для эталонного и предсказанного запроса, поэтому числа воспроизводят
официальную методику целиком, а не только верхнеуровневое сравнение SQL-дерева.

Запись `tables.json` обязана содержать поле `foreign_keys` (список пар индексов
столбцов; пустой список для баз без внешних ключей) — этого требует
`build_foreign_key_map_from_json`. Тестовая фикстура
`tests/data/fixtures/spider_mini/tables.json` дополнена этим полем при
реализации задачи; `src/kanlora/data/schema.py` (`docs/data.md`) это поле не
читает и от его добавления не зависит.

## `exact_match_by_hardness` и `exact_match`

```python
def exact_match_by_hardness(
    gold: list[str], predicted: list[str], db_ids: list[str], tables_json: Path,
) -> dict[str, float]

def exact_match(
    gold: list[str], predicted: list[str], db_ids: list[str], tables_json: Path,
) -> float
```

Для каждой тройки (эталон, предсказание, `db_id`):

1. Эталонный запрос разбирается `get_sql`, ошибка на эталоне не перехватывается
   — эталон обязан быть валидным SQL, и падение здесь означает ошибку в самих
   данных, а не в модели.
2. Сложность запроса (`easy` / `medium` / `hard` / `extra`) снимается с
   разобранного эталона через `Evaluator.eval_hardness`.
3. Предсказание разбирается и сравнивается тем же путём (`rebuild_sql_val` +
   `rebuild_sql_col` + `Evaluator.eval_exact_match`); любое исключение при
   разборе или сравнении предсказания превращается в `matched = False`, а не
   пробрасывается наружу — модель регулярно генерирует не-SQL текст, и это
   ожидаемый промах, а не повод остановить оценку всего прогона.
4. Совпадение засчитывается одновременно в счётчик своего уровня сложности и
   в `"all"`.

`exact_match()` — то же самое, но возвращает только `breakdown["all"]`, когда
разбивка по сложности не нужна вызывающему коду.

Обе функции требуют одинаковой длины `gold`, `predicted` и `db_ids` и бросают
`ValueError` при несовпадении — рассинхронизация длин почти всегда означает
ошибку сборки списков на стороне вызывающего кода, а не то, что стоит
молча обрезать по меньшей длине.

## Проверенные инварианты

- Идентичные запросы совпадают (`test_identical_queries_match`).
- Совпадение — по разобранному запросу, а не по строке: регистр ключевых слов
  и произвольные пробелы не влияют (`test_case_and_spacing_do_not_matter`).
- Разные по смыслу запросы не совпадают
  (`test_different_queries_do_not_match`).
- Неразбираемое предсказание — это промах без исключения на весь прогон
  (`test_unparsable_prediction_counts_as_a_miss`).
- Итоговая оценка — доля совпадений по всем примерам
  (`test_score_is_the_share_of_matches`).
- Несовпадение длин `gold`/`predicted`/`db_ids` — это `ValueError`, а не
  тихое усечение (`test_mismatched_lengths_are_rejected`).
- Разбивка по сложности всегда содержит все пять уровней официального
  сценария: `easy`, `medium`, `hard`, `extra`, `all`
  (`test_hardness_breakdown_covers_all_levels`).
- Словарь схемы, построенный из `data.schema.load_schemas`, побайтово
  совпадает с тем, что раньше строил ручной разбор `tables.json` внутри
  этого модуля — нижний регистр имён, отброшенный столбец `*`
  (`test_schema_dict_matches_hand_built_lowercase_format`). Регрессия на
  переход к общему парсеру из `data/schema.py` (задача 10 плана
  code-review-fixes).

## Смежные модули

Execution Accuracy реализована отдельно — см. [`execution.md`](execution.md);
мера выученной нелинейности — см. [`nonlinearity.md`](nonlinearity.md);
конфигурация и карточка результата прогона — см.
[`experiment-runner.md`](experiment-runner.md); замер задержки генерации
со слиянием адаптера и без — см. [`latency.md`](latency.md).
