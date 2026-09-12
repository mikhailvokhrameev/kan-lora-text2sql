# Execution Accuracy (`eval/execution.py`)

**Модуль:** `src/kanlora/eval/execution.py`
**Тесты:** `tests/eval/test_execution.py`

## Зачем метрика нужна отдельно от Exact Match

[`exact_match.md`](exact_match.md) сравнивает разобранные SQL-деревья:
запрос, написанный иначе, но синтаксически эквивалентный, засчитывается,
а запрос с другой структурой, но тем же результатом — нет (например,
`WHERE price > 15` и `WHERE NOT price <= 15`). Execution Accuracy закрывает
этот случай: запрос исполняется на настоящей базе `.sqlite`, и сравниваются
**результаты**, а не текст и не разобранное дерево. Это делает метрику
терпимее к переформулировкам и одновременно уязвимее к случайным
совпадениям результата у семантически разных запросов — поэтому в работе
приводятся обе метрики, а не одна вместо другой.

## `ExecutionOutcome` и `execute_query`

```python
@dataclass(frozen=True)
class ExecutionOutcome:
    rows: list[tuple] | None
    failed: bool

def execute_query(db_path: Path, query: str, timeout: float = 30.0) -> ExecutionOutcome
```

Любая ошибка — отсутствующий файл базы, синтаксическая ошибка в SQL,
таймаут — превращается в `ExecutionOutcome(None, True)`, а не в исключение.
Причина: модель регулярно генерирует неисполнимый текст, и это ожидаемая,
частая часть измерения, а не аварийная ситуация, которая должна остановить
оценку всего прогона (тот же принцип, что и у обработки ошибок разбора
в `exact_match_by_hardness`).

Подключение открывается в режиме только для чтения (`mode=ro` в URI
`sqlite3.connect`), поэтому сгенерированный моделью SQL не может изменить
файл базы данных на диске, даже если это `UPDATE`/`DELETE`/`DROP`.
`text_factory` декодирует текстовые столбцы в UTF-8 с заменой
недекодируемых байт, а не падает на них — базы PAUQ русскоязычные, и
байтовый мусор в отдельной строке не должен обрушивать исполнение всего
запроса.

## `results_match` и роль `ORDER BY`

```python
def results_match(gold: ExecutionOutcome, predicted: ExecutionOutcome, order_matters: bool) -> bool
```

Если хотя бы одно исполнение помечено `failed`, совпадения нет — сравнивать
`None` с чем бы то ни было бессмысленно, и промах гарантированно не должен
засчитываться как успех. Иначе: при `order_matters=True` строки сравниваются
позиционно (`gold.rows == predicted.rows`), при `order_matters=False` —
как мультимножества через сортировку `repr` каждой строки. Разница
принципиальна: SQL без `ORDER BY` не гарантирует никакого порядка строк, и
требование его совпадения занижало бы метрику по признаку, не имеющему
отношения к правильности запроса.

## `execution_accuracy`

```python
def execution_accuracy(
    gold: list[str], predicted: list[str], db_ids: list[str], root: Path, layout: DatasetLayout,
) -> float
```

Для каждой тройки (эталон, предсказание, `db_id`) путь к файлу базы строится
через `database_path(root, layout, db_id)` из `kanlora.data.loaders`
(`docs/data.md`) — та же раскладка, что используют загрузчики Spider и PAUQ,
поэтому метрика не заводит собственное представление о структуре каталогов
набора. Требование к `ORDER BY` берётся из **эталонного** запроса
(`"order by" in gold_query.lower()`), а не из предсказанного: предсказание
может как включать, так и опускать `ORDER BY` по своей структуре, и это не
должно определять, что именно с ним сравнивается.

Итоговая оценка — доля совпавших пар. Несовпадение длин `gold`/`predicted`/
`db_ids` — `ValueError`, а не тихое усечение по меньшей длине, по тому же
принципу, что и в `exact_match`/`exact_match_by_hardness`.

## `count_available_databases`

```python
def count_available_databases(root: Path, layout: DatasetLayout, db_ids: Iterable[str]) -> int
```

Считает, для скольких **уникальных** `db_id` на диске нашёлся файл базы —
дубликаты в `db_ids` схлопываются через `set(db_ids)` до подсчёта.
Существует для `run_experiment.py` (`docs/experiment-runner.md`): решение,
считать ли Execution Accuracy вообще, раньше принималось по одному
произвольному `db_id` отложенной выборки (`database_path(root, layout,
db_ids[0]).is_file()`), хотя выборка обычно ссылается на много разных баз —
наличие файла для первого примера ничего не говорит об остальных.

## Проверенные инварианты

- Успешное исполнение возвращает строки, `failed=False`
  (`test_executes_and_returns_rows`).
- Синтаксически некорректный запрос — промах без исключения
  (`test_broken_query_is_reported_not_raised`).
- Отсутствующий файл базы — промах без исключения
  (`test_missing_database_is_reported_not_raised`).
- Без `ORDER BY` в эталоне порядок строк игнорируется; с `order_matters=True`
  тот же порядок строк обязателен (`test_row_order_is_ignored_without_order_by`).
- Любая из двух неудачных попыток исполнения — это несовпадение, включая
  случай, когда обе стороны сломаны (`test_failed_execution_never_matches`).
- Разный текст запроса с одинаковым результатом засчитывается совпадением
  (`test_accuracy_counts_equivalent_rewrites_as_correct`).
- `ORDER BY` в эталоне делает разный порядок результата несовпадением
  (`test_accuracy_respects_order_by`).
- Несовпадение длин `gold`/`predicted`/`db_ids` — `ValueError`
  (`test_mismatched_lengths_are_rejected`).
- `count_available_databases` — `0` для пустого списка `db_ids`
  (`test_count_available_databases_is_zero_for_empty_input`), считает только
  реально присутствующие файлы, дубликаты `db_id` не завышают счёт
  (`test_count_available_databases_counts_present_files_only`), и `0`, если
  не найдено ни одной базы (`test_count_available_databases_is_zero_when_none_found`).

## Что известно отступление спецификации

Если на машине отсутствуют файлы баз данных PAUQ (см. Задачу 1 плана,
[`environment-check.md`](environment-check.md)), Execution Accuracy для PAUQ не считается:
для этого набора остаётся только Exact Match, а поле `execution_accuracy`
в карточке результата — `null`. Сам модуль `execution.py` этого не решает
и не знает: отсутствие базы он отражает как `failed=True` для каждого
запроса, что дало бы `execution_accuracy == 0.0`, а не `null` — различие
между «метрика не считалась» и «метрика равна нулю» делается на уровне
сборки карточки результата — см. [`experiment-runner.md`](experiment-runner.md).

Если найдены не все уникальные базы (частичная выкладка PAUQ), метрика всё
равно считается — `execute_query` уже трактует отсутствующий файл как
промах, что корректно занижает оценку, — но `run_experiment.py` печатает
предупреждение с числом недостающих баз, чтобы частично измеренное
значение не приняли за полностью чистое.

## Смежные модули

Мера выученной нелинейности — см. [`nonlinearity.md`](nonlinearity.md);
конфигурация и карточка результата прогона — см.
[`experiment-runner.md`](experiment-runner.md); замер задержки генерации
со слиянием адаптера и без — см. [`latency.md`](latency.md).
