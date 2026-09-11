# Адаптеры: базис, интерфейс, LoRA, DoRA, KAN-LoRA, внедрение в модель

**Модули:** `src/kanlora/adapters/spline.py`, `src/kanlora/adapters/kan_layer.py`,
`src/kanlora/adapters/base.py`, `src/kanlora/adapters/lora.py`, `src/kanlora/adapters/dora.py`,
`src/kanlora/adapters/kan_lora.py`, `src/kanlora/adapters/inject.py`
**Тесты:** `tests/adapters/test_spline.py`, `tests/adapters/test_kan_layer.py`,
`tests/adapters/test_base.py`, `tests/adapters/test_lora.py`, `tests/adapters/test_dora.py`,
`tests/adapters/test_kan_lora.py`, `tests/adapters/test_inject.py`, `tests/conftest.py`

## Зачем общий интерфейс

Работа сравнивает три способа адаптации (LoRA, DoRA, KAN-LoRA) на одной и
той же модели, одних и тех же данных и по одной и той же процедуре оценки.
Если бы каждый метод внедрялся в модель своим кодом обучения и генерации,
сравнение начало бы мерить качество реализаций, а не свойства методов.
Поэтому все адаптеры реализуют один интерфейс `AdapterLinear` и
подставляются вместо `nn.Linear` единым внедрителем (`inject.py`, см.
раздел ниже).

## `AdapterConfig` (`base.py`)

Замороженный (`frozen=True`) датакласс гиперпараметров, общих для всех
трёх методов:

```python
@dataclass(frozen=True)
class AdapterConfig:
    rank: int = 8
    alpha: float = 16.0
    grid_size: int = 5
    spline_order: int = 3
    learn_input_scale: bool = True
```

Заморожен намеренно: по методологии проекта гиперпараметры фиксируются
один раз и не подбираются в процессе экспериментов, поэтому случайная
правка поля в середине серии прогонов невозможна на уровне типов.

`grid_size`, `spline_order` и `learn_input_scale` используются только
KAN-LoRA-адаптером (пока не реализован); для LoRA и DoRA они игнорируются
и хранятся в конфиге ради того, чтобы карточка результата содержала
одинаковый набор полей для всех прогонов независимо от метода.

## `AdapterLinear` (`base.py`)

Абстрактный базовый класс (`nn.Module, ABC`). Конструктор принимает уже
существующий `nn.Linear` и `AdapterConfig`, замораживает веса основы через
`freeze()` и вычисляет масштаб поправки:

```python
self.scaling = config.alpha / config.rank
```

Поля:

- `self.base: nn.Linear` — замороженная исходная линейная проекция.
- `self.config: AdapterConfig`.
- `self.scaling: float`.

Абстрактные члены, которые обязана реализовать каждая конкретная адаптация:

- `forward(x: Tensor) -> Tensor` — прямой проход с поправкой.
- `analytic_parameter_count() -> int` — число обучаемых параметров,
  посчитанное по формуле метода, независимо от фактической реализации.
  Существует ради теста: расхождение с фактическим счётчиком означает, что
  утверждение о равном бюджете параметров в итоговых таблицах неверно.
- `can_merge: bool` (свойство) — сливается ли поправка с весами основы
  после обучения в обычный `nn.Linear`.

Конкретные (не переопределяемые по умолчанию) члены:

- `trainable_parameter_count() -> int` — сумма `numel()` по параметрам с
  `requires_grad=True`. Так как `self.base` заморожен в конструкторе,
  веса основы в счётчик не попадают.
- `merge() -> nn.Linear` — по умолчанию бросает `NotImplementedError`;
  переопределяется теми методами, у которых `can_merge is True`.
- `extra_repr()` — читаемое представление для `print(model)`.

### `freeze(module: nn.Module) -> None`

Снимает `requires_grad` со всех параметров модуля. Используется в
конструкторе `AdapterLinear` для заморозки `self.base`.

## B-сплайновый базис (`spline.py`)

Чистая математика без обучаемых параметров — строительный блок для слоя
Колмогорова — Арнольда (`KANLayer`), который, в свою очередь, вставлен
между двумя низкоранговыми матрицами в `KANLoRALinear` (см. ниже).

- `num_basis_functions(grid_size, spline_order) -> int` — возвращает
  `grid_size + spline_order`, число базисных функций на сетке.
- `make_knots(grid_size, spline_order, lo=-1.0, hi=1.0, *, dtype=None, device=None) -> Tensor`
  — равномерный узловой вектор на отрезке `[lo, hi]`, продлённый на
  `spline_order` узлов за каждую границу. Продление необходимо, чтобы сумма
  базисных функций оставалась равной единице у самих краёв отрезка, а не
  только в его внутренней части.
- `bspline_basis(x, knots, spline_order) -> Tensor` — значения всех
  базисных функций в точках `x` по рекурсии Кокса — де Бура; форма
  результата — `(*x.shape, len(knots) - 1 - spline_order)`.
- `greville_abscissae(knots, spline_order) -> Tensor` — абсциссы Гревилля:
  коэффициенты, при подстановке которых в разложение по базису сплайн
  становится тождественной функцией (`sum_i greville_i * B_i(x) == x`
  точно, а не приближённо). На этом свойстве линейной точности B-сплайнов
  держится тождественная инициализация `KANLayer` и, как следствие,
  точное численное совпадение будущего KAN-LoRA с LoRA при нулевой
  нелинейности.

## Слой Колмогорова — Арнольда (`kan_layer.py`)

`KANLayer(in_features, out_features, *, grid_size=5, spline_order=3, grid_range=(-1.0, 1.0), device=None, dtype=None)`
— отображение `R^in -> R^out`, где на каждом ребре `(i -> j)` определена
одномерная функция

```
phi_ji(u) = w_base_ji * silu(u) + w_spline_ji * spline_ji(u)
```

а выход слоя — сумма по входным каналам, с добавленным обучаемым масштабом
входа `s_i`:

```
out_j = sum_i s_i * phi_ji(x_i / s_i)
```

Формулировка сохранена в том виде, в каком она задана в исходной работе о
KAN, за одним отклонением от текста спецификации проекта: вместо `tanh` на
входе сплайна используется обучаемый масштаб `input_scale`. Причина —
B-сплайновый базис определён на конечной сетке `grid_range`, и вне неё все
базисные функции обращаются в ноль: слой молча вырождается в LoRA с
неработающими параметрами. `tanh(x) != x`, поэтому с `tanh` тождественная
инициализация стала бы лишь приближённой; `input_scale` входит и выходит
из формулы симметрично, поэтому тождество остаётся точным. Абляционный
режим — не «без `tanh`», а `input_scale`, заморозенный на 1.0 (см. ниже
`learn_input_scale`).

### Параметры слоя

- `spline_coefficients`: `(out_features, in_features, grid_size + spline_order)`.
- `spline_scale`: `(out_features, in_features)`.
- `base_weight`: `(out_features, in_features)`.
- `input_scale`: `(in_features,)`.
- `knots` — буфер (не параметр), вычисляется в `float64` независимо от
  `dtype` самого слоя. Узлы задают абсциссы Гревилля, а через них — точность
  тождественной инициализации; вычисление в одинарной точности стоило бы
  порядка двух десятичных разрядов точности при проверке совпадения с LoRA.

### Тождественная инициализация (`reset_to_identity`)

Вызывается в конце конструктора:

1. `spline_coefficients` обнуляются, затем на диагональные рёбра
   (`min(in_features, out_features)` штук) записываются абсциссы Гревилля.
2. `spline_scale` заполняется единицами, `base_weight` обнуляется,
   `input_scale` заполняется единицами.

Итог: на диагональных рёбрах сплайн-часть в точности равна входу, базовая
ветвь (`silu`) не участвует, недиагональные рёбра дают ноль. При
`in_features == out_features` слой в момент создания — тождественное
отображение.

### `forward(x) -> Tensor`

```python
scaled = x / self.input_scale
basis = bspline_basis(scaled, self.knots, self.spline_order)
spline = einsum("...ib,oib->...oi", basis, self.spline_coefficients)
base = silu(scaled).unsqueeze(-2) * self.base_weight
edges = base + self.spline_scale * spline
return (edges * self.input_scale).sum(dim=-1)
```

### `fraction_inside_grid(x) -> Tensor`

Диагностика вырождения: доля элементов `x / input_scale`, попавших в
`grid_range`. Значение, ощутимо меньшее 1.0, означает, что заметная часть
входов не задействует сплайн-часть слоя (базис вне сетки равен нулю) —
это самая частая скрытая ошибка в реализациях KAN-адаптеров, поэтому
величина рассчитана как отдельный метод, снимаемый на каждом проходе, а
не отдельным диагностическим прогоном.

### `parameter_count() -> int`

Сумма `numel()` по всем параметрам слоя, без разбора на обучаемые и
замороженные (в отличие от `AdapterLinear.trainable_parameter_count`).

## LoRA (`lora.py`)

`LoRALinear(AdapterLinear)` — опорный метод адаптации, реализован
максимально прямолинейно, без дополнительных модификаций формулы:

```
delta_W * x = (alpha / r) * B (A x)
```

где `A: (rank, in_features)`, `B: (out_features, rank)`.

### Параметры

- `lora_a: nn.Parameter` формы `(rank, in_features)`, инициализируется
  `nn.init.kaiming_uniform_(a=sqrt(5))`.
- `lora_b: nn.Parameter` формы `(out_features, rank)`, инициализируется
  нулём.

Обнуляется ровно одна из двух матриц, а не обе: при `A = B = 0` градиент
по обеим матрицам тождественно нулевой и обучение не сдвинулось бы с
места. Ненулевая `A` при нулевой `B` даёт нулевую поправку на старте
(`delta = 0`) и ненулевой градиент по `A` уже на первом шаге.

### Методы

- `delta(x) -> Tensor` — возвращает `scaling * linear(linear(x, lora_a), lora_b)`,
  то есть `(alpha / r) * B (A x)`, вычисленный через
  `torch.nn.functional.linear` (эквивалентно матричному произведению, но
  без явного транспонирования).
- `forward(x) -> Tensor` — `self.base(x) + self.delta(x)`. При инициализации
  численно (не приближённо) совпадает с `self.base(x)`, поскольку `delta`
  тождественно равна нулю.
- `analytic_parameter_count() -> int` — `rank * (in_features + out_features)`.
- `can_merge` — всегда `True`.
- `merge() -> nn.Linear` — создаёт новый `nn.Linear` с
  `weight = base.weight + scaling * (lora_b @ lora_a)` и копией `bias`
  (если он есть). Исходный `self.base` не изменяется: слияние выполняется в
  новый объект, старый остаётся пригодным для повторного использования
  (в частности, для будущего DoRA-адаптера, который также оборачивает
  `nn.Linear` как самостоятельную основу).

Слияние (`merge`) существует потому, что LoRA — линейная поправка: после
обучения её можно поглотить в веса и не платить дополнительной задержкой
на инференсе. Это отличает её от будущего KAN-LoRA, где поправка зависит
от входа нелинейно и слияние в принципе невозможно (`can_merge = False`,
`merge()` бросает `NotImplementedError` по умолчанию из `AdapterLinear`).

## DoRA (`dora.py`)

`DoRALinear(AdapterLinear)` — второй опорный метод. Раскладывает вес
основы на **модуль** (скаляр на выходной канал) и **направление**
(матрица той же формы, что и вес), а направление адаптирует той же
низкоранговой поправкой, что и LoRA:

```
V = W + (alpha / r) * B A                    направление, форма (out, in)
W_eff = m * V / ||V||_строк                  эффективный вес, норма построчная
```

где `A: (rank, in_features)`, `B: (out_features, rank)`, `m: (out_features,)`
— обучаемый вектор модуля.

Замысел метода (за него DoRA отвечает на защите): при полной тонкой
настройке модуль и направление веса меняются по независимым траекториям, а
LoRA, добавляя поправку прямо к `W`, вынуждена менять их совместно.
Отдельный вектор `m` снимает эту связанность ценой `out_features`
дополнительных параметров.

### Параметры

- `lora_a: nn.Parameter` формы `(rank, in_features)` — как в LoRA,
  инициализация `kaiming_uniform_(a=sqrt(5))`.
- `lora_b: nn.Parameter` формы `(out_features, rank)` — инициализируется
  нулём, по той же причине, что и в LoRA (иначе градиент по обеим матрицам
  тождественно нулевой).
- `magnitude: nn.Parameter` формы `(out_features,)` — инициализируется
  **построчной нормой исходного веса** `base.weight.norm(dim=1)`, а не
  единицей и не нулём.

Инициализация `magnitude` нормой `base.weight` — не произвольный выбор, а
условие того же инварианта «адаптер стартует немодифицированным», что и у
LoRA: при `B = 0` направление `V` в точности равно `base.weight`, и если
`m = ||base.weight||_строк`, то `m * V / ||V|| == base.weight` точно.
Любая другая инициализация `magnitude` сдвинула бы стартовую точку DoRA
относительно LoRA и обесценила бы сравнение методов.

### Методы

- `effective_weight() -> Tensor` — вычисляет `W_eff` по формуле выше;
  вызывается на каждом прямом проходе, а не кэшируется, так как
  `lora_a`, `lora_b` и `magnitude` обучаемы.
- `forward(x) -> Tensor` — `linear(x, effective_weight(), base.bias)`.
- `analytic_parameter_count() -> int` —
  `rank * (in_features + out_features) + out_features` (LoRA плюс вектор
  модуля).
- `can_merge` — `True`: как и LoRA, DoRA линейна по `x` при фиксированных
  параметрах адаптера, поэтому после обучения `effective_weight()`
  поглощается в обычный `nn.Linear` и на инференсе не остаётся никакой
  дополнительной надстройки.
- `merge() -> nn.Linear` — новый `nn.Linear` с
  `weight = effective_weight()` и скопированным `bias`; исходный `base` не
  изменяется, аналогично `LoRALinear.merge()`.

### Сводимость к LoRA

При `m = ||W + s*BA||_строк` (норма самого направления, без отдельного
обучения модуля) нормировка в `effective_weight()` сокращается и DoRA
поэлементно совпадает с LoRA — это формальное содержание утверждения
«DoRA обобщает LoRA»: без независимого модуля остаётся ровно LoRA.
Проверено `test_reduces_to_lora_when_magnitude_matches_direction_norm`.

## KAN-LoRA (`kan_lora.py`)

`KANLoRALinear(AdapterLinear)` — исследуемый метод и центр всей работы.
Отличается от LoRA ровно одним элементом: между двумя низкоранговыми
матрицами вставлен `KANLayer`, дающий обучаемую нелинейность:

```
delta_W * x = (alpha / r) * B * phi(A x)
```

где `A: (rank, in_features)`, `B: (out_features, rank)`, а
`phi = KANLayer(rank, rank, grid_size, spline_order)`.

### Параметры

- `lora_a: nn.Parameter` формы `(rank, in_features)` — как в LoRA,
  инициализация `kaiming_uniform_(a=sqrt(5))`.
- `lora_b: nn.Parameter` формы `(out_features, rank)` — инициализируется
  нулём, по той же причине, что и в LoRA и DoRA.
- `kan: KANLayer` формы `(rank, rank)` — тождественный слой при
  инициализации (см. `reset_to_identity` выше). Если `config.learn_input_scale
  is False`, у `kan.input_scale` снимается `requires_grad` сразу после
  создания слоя — это и есть абляционный режим, зафиксированный в
  «Отклонениях от спецификации» плана: не «без `tanh`», а «`input_scale`,
  заморожен на 1.0».

### Почему KAN-LoRA стартует численно как LoRA

`lora_b = 0` обнуляет поправку целиком независимо от того, что делает
`kan`, — этого достаточно для `test_zero_correction_at_initialization`. Но
центральное утверждение работы сильнее: не только на старте, а **при любых
`lora_b`**, пока сплайны внутри `kan` тождественны, `KANLoRALinear`
поэлементно совпадает с `LoRALinear` с теми же `lora_a`, `lora_b`. Это
следует из тождественности `KANLayer` (см. раздел выше): `phi(u) == u`
внутри сетки, поэтому `B * phi(A x) == B * (A x)` — ровно формула LoRA.
Совпадение точное (`atol=1e-10`), а не приближённое, потому что тождество
`KANLayer` само точное (свойство линейной точности B-сплайнов), а не
результат подгонки. Проверено `test_matches_lora_with_identity_splines`;
обратное свойство — что сдвинутые коэффициенты сплайна действительно уводят
выход от LoRA, а не остаются мёртвым грузом, — проверено
`test_learned_nonlinearity_moves_output_away_from_lora`.

Тест на совпадение с LoRA использует вход, намеренно смасштабированный так,
чтобы `A x` гарантированно попадало внутрь сетки `[-1, 1]` (`small_input`,
множитель `0.02`): за пределами сетки `KANLayer` не тождественен, а
обнуляется (см. `fraction_inside_grid`), и там формула LoRA не обязана
воспроизводиться.

### Методы

- `delta(x) -> Tensor` — вычисляет `A x`, пропускает через `kan`, применяет
  `B` и масштаб `scaling`. Попутно, как побочный эффект (`torch.no_grad()`),
  сохраняет `kan.fraction_inside_grid(A x)` во внутреннее поле
  `_fraction_inside_grid` — снимается на каждом проходе, а не отдельным
  диагностическим прогоном.
- `forward(x) -> Tensor` — `self.base(x) + self.delta(x)`.
- `last_fraction_inside_grid() -> float | None` — последнее сохранённое
  значение; `None` до первого прямого прохода.
- `analytic_parameter_count() -> int` — `rank * (in_features + out_features)`
  (как в LoRA) плюс параметры `KANLayer`:
  `rank^2 * (grid_size + spline_order)` коэффициентов сплайна,
  `rank^2` масштабов сплайна, `rank^2` весов базовой ветви и `rank` масштабов
  входа (последнее — только если `learn_input_scale is True`, иначе
  `input_scale` заморожен и не входит ни в аналитический, ни в фактический
  счётчик обучаемых параметров).
- `can_merge` — всегда `False`: поправка нелинейна по `x`, слить её в веса
  основы нельзя даже после обучения. `merge()` не переопределён и
  наследует поведение `AdapterLinear.merge()` — бросает `NotImplementedError`.
  Отсюда следствие для замера задержки (Задача 20 плана, ещё не
  реализована): KAN-LoRA всегда платит дополнительным проходом на
  инференсе, в отличие от LoRA и DoRA, которые после слияния не стоят
  ничего.

## Проверенные инварианты

Тесты закрепляют свойства, на которых держится всё дальнейшее сравнение
методов, а не только корректность отдельных формул:

- Все параметры `self.base` заморожены сразу после конструктора
  (`test_base_is_frozen`).
- Обучаемых параметров ровно столько, сколько даёт аналитическая формула
  (`test_trainable_count_excludes_frozen_base`,
  `test_parameter_count_matches_formula` в `test_lora.py`).
- LoRA, DoRA и KAN-LoRA стартуют численно неотличимыми от немодифицированной
  модели (`test_zero_correction_at_initialization` во всех трёх тестовых
  модулях, допуски `atol=1e-12`, `atol=1e-10` и `atol=1e-12` соответственно)
  — иначе разницу в качестве после обучения нельзя было бы приписать
  методу, а не разной точке старта.
- Слияние воспроизводит выход адаптера и не трогает исходный `base.weight`
  (`test_merged_linear_reproduces_adapter_output` для LoRA и DoRA,
  `test_merge_does_not_touch_the_original_base` для LoRA); KAN-LoRA слияние
  явно отвергает (`test_refuses_to_merge`).
- Градиент доходит до обучаемых параметров адаптера, но не до
  `base.weight` (`test_gradients_reach_both_matrices` для LoRA,
  `test_gradients_reach_magnitude_and_both_matrices` для DoRA,
  `test_gradients_reach_spline_coefficients` для KAN-LoRA, включая
  коэффициенты сплайна внутри `kan`).
- DoRA при `magnitude`, равной норме адаптированного направления,
  поэлементно совпадает с LoRA
  (`test_reduces_to_lora_when_magnitude_matches_direction_norm`); KAN-LoRA
  при тождественных сплайнах поэлементно совпадает с LoRA
  (`test_matches_lora_with_identity_splines`) — и расходится с ней, если
  коэффициенты сплайна сдвинуты
  (`test_learned_nonlinearity_moves_output_away_from_lora`).
- `AdapterLinear` нельзя инстанцировать напрямую
  (`test_interface_cannot_be_instantiated_directly`) — абстрактные методы
  действительно абстрактны.
- Аналитический счётчик параметров KAN-LoRA совпадает с фактическим и в
  обычном режиме, и в режиме абляции с замороженным `input_scale`
  (`test_parameter_count_matches_formula`,
  `test_frozen_input_scale_is_excluded_from_training`) — расхождение между
  ними означало бы неверную колонку «число параметров» в итоговых таблицах.

## Внедрение адаптеров в модель (`inject.py`)

Один внедритель на все три метода — по той же причине, по которой у них
один интерфейс `AdapterLinear`: если бы каждый метод правил модель своим
кодом, различие в наборе затронутых слоёв просочилось бы в результаты, не
выдав ошибки.

`ADAPTER_TYPES: dict[str, type[AdapterLinear]]` — реестр `{"lora": LoRALinear,
"dora": DoRALinear, "kan_lora": KANLoRALinear}` по имени метода из
конфигурации прогона.

`DEFAULT_TARGET_MODULES: tuple[str, ...]` — все семь проекций блока Qwen2 по
методологии проекта: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,
down_proj` (внимание целиком и все три матрицы MLP).

```python
def inject_adapters(
    model: nn.Module,
    method: str,
    config: AdapterConfig,
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
) -> InjectionReport
```

Правит модель на месте (in place) и делает это в четыре шага:

1. Замораживает **всю** модель через `freeze(model)` — включая слои, которые
   не будут заменены (эмбеддинги, нормализация, голова), чтобы обучаемыми
   остались только параметры адаптеров.
2. Находит все `nn.Linear`, чьё последнее имя атрибута (`name.split(".")[-1]`)
   входит в `target_modules`. Если хотя бы одно из запрошенных имён не
   встретилось ни разу в модели — бросает `ValueError` с перечислением
   отсутствующих имён, а не внедряет меньше слоёв молча: опечатка в имени
   проекции (например, `"qkv_proj"` вместо раздельных `q_proj`/`k_proj`/`v_proj`)
   иначе привела бы к сравнению методов на разном числе адаптированных слоёв.
3. Заменяет каждый найденный `nn.Linear` на `adapter_type(module, config)`
   через `setattr` на родительском модуле (родитель ищется по
   `model.get_submodule(parent_name)`, атрибут — последний сегмент имени).
4. Возвращает `InjectionReport(method, replaced, trainable_parameters, total_parameters)`,
   где `replaced` — отсортированный кортеж полных имён заменённых модулей, а
   счётчики параметров считаются по **всей** модели после замены (не только
   по адаптерам) — это число прямо идёт в карточку результата эксперимента.

Метод, отсутствующий в `ADAPTER_TYPES` (`inject_adapters(model, "qlora",
...)`), падает `KeyError` при обращении к реестру — так же, как
`load_examples` из `data/loaders.py` падает на неизвестном разбиении.

```python
def adapter_modules(model: nn.Module) -> list[tuple[str, AdapterLinear]]
```

Перечисляет все внедрённые адаптеры парами `(полное_имя, модуль)`,
отсортированными по имени. Порядок зафиксирован намеренно: по нему
[`model_nonlinearity`](nonlinearity.md) собирает построчную диагностику
сплайнов, и строки таблицы обязаны совпадать между разными прогонами
одной и той же модели.

### Фикстура `tiny_causal_lm` (`tests/conftest.py`)

Общая фикстура для этого и последующих модулей (обучение, генерация,
анализ) — крошечная, но **настоящая** модель архитектуры Qwen2
(`Qwen2Config` с `hidden_size=32`, `num_hidden_layers=2`, `vocab_size=64`),
а не заглушка с произвольными именами слоёв. Имена проекций совпадают с
именами в реальной `Qwen/Qwen2.5-Coder-0.5B-Instruct`, поэтому внедритель,
цикл обучения и генерация проверяются на том же коде, который пойдёт в
эксперименты на GPU, но за секунды и на процессоре.

## Проверенные инварианты внедрения

- Все семь целевых проекций заменяются в каждом из двух слоёв тестовой
  модели, независимо от метода (`test_replaces_every_target_projection`).
- LoRA, DoRA и KAN-LoRA затрагивают **ровно один и тот же** набор полных
  имён слоёв (`test_all_methods_touch_the_same_slots`) — иначе сравнение
  методов сравнивало бы ещё и разный набор адаптированных проекций.
- Выход всей модели не меняется сразу после внедрения, при любом из трёх
  методов (`test_output_is_unchanged_right_after_injection`, `atol=1e-5`) —
  это тот же инвариант «нулевая поправка на старте», что и у отдельных
  адаптеров, но проверенный на уровне целой модели, а не одного слоя.
- Обучаемыми остаются только параметры адаптеров, ни один заморожен
  `nn.Linear.base` не просачивается в список обучаемых
  (`test_only_adapter_parameters_are_trainable`, проверка `".base." not in
  name`).
- Счётчики в `InjectionReport` совпадают с фактическим подсчётом по модели
  (`test_report_counts_match_the_model`).
- `adapter_modules` перечисляет ровно то, что внедрил `inject_adapters`, в
  отсортированном порядке (`test_adapter_modules_enumerates_what_was_injected`).
- KAN-LoRA при равном ранге даёт больше обучаемых параметров, чем LoRA
  (`test_kan_lora_costs_a_couple_of_percent_more_parameters`) — число,
  которое пойдёт в таблицу вместо отдельного прогона LoRA с искусственно
  завышенным рангом ради «честного» сравнения по бюджету параметров.
- Неизвестный метод и опечатка в имени целевого слоя падают явно, а не
  внедряют часть адаптеров молча (`test_unknown_method_is_rejected`,
  `test_missing_target_module_is_rejected`).

## Не реализовано

- Чтение схемы БД и наборов данных Spider/PAUQ (`kanlora/data/`) описано
  отдельно в `docs/data.md`.
- Промпт, токенизация и маска функции потерь (`data/collate.py`, задача 9
  плана) — соединяют внедрённую модель с данными; ещё не написаны.

Эти пункты не описываются подробнее здесь, чтобы не документировать
несуществующее поведение; их интерфейсы зафиксированы в
`docs/superpowers/plans/2026-09-09-kan-lora-text2sql.md`.
