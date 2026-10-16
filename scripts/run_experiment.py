# scripts/run_experiment.py
"""Один прогон целиком: обучение -> оценка -> карточка результата.

`--method none` — базовая модель без адаптера: адаптеры не внедряются,
обучения нет, сразу оценка тем же путём, что и у обученных методов.

Эта же команда без изменений запускает и проверку на 200 примерах, и полный
прогон. Отдельного «настоящего» пути запуска не существует: иначе проверенным
оказался бы не тот код, который считает результаты.

Примеры:
    python scripts/run_experiment.py --config configs/smoke.yaml
    python scripts/run_experiment.py --config configs/base.yaml --method kan_lora \
        --dataset pauq --seed 2
    python scripts/run_experiment.py --config configs/base.yaml --method none
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kanlora.adapters.inject import inject_adapters  # noqa: E402
from kanlora.analysis.spline_stats import model_nonlinearity  # noqa: E402
from kanlora.config import METHODS, apply_overrides, load_config  # noqa: E402
from kanlora.data.collate import Collator, build_dataset, build_prompts  # noqa: E402
from kanlora.data.loaders import (  # noqa: E402
    LAYOUTS, load_dataset_schemas, load_examples, subsample, validation_subsample,
)
from kanlora.eval.exact_match import exact_match_by_hardness  # noqa: E402
from kanlora.eval.execution import count_available_databases, execution_accuracy  # noqa: E402
from kanlora.eval.generate import generate_sql  # noqa: E402
from kanlora.eval.latency import measure_latency
from kanlora.results import build_result_card, file_hashes, write_result_card  # noqa: E402
from kanlora.train.loop import set_seed, train  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Один прогон эксперимента")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument(
        "--method", choices=METHODS, help="none — базовая модель без адаптера, только оценка"
    )
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


def adapt_and_train(
    config, model, tokenizer, train_examples, schemas, device, validation_examples=(),
):
    """Внедряет адаптеры и обучает; возвращает (injection, report, truncated).

    Валидационные примеры кодируются тем же `build_dataset`, что и обучающие;
    их усечённые промпты в счётчик `truncated` не входят.

    Для `method: none` ничего не делает: модель остаётся исходной, обучаемых
    параметров, отчёта обучения и счётчика усечённых обучающих промптов нет
    (None — «не применимо», в карточке null).
    """
    if config.method == "none":
        total = sum(parameter.numel() for parameter in model.parameters())
        print(f"базовая модель без адаптера: параметров {total}, обучения нет")
        return {"trainable_parameters": None, "total_parameters": total, "replaced": []}, None, None

    injection = inject_adapters(model, config.method, config.adapter)
    print(
        f"обучаемых параметров: {injection.trainable_parameters} "
        f"из {injection.total_parameters}, затронуто слоёв: {len(injection.replaced)}"
    )

    dataset, truncated = build_dataset(train_examples, schemas, tokenizer, config.max_length)
    if truncated:
        print(f"внимание: усечено промптов: {truncated} из {len(dataset)}")

    validation_dataset, _ = build_dataset(
        list(validation_examples), schemas, tokenizer, config.max_length
    )

    report = train(
        model=model,
        dataset=dataset,
        collator=Collator(pad_token_id=tokenizer.pad_token_id),
        optimizer_config=config.optimizer,
        train_config=config.train,
        device=device,
        validation_dataset=validation_dataset,
    )
    injection_info = {
        "trainable_parameters": injection.trainable_parameters,
        "total_parameters": injection.total_parameters,
        "replaced": list(injection.replaced),
    }
    return injection_info, report, truncated


def nonlinearity_and_latency(config, model, tokenizer, prompt: str, device):
    """Индекс нелинейности, затем задержка генерации — именно в таком порядке.

    Генерация замера проходит через адаптеры и расширяет observed_range KAN-LoRA,
    а индекс мерится по размаху, накопленному на оценке. Задержка мерится без
    слияния: у LoRA и DoRA после слияния она равна задержке базовой модели
    (прогон `method: none`), а KAN-LoRA не сливается в принципе.
    """
    spline_stats = model_nonlinearity(model) or None
    report = measure_latency(
        model, tokenizer, prompt, device=device, max_new_tokens=config.max_new_tokens
    )
    return spline_stats, {**asdict(report), "device": device.type}


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

    train_examples, validation_examples = [], []
    if config.method != "none":
        all_train = load_examples(root, layout, "train")
        train_examples = subsample(all_train, config.train_subset, config.subsample_seed)
        validation_examples = validation_subsample(
            all_train, config.train_subset, config.validation_size, config.subsample_seed
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
    injection, report, truncated = adapt_and_train(
        config, model, tokenizer, train_examples, schemas, device, validation_examples
    )

    prompts = build_prompts(eval_examples, schemas, tokenizer)
    predicted = generate_sql(
        model, tokenizer, prompts,
        device=device, batch_size=config.eval_batch_size,
        max_new_tokens=config.max_new_tokens, max_length=config.max_length,
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
    # Проверяются все уникальные db_id отложенной выборки, а не только первый:
    # набор ссылается на много разных баз, и первая ничего не говорит про
    # остальные.
    unique_db_ids = set(db_ids)
    available = count_available_databases(root, layout, unique_db_ids)
    if available == 0:
        print("файлы баз данных не найдены: Execution Accuracy не считается")
    else:
        if available < len(unique_db_ids):
            print(
                f"внимание: не найдено баз данных: {len(unique_db_ids) - available} из "
                f"{len(unique_db_ids)} — Execution Accuracy посчитана частично"
            )
        metrics["execution_accuracy"] = execution_accuracy(
            gold, predicted, db_ids, root, layout
        )

    spline_stats, latency = nonlinearity_and_latency(config, model, tokenizer, prompts[0], device)
    card = build_result_card(
        config=config,
        injection=injection,
        train_report=report,
        metrics=metrics,
        spline_stats=spline_stats,
        latency=latency,
        data_files=file_hashes(root, [layout.train_file, layout.eval_file, layout.tables_file]),
        # transformers записывает коммит репозитория модели при загрузке с Hub.
        model_revision=getattr(model.config, "_commit_hash", None),
    )
    path = write_result_card(card, arguments.output)

    summary = (
        f"Exact Match {metrics['exact_match']:.4f}, "
        f"Execution Accuracy {metrics['execution_accuracy']}"
    )
    if report is not None:
        summary += (
            f", пик видеопамяти {report.peak_memory_bytes / 2**30:.2f} ГиБ, "
            f"время {report.seconds / 60:.1f} мин"
        )
    summary += f", задержка {latency['seconds_per_token'] * 1000:.1f} мс/токен"
    print(summary)
    print(f"карточка: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
