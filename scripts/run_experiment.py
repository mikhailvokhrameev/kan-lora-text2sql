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
