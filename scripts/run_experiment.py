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
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kanlora.adapters.inject import inject_adapters
from kanlora.config import METHODS, apply_overrides, load_config
from kanlora.data.collate import Collator, build_dataset, build_prompts
from kanlora.data.loaders import (
    LAYOUTS,
    database_path,
    load_dataset_schemas,
    load_examples,
    select_indices,
)
from kanlora.eval.exact_match import exact_match_by_hardness
from kanlora.eval.execution import EvaluationDataError, execution_accuracy, validate_execution_data
from kanlora.eval.generate import count_truncated_prompts, generate_sql
from kanlora.provenance import collect_provenance, file_hashes, require_clean_tracked_tree
from kanlora.results import build_result_card, write_result_card
from kanlora.train.loop import set_seed, train


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Один прогон эксперимента")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument(
        "--method", choices=METHODS, help="none — базовая модель без адаптера, только оценка"
    )
    parser.add_argument("--dataset", choices=["spider", "pauq"])
    parser.add_argument("--seed", type=int)
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


def validate_run(config, device, eval_examples):
    if config.run_kind != "final":
        return
    if device.type != "cuda" or "GTX 1080 Ti" not in torch.cuda.get_device_name(device) or torch.cuda.get_device_capability(device) != (6, 1):
        raise RuntimeError("final требует целевую GPU GTX 1080 Ti")
    if config.seed != 0:
        raise RuntimeError("final требует seed 0")
    if config.eval_limit is not None or not eval_examples:
        raise RuntimeError("final требует полный непустой dev")


def load_model_and_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, revision=config.model_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, revision=config.model_revision, dtype=torch.float32
    )
    return model, tokenizer


def main() -> int:
    arguments = parse_arguments()
    config = apply_overrides(
        load_config(arguments.config),
        method=arguments.method,
        dataset=arguments.dataset,
        seed=arguments.seed,
        train_subset=arguments.train_subset,
        eval_limit=arguments.eval_limit,
    )
    print(f"прогон: {config.run_id}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.run_kind == "final":
        require_clean_tracked_tree()
    set_seed(config.seed)

    root = Path(config.data_root) / config.dataset
    layout = LAYOUTS[config.dataset]
    schemas = load_dataset_schemas(root, layout)

    train_examples, validation_examples = [], []
    indices = {"train": [], "validation": [], "eval": []}
    paths = [root / layout.eval_file, root / layout.tables_file]
    if config.method != "none":
        all_train = load_examples(root, layout, "train")
        train_indices, validation_indices = select_indices(
            len(all_train), config.train_subset, config.validation_size, config.subsample_seed
        )
        indices["train"], indices["validation"] = train_indices, validation_indices
        train_examples = [all_train[i] for i in train_indices]
        validation_examples = [all_train[i] for i in validation_indices]
        paths.append(root / layout.train_file)
    eval_examples = load_examples(root, layout, "eval")
    validate_run(config, device, eval_examples)
    if config.eval_limit is not None:
        eval_examples = eval_examples[: config.eval_limit]
    indices["eval"] = list(range(len(eval_examples)))
    paths.extend(database_path(root, layout, db_id) for db_id in sorted({e.db_id for e in eval_examples}))

    gold = [example.query for example in eval_examples]
    db_ids = [example.db_id for example in eval_examples]
    execution_reason = None
    try:
        validate_execution_data(gold, db_ids, root, layout)
    except EvaluationDataError as error:
        if config.run_kind != "smoke":
            raise
        execution_reason = str(error)

    model, tokenizer = load_model_and_tokenizer(config)
    model_revision = getattr(model.config, "_commit_hash", None)
    tokenizer_revision = tokenizer.init_kwargs.get("_commit_hash")
    provenance = collect_provenance(config, device, indices, paths, model_revision, tokenizer_revision)
    injection, report, truncated = adapt_and_train(
        config, model, tokenizer, train_examples, schemas, device, validation_examples
    )

    prompts = build_prompts(eval_examples, schemas, tokenizer)
    eval_truncated = count_truncated_prompts(
        tokenizer, prompts, max_length=config.max_length, max_new_tokens=config.max_new_tokens
    )
    if eval_truncated:
        print(f"внимание: на оценке усечено промптов: {eval_truncated} из {len(prompts)}")
    predicted = generate_sql(
        model, tokenizer, prompts,
        device=device, batch_size=config.eval_batch_size,
        max_new_tokens=config.max_new_tokens, max_length=config.max_length,
    )
    hardness = exact_match_by_hardness(gold, predicted, db_ids, root / layout.tables_file)
    metrics = {
        "exact_match": hardness["all"],
        "exact_match_by_hardness": hardness,
        "execution_accuracy": None,
        "execution_accuracy_reason": execution_reason,
        "truncated_prompts": truncated,
        "eval_truncated_prompts": eval_truncated,
        "evaluated": len(gold),
    }

    if execution_reason is None:
        try:
            metrics["execution_accuracy"] = execution_accuracy(gold, predicted, db_ids, root, layout)
        except EvaluationDataError as error:
            if config.run_kind != "smoke":
                raise
            metrics["execution_accuracy_reason"] = str(error)


    card = build_result_card(
        config=config,
        injection=injection,
        train_report=report,
        metrics=metrics,
        data_files=file_hashes(Path("."), [str(path) for path in paths]),
        # transformers записывает коммит репозитория модели при загрузке с Hub.
        model_revision=getattr(model.config, "_commit_hash", None),
    )
    card["provenance"] = provenance
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
    print(summary)
    print(f"карточка: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
