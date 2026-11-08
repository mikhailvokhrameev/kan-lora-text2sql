"""Итоговая таблица из завершённых сопоставимых карточек версии 2."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from kanlora.data.loaders import LAYOUTS

METHODS = ("none", "lora", "dora", "kan_lora")


def load_cards(directory: Path) -> list[dict]:
    directory = Path(directory)
    cards = []
    allowed = set()
    for path in sorted(directory.glob("*/card.json")):
        card = json.loads(path.read_text(encoding="utf-8"))
        if card.get("schema_version") != 2:
            raise ValueError(f"{path}: требуется schema_version=2; выберите каталог v2")
        artifacts = card.get("artifacts", {})
        required = {"predictions"} | ({"adapter"} if card.get("method") != "none" else set())
        for name in sorted(required | artifacts.keys(), reverse=True):
            value = artifacts.get(name)
            if not isinstance(value, str) or Path(value).is_absolute() or ".." in Path(value).parts or not (path.parent / value).is_file():
                raise ValueError(f"{path}: отсутствует или недопустим артефакт {name}")
        _validate_card(card)
        prediction_path = path.parent / artifacts["predictions"]
        predictions = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines()]
        if [item["index"] for item in predictions] != card["provenance"]["indices"]["eval"]:
            raise ValueError(f"{path}: predictions не соответствуют indices.eval")
        root = Path(card["config"]["data_root"]) / card["dataset"]
        layout = LAYOUTS[card["dataset"]]
        expected = {str(root / layout.database_dir / item["db_id"] / (item["db_id"] + ".sqlite")) for item in predictions}
        hashes = card["provenance"]["file_hashes"]
        if any(not hashes.get(name) for name in expected):
            raise ValueError(f"{path}: отсутствует file_hashes оценочной SQLite")
        allowed.add(path)
        allowed.update(path.parent / name for name in artifacts.values())
        cards.append(card)
    for path in sorted(directory.rglob("*.json")):
        if path not in allowed:
            raise ValueError(f"{path}: устаревший JSON; выберите каталог v2 одного run_kind")
    return cards


def _flatten(value: dict, prefix: str = "") -> dict:
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(_flatten(item, name))
        else:
            result[name] = item
    return result


def _conditions(card: dict, baseline_pair: bool, kan_pair: bool) -> dict:
    config = dict(card["config"])
    config.pop("method", None)
    if baseline_pair:
        for key in ("adapter", "train", "optimizer", "train_subset", "validation_size", "subsample_seed"):
            config.pop(key, None)
    elif not kan_pair:
        adapter = dict(config.get("adapter", {}))
        for key in ("grid_size", "spline_order", "learn_input_scale"):
            adapter.pop(key, None)
        config["adapter"] = adapter
    provenance = dict(card["provenance"])
    provenance.pop("config", None)  # Сравнивается выше с исключениями baseline.
    if baseline_pair:
        indices = dict(provenance.get("indices", {}))
        indices.pop("train", None)
        indices.pop("validation", None)
        provenance["indices"] = indices
        # У baseline нет хэша train; общие файлы сверяются отдельно ниже.
        provenance.pop("file_hashes", None)
    return _flatten({"config": config, "provenance": provenance,
                     "versions": card["versions"], "git_commit": card["git_commit"]})


def _validate_card(card: dict) -> None:
    provenance = card.get("provenance", {})
    for field, length in (("evaluator_commit", 40), ("lock_sha256", 64)):
        if not re.fullmatch(rf"[0-9a-f]{{{length}}}", str(provenance.get(field, ""))):
            raise ValueError(f"{card.get('run_id')}: отсутствует или неверен provenance.{field}")
    dataset = card["dataset"]
    layout = LAYOUTS[dataset]
    root = Path(card["config"]["data_root"]) / dataset
    hashes = provenance.get("file_hashes", {})
    required = [str(root / layout.eval_file), str(root / layout.tables_file)]
    db_root = root / layout.database_dir
    databases = [name for name in hashes if Path(name).parent.parent == db_root
                 and Path(name).name == Path(name).parent.name + ".sqlite"]
    if not databases or any(not hashes.get(name) for name in required + databases):
        raise ValueError(f"{card.get('run_id')}: отсутствует file_hashes оценочных данных или SQLite")


def validate_comparability(cards: list[dict]) -> None:
    if not cards:
        raise ValueError("нет карточек для итоговой таблицы")
    cells = set()
    kinds = set()
    for card in cards:
        if card.get("schema_version") != 2:
            raise ValueError("требуется schema_version=2; выберите каталог v2")
        _validate_card(card)
        key = (card["dataset"], card["method"], card["seed"])
        if key in cells:
            raise ValueError(f"повтор ячейки {key}")
        cells.add(key)
        if card["method"] not in METHODS:
            raise ValueError(f"неизвестный method: {card['method']}")
        kinds.add(card["config"]["run_kind"])
    if len(kinds) != 1:
        raise ValueError("смешаны run_kind")
    for index, first in enumerate(cards):
        for second in cards[index + 1:]:
            if first["dataset"] != second["dataset"]:
                continue
            baseline_pair = "none" in (first["method"], second["method"])
            kan_pair = first["method"] == second["method"] == "kan_lora"
            left = _conditions(first, baseline_pair, kan_pair)
            right = _conditions(second, baseline_pair, kan_pair)
            if baseline_pair:
                hashes1 = first["provenance"].get("file_hashes", {})
                hashes2 = second["provenance"].get("file_hashes", {})
                baseline_hashes = hashes1 if first["method"] == "none" else hashes2
                for name in baseline_hashes:
                    left[f"provenance.file_hashes.{name}"] = hashes1.get(name)
                    right[f"provenance.file_hashes.{name}"] = hashes2.get(name)
            missing = object()
            differing = sorted(key for key in left.keys() | right.keys() if left.get(key, missing) != right.get(key, missing))
            if differing:
                raise ValueError(f"{first['run_id']} и {second['run_id']}: различаются {', '.join(differing)}")


def _metric(card: dict, name: str) -> str:
    value = card["metrics"].get(name)
    return "не измерялась" if value is None else f"{value:.4f}"


def main_table(cards: list[dict]) -> str:
    validate_comparability(cards)
    lines = ["| Набор | Метод | Seed | Exact Match | Execution Accuracy | Обучаемых параметров | Пик памяти обучения, ГиБ |",
             "|---|---|---|---|---|---|---|"]
    for card in sorted(cards, key=lambda c: (c["dataset"], METHODS.index(c["method"]), c["seed"])):
        if card["method"] == "none":
            parameters = memory = "не применимо"
        else:
            parameters = str(card["trainable_parameters"])
            peak = card["peak_memory_bytes"]
            memory = f"{peak / 2**30:.2f}" if peak else "не измерялась"
        lines.append(f"| {card['dataset']} | {card['method']} | {card['seed']} | {_metric(card, 'exact_match')} | {_metric(card, 'execution_accuracy')} | {parameters} | {memory} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Итоговая таблица карточек v2")
    parser.add_argument("--results", type=Path, default=Path("results/v2/final"))
    parser.add_argument("--output", type=Path, default=Path("results/v2/tables"))
    arguments = parser.parse_args()
    table = main_table(load_cards(arguments.results))
    arguments.output.mkdir(parents=True, exist_ok=True)
    output = arguments.output / "main.md"
    output.write_text("# Результаты\n\n" + table + "\n", encoding="utf-8")
    print(f"таблица: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
