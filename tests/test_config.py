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
