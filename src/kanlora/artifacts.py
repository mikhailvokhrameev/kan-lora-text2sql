"""Артефакты отдельного прогона и восстановление адаптера."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from uuid import uuid4

import torch
from torch import nn

from kanlora.adapters.base import AdapterConfig
from kanlora.adapters.inject import DEFAULT_TARGET_MODULES, inject_adapters
from kanlora.config import ExperimentConfig, config_to_dict


def new_run_id(config: ExperimentConfig) -> str:
    return f'{config.method}-{config.dataset}-seed{config.seed}-{uuid4().hex}'


def reserve_run_directory(output: Path, run_kind: str, run_id: str) -> Path:
    directory = Path(output) / run_kind / run_id
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def write_predictions(path: Path, indices: list[int], db_ids: list[str], gold: list[str],
                      predicted: list[str]) -> None:
    if len({len(indices), len(db_ids), len(gold), len(predicted)}) != 1:
        raise ValueError('длины списков предсказаний различаются')
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.writelines(json.dumps({'index': index, 'db_id': db_id, 'gold_sql': gold_sql,
                                         'predicted_sql': predicted_sql}, ensure_ascii=False) + '\n' for index, db_id, gold_sql, predicted_sql in zip(indices, db_ids, gold, predicted))


def save_adapter(model: nn.Module, config: ExperimentConfig, path: Path) -> None:
    if config.method == 'none':
        raise ValueError('у базовой модели нет адаптера')
    tensors = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
               if parameter.requires_grad}
    with Path(path).open('xb') as stream:
        torch.save({'config': config_to_dict(config), 'tensors': tensors}, stream)


def restore_adapter(model: nn.Module, path: Path) -> dict:
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        config = checkpoint['config']
        tensors = checkpoint['tensors']
        base = model.config
        if (getattr(base, '_name_or_path', None) != config['model_name'] or
                getattr(base, '_commit_hash', None) != config['model_revision']):
            raise ValueError('имя или ревизия базовой модели не совпадает')
        adapter_config = AdapterConfig(**config['adapter'])
        method = config['method']
        if not isinstance(tensors, dict) or not all(isinstance(t, torch.Tensor) for t in tensors.values()):
            raise ValueError('некорректные тензоры checkpoint')
    except (KeyError, TypeError, AttributeError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
        raise ValueError('несовместимый checkpoint или базовая модель') from error

    # Сохраняем ссылки, а не копию всей базы: rollback возвращает исходные модули
    # и requires_grad даже если внедрение или копирование оборвалось на середине.
    targets = [(name, module) for name, module in model.named_modules()
               if isinstance(module, nn.Linear) and name.split('.')[-1] in DEFAULT_TARGET_MODULES]
    flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    try:
        inject_adapters(model, method, adapter_config)
        parameters = {name: parameter for name, parameter in model.named_parameters()
                      if parameter.requires_grad}
        if set(parameters) != set(tensors) or any(
            parameters[name].shape != tensors[name].shape for name in parameters
        ):
            raise ValueError('имена или формы параметров адаптера не совпадают')
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(tensors[name])
    except Exception as error:
        for name, original in targets:
            parent_name, _, attribute = name.rpartition('.')
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, attribute, original)
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
        if isinstance(error, ValueError):
            raise
        raise ValueError('несовместимый checkpoint адаптера') from error
    return config
