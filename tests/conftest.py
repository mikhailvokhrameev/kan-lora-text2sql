# tests/conftest.py
"""Общие фикстуры.

Крошечная модель той же архитектуры, что и рабочая: имена проекций совпадают
с настоящими, поэтому внедритель, цикл обучения и генерация проверяются на том
же коде, который пойдёт в эксперименты, но за секунды и на процессоре.
"""

import pytest
import torch
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM

VOCAB_SIZE = 64


@pytest.fixture
def tiny_causal_lm() -> Qwen2ForCausalLM:
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    model = Qwen2ForCausalLM(config)
    return model.to(torch.float32).eval()


def unfreeze_norms(model: nn.Module) -> None:
    """Размораживает веса RMSNorm поверх уже внедрённых адаптеров.

    Только для санитарного теста на переобучение двадцати примеров, вместе с
    unfreeze_embeddings ниже. У tiny_causal_lm эмбеддинги привязаны
    (tie_word_embeddings) и случайны, поэтому с замороженной нормировкой и
    замороженным lm_head функция потерь не может опуститься к нулю ни для
    одного из трёх методов адаптации, независимо от ранга или скорости
    обучения (проверено эмпирически при отладке Задачи 11: при полностью
    замороженной основе — плато около 3.5–4.0 для LoRA/DoRA/KAN-LoRA
    одинаково, то есть дело не в конкретном методе). У реальной предобученной
    модели эмбеддинги и нормировка уже несут полезную структуру, поэтому там
    это ограничение не проявляется, и production-код `inject_adapters` эти
    функции не использует.
    """
    for name, parameter in model.named_parameters():
        if "norm" in name:
            parameter.requires_grad_(True)


def unfreeze_embeddings(model: nn.Module) -> None:
    """Размораживает привязанные (tie_word_embeddings) веса эмбеддингов.

    Одного unfreeze_norms недостаточно: без него функция потерь у tiny_causal_lm
    опускается лишь до ~1.6 за 150 эпох вместо близкого к нулю значения —
    случайная 32-мерная таблица эмбеддингов на 64 токена слишком неразделима,
    чтобы этого добилась одна нормировка.
    """
    for name, parameter in model.named_parameters():
        if "embed_tokens" in name:
            parameter.requires_grad_(True)
