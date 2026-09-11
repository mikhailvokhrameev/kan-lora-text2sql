# tests/conftest.py
"""Общие фикстуры.

Крошечная модель той же архитектуры, что и рабочая: имена проекций совпадают
с настоящими, поэтому внедритель, цикл обучения и генерация проверяются на том
же коде, который пойдёт в эксперименты, но за секунды и на процессоре.
"""

import pytest
import torch
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
