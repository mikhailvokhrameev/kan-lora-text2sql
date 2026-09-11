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
from transformers.tokenization_utils_base import BatchEncoding

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


class TinyTokenizer:
    """Посимвольный токенизатор для батчевой генерации.

    Как FakeTokenizer в tests/data/test_collate.py, но принимает список строк
    за раз, дополняет слева (как настоящий токенизатор при генерации) и умеет
    пакетно декодировать — этого достаточно, чтобы прогнать generate_sql без
    сети и без словаря в 150k токенов реального Qwen2.5.
    """

    pad_token_id = 0
    eos_token_id = 1

    def __init__(self) -> None:
        self.padding_side = "left"

    def _encode_one(self, text: str) -> list[int]:
        return [ord(character) % VOCAB_SIZE for character in text]

    def __call__(
        self, texts: list[str], return_tensors: str = "pt", padding: bool = True
    ) -> BatchEncoding:
        assert return_tensors == "pt"
        sequences = [self._encode_one(text) for text in texts]
        width = max(len(sequence) for sequence in sequences)

        input_ids = []
        attention_mask = []
        for sequence in sequences:
            pad_length = width - len(sequence)
            pad = [self.pad_token_id] * pad_length
            mask_pad = [0] * pad_length
            mask_real = [1] * len(sequence)
            if self.padding_side == "left":
                input_ids.append(pad + sequence)
                attention_mask.append(mask_pad + mask_real)
            else:
                input_ids.append(sequence + pad)
                attention_mask.append(mask_real + mask_pad)

        return BatchEncoding(
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        )

    def batch_decode(self, sequences: torch.Tensor, skip_special_tokens: bool = True) -> list[str]:
        texts = []
        for row in sequences.tolist():
            if skip_special_tokens:
                row = [code for code in row if code not in (self.pad_token_id, self.eos_token_id)]
            texts.append("".join(chr(code) for code in row))
        return texts


@pytest.fixture
def tiny_tokenizer() -> TinyTokenizer:
    return TinyTokenizer()


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
