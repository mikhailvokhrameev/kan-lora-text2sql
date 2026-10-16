"""Проверки замера задержки.

LoRA и DoRA после обучения сливаются с весами и на инференсе не стоят ничего.
KAN-LoRA не сливается в принципе: поправка нелинейно зависит от входа. Это его
цена, и здесь она получает число.
"""

import pytest
import torch

from kanlora.adapters.base import AdapterConfig
from kanlora.adapters.inject import adapter_modules, inject_adapters
from kanlora.eval.latency import LatencyReport, measure_latency, merge_adapters

DEVICE = torch.device("cpu")


@pytest.mark.parametrize("method", ["lora", "dora"])
def test_merging_replaces_adapters_with_plain_linear(tiny_causal_lm, method: str) -> None:
    inject_adapters(tiny_causal_lm, method, AdapterConfig(rank=4))
    merged = merge_adapters(tiny_causal_lm)

    assert merged == 14  # 2 слоя * 7 проекций
    assert adapter_modules(tiny_causal_lm) == []


@pytest.mark.parametrize("method", ["lora", "dora"])
def test_merging_does_not_change_the_output(tiny_causal_lm, method: str) -> None:
    """Слияние обязано быть тождественным преобразованием, иначе замер сравнивает разные модели."""
    inject_adapters(tiny_causal_lm, method, AdapterConfig(rank=4))
    for _, module in adapter_modules(tiny_causal_lm):
        with torch.no_grad():
            module.lora_b.normal_(std=0.01)

    tokens = torch.randint(0, 64, (2, 9))
    with torch.no_grad():
        before = tiny_causal_lm(tokens).logits.clone()
        merge_adapters(tiny_causal_lm)
        after = tiny_causal_lm(tokens).logits

    assert torch.allclose(before, after, atol=1e-5)


def test_kan_lora_refuses_to_merge(tiny_causal_lm) -> None:
    """Не «пока не реализовано», а свойство метода: поправка зависит от входа."""
    inject_adapters(tiny_causal_lm, "kan_lora", AdapterConfig(rank=4))
    with pytest.raises(NotImplementedError):
        merge_adapters(tiny_causal_lm)


@pytest.mark.slow
def test_latency_report_is_filled(tiny_causal_lm, tiny_tokenizer) -> None:
    inject_adapters(tiny_causal_lm, "lora", AdapterConfig(rank=4))
    report = measure_latency(
        tiny_causal_lm, tiny_tokenizer, "SELECT",
        device=DEVICE, max_new_tokens=8, repeats=2, warmup=1,
    )

    assert isinstance(report, LatencyReport)
    assert report.tokens == 8
    assert report.total_seconds > 0
    assert report.seconds_per_token == pytest.approx(report.total_seconds / report.tokens)
    assert report.merged is False


class _BosTokenizer:
    """Добавляет BOS при add_special_tokens=True; на обучении и оценке его нет."""

    pad_token_id = 0
    bos_token_id = 2

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        ids = [ord(character) % 64 for character in text]
        return {"input_ids": ([self.bos_token_id] if add_special_tokens else []) + ids}


class _RecordingModel:
    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []

    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, input_ids, attention_mask, **kwargs):
        self.calls.append(input_ids)
        return input_ids


def test_latency_prompt_is_tokenized_without_special_tokens() -> None:
    """Промпт замера токенизируется так же, как на обучении и оценке (без BOS)."""
    model = _RecordingModel()
    measure_latency(
        model, _BosTokenizer(), "SELECT", device=DEVICE, max_new_tokens=2, repeats=1, warmup=0
    )
    assert model.calls[0][0].tolist() == [ord(c) % 64 for c in "SELECT"]
