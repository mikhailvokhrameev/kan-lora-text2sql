"""Проверки замера пиковой видеопамяти.

Сравнение методов по памяти осмысленно только при одинаковых длине
последовательности, размере батча и точности — а сам замер обязан быть
одинаковым куском кода для всех трёх методов и не падать там, где CUDA нет.
"""

import torch

from kanlora.train.memory import PeakMemoryTracker


def test_reports_zero_on_cpu() -> None:
    tracker = PeakMemoryTracker(torch.device("cpu"))
    tracker.reset()
    assert tracker.peak_bytes() == 0


def test_reset_is_idempotent_on_cpu() -> None:
    tracker = PeakMemoryTracker(torch.device("cpu"))
    tracker.reset()
    tracker.reset()
    assert tracker.peak_bytes() == 0


@torch.inference_mode()
def test_tracks_allocation_growth_on_cuda() -> None:
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA недоступна")

    device = torch.device("cuda")
    tracker = PeakMemoryTracker(device)
    tracker.reset()
    baseline = tracker.peak_bytes()

    held = torch.empty(1024, 1024, dtype=torch.float32, device=device)
    assert tracker.peak_bytes() >= baseline + held.numel() * 4
