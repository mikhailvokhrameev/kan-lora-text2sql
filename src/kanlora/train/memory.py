"""Замер пиковой видеопамяти.

Величина сравнима между методами только при жёстко одинаковых длине
последовательности, размере батча и точности. Замер вынесен сюда одним куском
кода именно для того, чтобы «удобная» правка под один метод не разошлась
с остальными.
"""

from __future__ import annotations

import torch

__all__ = ["PeakMemoryTracker"]


class PeakMemoryTracker:
    def __init__(self, device: torch.device) -> None:
        self.device = torch.device(device)
        self.enabled = self.device.type == "cuda"

    def reset(self) -> None:
        if self.enabled:
            torch.cuda.reset_peak_memory_stats(self.device)

    def peak_bytes(self) -> int:
        """Ноль на процессоре: карточка результата тогда честно скажет, что замера не было."""
        if not self.enabled:
            return 0
        return int(torch.cuda.max_memory_allocated(self.device))
