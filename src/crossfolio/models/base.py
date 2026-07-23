"""Allocator contract: forward(X: (B, N, T)) -> (w: (B, N) softmaxed, aux: dict).

The allocation is the OUTPUT; alpha is computed by the loss, never predicted.
Dollars = capital * w, applied at reporting time only.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelCfg


class Allocator(nn.Module):
    def __init__(self, N: int, T: int, cfg: ModelCfg):
        super().__init__()
        self.N, self.T, self.cfg = N, T, cfg

    def logits(self, X: torch.Tensor) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, dict]:
        z, aux = self.logits(X)
        return torch.softmax(z, dim=-1), aux
