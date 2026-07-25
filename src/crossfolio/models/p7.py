"""The attention-earns-its-place pair.

P7Encoder: attention-free per-stock backbone (shared encoder + FFN + head) —
the probe-verdict baseline. GatedCrossSectional: the SAME backbone plus
attention as a zero-init gated residual side-channel: x + g_b * MHSA(ln(x)),
one learned scalar gate per block (scalar, not per-head, so gate=0 gives EXACT
P7 equivalence — a per-head gate can't cancel the proj bias). At init it IS
the P7 baseline; gradients open gates only if cross-sectional context pays.
The fitted gate magnitude is the round's instrument: how much context does
this data demand?
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiHeadSelfAttention
from .base import Allocator


class _Backbone(Allocator):
    def __init__(self, N, T, cfg):
        super().__init__(N, T, cfg)
        d = cfg.d_model
        self.encoder = nn.Sequential(
            nn.Linear(T, cfg.enc_hidden), nn.GELU(), nn.Linear(cfg.enc_hidden, d)
        )
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.ln_out = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        with torch.no_grad():
            self.head.weight *= cfg.head_init_scale
            self.head.bias *= cfg.head_init_scale

    def _finish(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ffn(x)
        return self.head(self.ln_out(x)).squeeze(-1)


class P7Encoder(_Backbone):
    """No cross-stock interaction anywhere before the softmax."""

    def logits(self, X: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return self._finish(self.encoder(X)), {}


class _GatedBlock(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.mhsa = MultiHeadSelfAttention(d, heads)
        self.gate = nn.Parameter(torch.zeros(()))


class GatedCrossSectional(_Backbone):
    def __init__(self, N, T, cfg):
        super().__init__(N, T, cfg)
        self.blocks = nn.ModuleList(
            _GatedBlock(cfg.d_model, cfg.heads) for _ in range(cfg.n_blocks)
        )

    def logits(self, X: torch.Tensor) -> tuple[torch.Tensor, dict]:
        x = self.encoder(X)
        attn = None
        for b in self.blocks:
            a, attn = b.mhsa(b.ln(x))
            x = x + b.gate * a
        return self._finish(x), {
            "attn": attn, "gates": torch.stack([b.gate for b in self.blocks]),
        }
