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


class CorrBiasAttention(GatedCrossSectional):
    """Gated attention with a relational inductive bias: attention logits get
    lambda_h * B added pre-softmax, where B_ij = corr(window_i, window_j) is
    computed per anchor from the RAW input windows. lambda per (block, head)
    init 0 — a second zero-init dial steering WHERE to attend. At init this is
    exactly the P7 baseline (gate=0, lambda=0)."""

    def __init__(self, N, T, cfg):
        super().__init__(N, T, cfg)
        self.lambdas = nn.ParameterList(
            nn.Parameter(torch.zeros(cfg.heads)) for _ in self.blocks
        )
        # dL/dlambda is proportional to the gate: both dials at 0 would
        # deadlock (lambda can't learn while the gate is shut). Small nonzero
        # gate keeps both gradients alive; init-equivalence becomes approximate.
        with torch.no_grad():
            for b in self.blocks:
                b.gate.fill_(0.1)

    def logits(self, X: torch.Tensor) -> tuple[torch.Tensor, dict]:
        Xc = X - X.mean(-1, keepdim=True)
        Xn = Xc / (Xc.norm(dim=-1, keepdim=True) + 1e-8)
        B_corr = Xn @ Xn.transpose(-2, -1)              # (B, N, N)
        x = self.encoder(X)
        attn = None
        for blk, lam in zip(self.blocks, self.lambdas):
            blk.mhsa.attn_bias = lam.view(1, -1, 1, 1) * B_corr.unsqueeze(1)
            a, attn = blk.mhsa(blk.ln(x))
            blk.mhsa.attn_bias = None
            x = x + blk.gate * a
        return self._finish(x), {
            "attn": attn,
            "gates": torch.stack([b.gate for b in self.blocks]),
            "lambdas": torch.stack([l.detach().clone() for l in self.lambdas]),
        }
