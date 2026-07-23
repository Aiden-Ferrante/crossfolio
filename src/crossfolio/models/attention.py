"""Cross-sectional attention: a stock is a token, the universe at a moment is
the sentence. Attention runs ACROSS THE N STOCKS (not across time), so each
stock's representation is contextualized by the rest of the market — a 5% drop
means something different when everything else dropped 5% too.

Hand-rolled multi-head attention (not nn.MultiheadAttention) so per-head
weights are cleanly exposed in aux["attn"] — for pedagogy, and for the interp
question: do heads recover sector/correlation structure unsupervised?

Parameter sharing is the point: the encoder and head are the SAME weights for
every stock ("one copy of the same neuron per stock"), so params barely grow
with N — the counter-lesson to the linear baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import Allocator


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, heads: int):
        super().__init__()
        assert d_model % heads == 0
        self.heads, self.dk = heads, d_model // heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # (B, heads, S, dk)
        q, k, v = (
            t.view(B, S, self.heads, self.dk).transpose(1, 2) for t in (q, k, v)
        )
        attn = torch.softmax(q @ k.transpose(-2, -1) / self.dk**0.5, dim=-1)  # (B, h, S, S)
        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        return self.proj(out), attn


class Block(nn.Module):
    """Pre-LN transformer block; attention runs over the stock axis."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mhsa = MultiHeadSelfAttention(d, heads)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a, attn = self.mhsa(self.ln1(x))
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return x, attn


class CrossSectionalAttention(Allocator):
    def __init__(self, N, T, cfg):
        super().__init__(N, T, cfg)
        d = cfg.d_model
        # shared per-stock encoder: same weights applied to every stock's window
        self.encoder = nn.Sequential(
            nn.Linear(T, cfg.enc_hidden), nn.GELU(), nn.Linear(cfg.enc_hidden, d)
        )
        # the "token embedding": who the stock IS, alongside how it's been moving
        self.id_embed = nn.Embedding(N, d) if cfg.use_id_embed else None
        self.blocks = nn.ModuleList(Block(d, cfg.heads) for _ in range(cfg.n_blocks))
        self.ln_out = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)  # shared across stocks
        with torch.no_grad():
            self.head.weight *= cfg.head_init_scale
            self.head.bias *= cfg.head_init_scale

    def logits(self, X: torch.Tensor) -> tuple[torch.Tensor, dict]:
        x = self.encoder(X)                                       # (B, N, d)
        if self.id_embed is not None:
            x = x + self.id_embed.weight.unsqueeze(0)
        attn = None
        for block in self.blocks:
            x, attn = block(x)                                    # last block's maps kept
        return self.head(self.ln_out(x)).squeeze(-1), {"attn": attn}  # (B, N), (B, h, N, N)
