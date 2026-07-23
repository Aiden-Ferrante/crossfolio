"""Losses: the network outputs an allocation; alpha is the GRADE, computed here.

All are pure functions of (w, y, y_spy):
  w     (B, N) softmaxed allocation
  y     (B, N) forward simple returns over the grading horizon
  y_spy (B,)   benchmark forward simple return

`sharpe` is batch-level: each example contributes its horizon excess return,
Sharpe = mean/std across the batch (so the loss value depends on batch size —
compare losses only at fixed B). `mean_excess` is kept deliberately to
demonstrate the all-in-on-the-ex-post-winner failure mode.
"""

from __future__ import annotations

import torch

EPS = 1e-8


def excess_returns(w: torch.Tensor, y: torch.Tensor, y_spy: torch.Tensor) -> torch.Tensor:
    return (w * y).sum(-1) - y_spy


def hhi(w: torch.Tensor) -> torch.Tensor:
    """Herfindahl concentration, mean over batch. 1/N (diversified) .. 1 (all-in)."""
    return (w**2).sum(-1).mean()


def neg_batch_sharpe(w, y, y_spy) -> torch.Tensor:
    e = excess_returns(w, y, y_spy)
    return -e.mean() / (e.std(unbiased=True) + EPS)


def neg_mean_excess(w, y, y_spy) -> torch.Tensor:
    return -excess_returns(w, y, y_spy).mean()


_BASE = {"sharpe": neg_batch_sharpe, "mean_excess": neg_mean_excess}


def make_loss(name: str, hhi_lambda: float):
    base = _BASE[name]

    def loss(w, y, y_spy) -> torch.Tensor:
        return base(w, y, y_spy) + hhi_lambda * hhi(w)

    loss.__name__ = f"{name}+{hhi_lambda}*hhi"
    return loss
