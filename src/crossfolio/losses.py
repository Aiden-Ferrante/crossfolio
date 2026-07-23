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


def mean_ic(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean per-anchor Pearson corr of logits vs y across the N stocks.

    Centered-dot-product form: stays differentiable at zero logit variance
    (never route through .std()). Grades LOGITS, not softmax weights — near
    uniform they're equivalent (softmax is locally affine, corr is affine-
    invariant) and logits avoid winner-domination once softmax saturates.
    """
    l = logits - logits.mean(-1, keepdim=True)
    t = y - y.mean(-1, keepdim=True)
    num = (l * t).sum(-1)
    den = torch.sqrt(((l * l).sum(-1) + EPS) * ((t * t).sum(-1) + EPS))
    return (num / den).mean()


def rank_ic(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Spearman: Pearson corr of ranks. The headline EVAL metric at H=5 —
    weekly returns are fat-tailed and Pearson is outlier-sensitive. Not
    differentiable (argsort); the training loss stays Pearson `mean_ic`."""
    rl = logits.argsort(-1).argsort(-1).float()
    ry = y.argsort(-1).argsort(-1).float()
    return mean_ic(rl, ry)


def make_loss(name: str, hhi_lambda: float):
    """sharpe | mean_excess grade the portfolio scalar (1 outcome/anchor);
    ic grades the cross-section (N outcomes/anchor — densified supervision).
    The HHI penalty applies only to portfolio losses: scale-invariant corr
    doesn't care about the one direction HHI pushes on, they'd only fight."""
    if name == "ic":
        def loss(w, y, y_spy, logits) -> torch.Tensor:
            return -mean_ic(logits, y)
    else:
        base = {"sharpe": neg_batch_sharpe, "mean_excess": neg_mean_excess}[name]

        def loss(w, y, y_spy, logits=None) -> torch.Tensor:
            return base(w, y, y_spy) + hhi_lambda * hhi(w)

    loss.__name__ = f"loss_{name}"
    return loss
