"""Stage 0: optimization wearing NN clothes — the idea this repo grew from.

No input, no network: the allocation vector IS the trainable parameters,
w = softmax(theta), graded by gradient descent on realized excess returns over
the train block. Run with both losses side by side to watch `mean_excess` go
all-in on the ex-post winner while `sharpe`+HHI diversifies.
"""

from __future__ import annotations

import numpy as np
import torch

from .config import Cfg
from .data.dataset import Panel, load_panel
from .data.splits import purged_split, valid_anchors
from .losses import hhi, make_loss

STEPS = 500
LR = 0.05


def _targets(panel: Panel, anchors: np.ndarray, H: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward H-day simple returns for every anchor at once: (n, N) and (n,)."""
    logc = np.cumsum(panel.returns, axis=0)
    logc_spy = np.cumsum(panel.spy)
    y = np.expm1(logc[anchors + H] - logc[anchors])
    y_spy = np.expm1(logc_spy[anchors + H] - logc_spy[anchors])
    return torch.from_numpy(y.astype(np.float32)), torch.from_numpy(y_spy.astype(np.float32))


def _sharpe_monthly(w: torch.Tensor, y: torch.Tensor, y_spy: torch.Tensor, H: int) -> float:
    """Annualized Sharpe of excess returns on non-overlapping (stride-H) anchors."""
    e = ((w * y).sum(-1) - y_spy)[::H]
    return float(e.mean() / (e.std() + 1e-8) * np.sqrt(252 / H))


def run(cfg: Cfg | None = None) -> None:
    cfg = cfg or Cfg()
    panel = load_panel()
    H = cfg.data.H
    anchors = valid_anchors(panel.D, cfg.data.T, H)
    train, _, test = purged_split(anchors, cfg.split.train_frac, cfg.split.val_frac, H)
    y_tr, spy_tr = _targets(panel, train, H)
    y_te, spy_te = _targets(panel, test, H)

    print(f"stage 0: theta({panel.N}) -> softmax -> w, graded on {len(train)} train anchors")
    print(f"{'':14} {'top holdings':44} {'HHI':>6} {'effN':>6} {'train':>7} {'test':>7}")
    for name in ("mean_excess", "sharpe"):
        torch.manual_seed(cfg.train.seed)
        theta = torch.zeros(panel.N, requires_grad=True)
        opt = torch.optim.Adam([theta], lr=LR)
        loss_fn = make_loss(name, cfg.loss.hhi_lambda)
        for _ in range(STEPS):
            opt.zero_grad()
            w = torch.softmax(theta, dim=-1).unsqueeze(0).expand(len(train), -1)
            loss = loss_fn(w, y_tr, spy_tr)
            assert torch.isfinite(loss), "non-finite stage-0 loss"
            loss.backward()
            opt.step()

        with torch.no_grad():
            w = torch.softmax(theta, dim=-1)
            top = sorted(zip(panel.tickers, w.tolist()), key=lambda kv: -kv[1])[:4]
            top_s = " ".join(f"{t}:{v:.0%}" for t, v in top)
            h = float(hhi(w.unsqueeze(0)))
            wb = w.unsqueeze(0)
            print(
                f"{name:14} {top_s:44} {h:6.3f} {1 / h:6.1f} "
                f"{_sharpe_monthly(wb.expand(len(train), -1), y_tr, spy_tr, H):7.2f} "
                f"{_sharpe_monthly(wb.expand(len(test), -1), y_te, spy_te, H):7.2f}"
            )
    print("\nmean_excess concentrates on the ex-post winner; sharpe+HHI diversifies.")
    print("train Sharpe is in-sample fit, not skill — the test column is the honest one.")
