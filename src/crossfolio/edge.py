"""Round 5: net-of-costs edge evaluation under pre-registered discipline.

The holdout wall: development code loads panels ONLY through dev_panel(),
which truncates the return history so that no development anchor's grading
window can reach the holdout period. scripts/holdout_once.py is the single
permitted reader of the full panel, run once, in Phase 3.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .config import REPO, Cfg
from .data.dataset import Panel, load_panel

HOLDOUT_START = np.datetime64("2024-07-26")   # final ~2y of the panel
COST_BPS_HEADLINE = 10.0                       # each way, on turnover
COST_BPS_SENSITIVITY = (5.0, 20.0)
LEDGER = REPO / "docs" / "edge" / "ledger.jsonl"


def dev_panel() -> Panel:
    """The ONLY panel loader development code may use: truncated so the last
    row is strictly before HOLDOUT_START minus the grading horizon's reach."""
    p = load_panel()
    cfg = Cfg()
    cut = int(np.searchsorted(p.dates, HOLDOUT_START)) - cfg.data.H
    assert cut > 0, "holdout boundary outside panel"
    return Panel(p.dates[:cut], p.returns[:cut], p.spy[:cut], p.tickers)


def net_monthly_excess(weights: np.ndarray, y: np.ndarray, y_spy: np.ndarray,
                       dates: np.ndarray, cost_bps: float,
                       smooth_alpha: float = 1.0) -> dict:
    """Weekly-rebalanced net performance. weights (A, N) at stride-H anchors,
    y (A, N) forward simple returns, dates (A,). smooth_alpha < 1 EMA-smooths
    positions (turnover control); costs = cost_bps * sum|delta w| per rebalance.
    Monthly = calendar-month sums of weekly net excess (approximation stated
    in PREREGISTRATION.md)."""
    w = weights.copy()
    for t in range(1, len(w)):
        w[t] = smooth_alpha * w[t] + (1 - smooth_alpha) * w[t - 1]
        w[t] /= w[t].sum()
    turn = np.abs(np.diff(w, axis=0)).sum(1)
    turn = np.r_[w[0:1].sum() * 0, turn]                  # no cost at t=0
    net = (w * y).sum(1) - y_spy - cost_bps * 1e-4 * turn
    months = dates.astype("datetime64[M]")
    uniq = np.unique(months)
    m = np.array([net[months == mo].sum() for mo in uniq])
    return {"net_sharpe_ann": float(m.mean() / (m.std(ddof=1) + 1e-12) * np.sqrt(12)),
            "gross_sharpe_ann": float(((weights * y).sum(1) - y_spy)[months.argsort()].mean()
                                      / ((weights * y).sum(1) - y_spy).std(ddof=1) * np.sqrt(52)),
            "mean_weekly_turnover": float(turn[1:].mean()),
            "n_months": int(len(m)),
            "monthly": m.tolist()}


def log_trial(entry: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **entry}
    with LEDGER.open("a") as f:
        f.write(json.dumps(entry) + "\n")
