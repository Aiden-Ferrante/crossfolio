"""Synthetic panel generator with a planted cross-sectional momentum signal.

The power-test instrument: generate returns whose null looks like real equities
(market + sector + idio, cross-correlated), plant a signal of known strength
that is observable from the model's own input (trailing 60-day returns), and
measure whether the harness recovers it.

Paired design (common random numbers): draw base shocks ONCE per seed with
`make_shocks`, then build one panel per gamma from the SAME shocks — differences
across gamma are signal-only, not panel lottery.

The planted drift feeds back into future trailing windows (momentum begets
momentum); loop gain at the top gamma is ~0.16, well below 1, so it's a mild
realistic persistence boost, not explosive. It is cross-sectionally ~mean-zero,
so the equal-weight benchmark is unaffected.
"""

from __future__ import annotations

import numpy as np

from .data.dataset import Panel

# daily vol constants (fractions): ~1.9% total daily, ~8.7% monthly
MARKET_VOL = 0.010
SECTOR_VOL = 0.005
IDIO_VOL = 0.015
BETA_RANGE = (0.8, 1.2)
N_SECTORS = 10


def make_shocks(D: int, N: int, seed: int, n_sectors: int = N_SECTORS) -> dict:
    rng = np.random.default_rng(seed)
    sector_id = np.arange(N) % n_sectors
    rng.shuffle(sector_id)
    return {
        "m": rng.normal(0, MARKET_VOL, D),
        "beta": rng.uniform(*BETA_RANGE, N),
        "sector_id": sector_id,
        "s": rng.normal(0, SECTOR_VOL, (D, n_sectors)),
        "eps": rng.normal(0, IDIO_VOL, (D, N)),
        "seed": seed,
    }


def make_panel(gamma_bps: float, shocks: dict, T: int = 60) -> tuple[Panel, dict]:
    """Build a panel from shared shocks with planted momentum of strength
    `gamma_bps` (expected monthly drift, in bps, per 1-sigma of the trailing
    cross-sectional momentum z-score)."""
    m, beta, sector_id = shocks["m"], shocks["beta"], shocks["sector_id"]
    base = beta * m[:, None] + shocks["s"][:, sector_id] + shocks["eps"]
    D, N = base.shape
    gamma_daily = gamma_bps * 1e-4 / 21

    r = np.empty_like(base)
    cum = np.zeros((D + 1, N))  # cum[t] = sum of r[0:t]
    for t in range(D):
        if gamma_daily and t >= T:
            win = cum[t] - cum[t - T]  # trailing T-day cum log return through t-1
            z = (win - win.mean()) / (win.std() + 1e-12)
            r[t] = base[t] + gamma_daily * z
        else:
            r[t] = base[t]
        cum[t + 1] = cum[t] + r[t]

    # true signal AS THE MODEL SEES IT: z of the window ENDING at day d
    z_anchor = np.full((D, N), np.nan)
    for d in range(T - 1, D):
        win = cum[d + 1] - cum[d + 1 - T]
        z_anchor[d] = (win - win.mean()) / (win.std() + 1e-12)

    tickers = [f"X{i:03d}" for i in range(N)]
    panel = Panel(
        dates=np.datetime64("2000-01-03") + np.arange(D),
        returns=r.astype(np.float32),
        # daily-rebalanced EW portfolio: keeps excess-vs-EW nonzero at uniform
        # weights (buy-and-hold EW mean would zero it and NaN the Sharpe grad)
        spy=np.log(np.exp(r).mean(axis=1)).astype(np.float32),
        tickers=tickers,
    )
    meta = {
        "gamma_bps": gamma_bps,
        "seed": shocks["seed"],
        "z": z_anchor,
        "sectors": {t: f"S{g}" for t, g in zip(tickers, sector_id)},
    }
    return panel, meta


def forward_returns(panel: Panel, anchors: np.ndarray, H: int) -> tuple[np.ndarray, np.ndarray]:
    """(n, N) forward H-day simple returns per anchor, and (n,) for the benchmark."""
    cum = np.cumsum(panel.returns.astype(np.float64), axis=0)
    cum_spy = np.cumsum(panel.spy.astype(np.float64))
    y = np.expm1(cum[anchors + H] - cum[anchors])
    y_spy = np.expm1(cum_spy[anchors + H] - cum_spy[anchors])
    return y, y_spy


def cross_sectional_ic(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-anchor Pearson corr across the N axis. scores, y: (n, N) -> (n,)."""
    s = scores - scores.mean(axis=1, keepdims=True)
    t = y - y.mean(axis=1, keepdims=True)
    num = (s * t).sum(axis=1)
    den = np.sqrt((s * s).sum(axis=1) * (t * t).sum(axis=1)) + 1e-12
    return num / den


def oracle_ic(panel: Panel, meta: dict, anchors: np.ndarray, H: int) -> np.ndarray:
    """Empirical ceiling: per-anchor IC of the TRUE planted z vs realized
    forward returns. Measured, not derived — the feedback loop shifts the
    analytic value."""
    y, _ = forward_returns(panel, anchors, H)
    return cross_sectional_ic(meta["z"][anchors], y)
