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


# ---------------------------------------------------------------------------
# Sim2real pretraining: randomized "cocktail" panels (Stage C).
# Signals keep fixed economic signs (+momentum, -reversal) — a per-anchor model
# cannot infer a panel-level latent sign from one snapshot, so free sign
# randomization would collapse the optimal output to zero. In-context inference
# is forced instead by regime-conditioned strengths, random lookbacks/subsets,
# sector-neutral-or-raw z, distractor structure, and a beta "factor trap".
# Stateless-by-index: panel i is fully determined by rng([base_seed, i]).
# ---------------------------------------------------------------------------

def make_cocktail_panel(index: int, base_seed: int, N: int, D: int = 1500) -> Panel:
    rng = np.random.default_rng([base_seed, index])
    n_sectors = int(rng.integers(5, 21))
    sector_id = np.arange(N) % n_sectors
    rng.shuffle(sector_id)
    beta = rng.uniform(0.8, 1.2, N)
    mkt_vol, sec_vol, idio_vol = (v * rng.uniform(0.5, 1.5) for v in
                                  (MARKET_VOL, SECTOR_VOL, IDIO_VOL))

    # vol regimes: 2-4 segments, alternating calm/stressed
    n_seg = int(rng.integers(2, 5))
    bounds = np.sort(rng.integers(D // 8, D, n_seg - 1))
    vol_mult = np.ones(D)
    for k, (a, b) in enumerate(zip(np.r_[0, bounds], np.r_[bounds, D])):
        vol_mult[a:b] = 1.0 if k % 2 == 0 else rng.uniform(1.5, 2.5)

    t_shocks = rng.standard_t(4, (D, N)) / np.sqrt(2.0)  # unit-variance t(4)
    base = (beta * rng.normal(0, mkt_vol, D)[:, None]
            + rng.normal(0, sec_vol, (D, n_sectors))[:, sector_id]
            + idio_vol * vol_mult[:, None] * t_shocks)

    # signal cocktail (25% mass at zero each)
    g_mom = rng.uniform(0, 40) * (rng.random() > 0.25) * 1e-4 / 21
    g_rev = rng.uniform(0, 40) * (rng.random() > 0.25) * 1e-4 / 21
    L_mom, gap = int(rng.integers(20, 61)), int(rng.integers(0, 11))
    L_rev = int(rng.integers(3, 11))
    subset = rng.random(N) < rng.uniform(0.5, 1.0)          # who carries the signal
    sector_neutral = bool(rng.random() < 0.5)               # z within sector vs raw
    # factor trap: beta-correlated drift, sign flips across panels — harvestable
    # in-panel, poison across panels; the model must learn to neutralize beta
    g_trap = rng.uniform(0, 20) * rng.choice([-1.0, 1.0]) * 1e-4 / 21
    trap = g_trap * (beta - beta.mean()) / (beta.std() + 1e-12)
    stale = rng.random(N) < 0.02                            # ~2% stale-price stocks

    def _z(win: np.ndarray) -> np.ndarray:
        if sector_neutral:
            z = np.empty_like(win)
            for s in range(n_sectors):
                m = sector_id == s
                z[m] = (win[m] - win[m].mean()) / (win[m].std() + 1e-12)
            return z
        return (win - win.mean()) / (win.std() + 1e-12)

    r = np.empty_like(base)
    cum = np.zeros((D + 1, N))
    warm = max(L_mom + gap, L_rev) + 1
    for t in range(D):
        drift = trap.copy()
        if t >= warm:
            stressed = vol_mult[t] > 1.2
            if g_mom and not stressed:                       # momentum dies in stress
                win = cum[t - gap] - cum[t - gap - L_mom]
                drift = drift + g_mom * _z(win) * subset
            if g_rev:                                        # reversal grows with vol
                win = cum[t] - cum[t - L_rev]
                drift = drift - g_rev * vol_mult[t] * _z(win) * subset
        r[t] = np.where(stale & (rng.random(N) < 0.3), 0.0, base[t] + drift)
        cum[t + 1] = cum[t] + r[t]

    return Panel(
        dates=np.datetime64("2000-01-03") + np.arange(D),
        returns=r.astype(np.float32),
        spy=np.log(np.exp(r).mean(axis=1)).astype(np.float32),
        tickers=[f"X{i:03d}" for i in range(N)],
    )


def make_relational_panel(gamma_bps: float, shocks: dict, L: int = 20) -> tuple[Panel, dict]:
    """Purely-relational planted signal: stock i drifts toward its SECTOR-MATES'
    (excluding self) mean trailing-L return, cross-sectionally z-scored. A
    per-stock path sees the signal only through the shared sector factor; the
    two-oracle machinery in power.py measures that leak instead of assuming it
    away. Same paired-shock discipline as make_panel."""
    m, beta, sector_id = shocks["m"], shocks["beta"], shocks["sector_id"]
    base = beta * m[:, None] + shocks["s"][:, sector_id] + shocks["eps"]
    D, N = base.shape
    g = gamma_bps * 1e-4 / 21
    n_sec = sector_id.max() + 1
    counts = np.bincount(sector_id, minlength=n_sec).astype(np.float64)

    r = np.empty_like(base)
    cum = np.zeros((D + 1, N))
    z_anchor = np.full((D, N), np.nan)

    def mates_z(row_hi: int) -> np.ndarray:
        win = cum[row_hi] - cum[row_hi - L]
        sec_sum = np.bincount(sector_id, weights=win, minlength=n_sec)
        mates = (sec_sum[sector_id] - win) / np.maximum(counts[sector_id] - 1, 1)
        return (mates - mates.mean()) / (mates.std() + 1e-12)

    for t in range(D):
        r[t] = base[t] + (g * mates_z(t) if (g and t >= L) else 0.0)
        cum[t + 1] = cum[t] + r[t]
    for d in range(L - 1, D):
        z_anchor[d] = mates_z(d + 1)

    tickers = [f"X{i:03d}" for i in range(N)]
    panel = Panel(
        dates=np.datetime64("2000-01-03") + np.arange(D),
        returns=r.astype(np.float32),
        spy=np.log(np.exp(r).mean(axis=1)).astype(np.float32),
        tickers=tickers,
    )
    return panel, {"gamma_bps": gamma_bps, "seed": shocks["seed"], "z": z_anchor,
                   "sectors": {t: f"S{s}" for t, s in zip(tickers, sector_id)}}
