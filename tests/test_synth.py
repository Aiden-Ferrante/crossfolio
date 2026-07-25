import numpy as np
import torch

from crossfolio.config import DataCfg
from crossfolio.data.dataset import AnchorDataset, Panel
from crossfolio.data.splits import valid_anchors
from crossfolio.synth import (
    IDIO_VOL,
    MARKET_VOL,
    SECTOR_VOL,
    make_panel,
    make_shocks,
    oracle_ic,
)

D, N, T, H = 1500, 120, 60, 21


def _panel(gamma, seed=0):
    return make_panel(gamma, make_shocks(D, N, seed), T=T)


def test_deterministic_by_seed():
    p1, _ = _panel(40, seed=3)
    p2, _ = _panel(40, seed=3)
    np.testing.assert_array_equal(p1.returns, p2.returns)
    p3, _ = _panel(40, seed=4)
    assert not np.array_equal(p1.returns, p3.returns)


def test_paired_design_shares_noise():
    """Same seed, different gamma: returns differ only via the small drift."""
    p0, _ = _panel(0, seed=1)
    p80, _ = _panel(80, seed=1)
    diff = np.abs(p80.returns - p0.returns)
    assert diff.max() > 0
    assert diff.max() < 0.01  # drift + feedback is bps-scale, noise is shared


def test_vol_calibration():
    p, _ = _panel(0)
    expected = np.sqrt((1.0 * MARKET_VOL) ** 2 + SECTOR_VOL**2 + IDIO_VOL**2)
    assert abs(p.returns.std() - expected) < 0.002


def test_null_oracle_ic_is_zero():
    p, meta = _panel(0)
    anchors = valid_anchors(p.D, T, H)[::H]  # non-overlapping
    ic = oracle_ic(p, meta, anchors, H)
    # SE of mean IC ~ (1/sqrt(N))/sqrt(n_anchors)
    se = (1 / np.sqrt(N)) / np.sqrt(len(ic))
    assert abs(ic.mean()) < 3 * se


def test_oracle_ic_grows_with_gamma():
    anchors = None
    means = []
    for gamma in (0, 20, 80):
        p, meta = _panel(gamma, seed=2)
        anchors = valid_anchors(p.D, T, H)[::H]
        means.append(oracle_ic(p, meta, anchors, H).mean())
    assert means[0] < means[1] < means[2]
    assert means[2] > 0.05  # 80 bps is unambiguous


def test_synthetic_panel_passes_leakage_probe():
    """The perturb-post-anchor probe holds on synthetic panels too."""
    p, _ = _panel(40)
    cfg = DataCfg(T=T, H=H, normalize=False)
    anchors = valid_anchors(p.D, T, H)
    ds = AnchorDataset(p, anchors, cfg)
    i = len(anchors) // 2
    d = int(anchors[i])
    X_before, y_before, _ = ds[i]
    poisoned = Panel(p.dates, p.returns.copy(), p.spy.copy(), p.tickers)
    poisoned.returns[d + 1 :] = 9.9
    X_after, y_after, _ = AnchorDataset(poisoned, anchors, cfg)[i]
    assert torch.equal(X_before, X_after)
    assert not torch.equal(y_before, y_after)


def test_time_axis_normalization_erases_signal():
    """Regression test for the design-review catch: per-window TIME-axis
    z-scoring removes the trailing-return level the planted z is built from."""
    p, meta = _panel(80)
    anchors = valid_anchors(p.D, T, H)
    d = int(anchors[500])
    cfg_norm = DataCfg(T=T, H=H, normalize=True)
    X, _, _ = AnchorDataset(p, anchors, cfg_norm)[500]
    # every stock's window mean is forced to ~0: the momentum level is gone
    assert float(X.mean(dim=1).abs().max()) < 1e-4
    cfg_raw = DataCfg(T=T, H=H, normalize=False)
    X_raw, _, _ = AnchorDataset(p, anchors, cfg_raw)[500]
    win_sums = X_raw.sum(dim=1).numpy()
    z = meta["z"][d]
    corr = np.corrcoef(win_sums, z)[0, 1]
    assert corr > 0.99  # unnormalized input carries the true signal directly


def test_relational_oracle_gap():
    """Full oracle sees the signal; an own-window-only ridge is far weaker."""
    from crossfolio.synth import cross_sectional_ic, forward_returns, make_relational_panel

    p, meta = make_relational_panel(120, make_shocks(D, N, seed=5))
    anchors = valid_anchors(p.D, T, H)[::H]
    y, _ = forward_returns(p, anchors, H)
    full = cross_sectional_ic(meta["z"][anchors], y).mean()
    # own-window features: trailing cum returns at several lookbacks
    cum = np.cumsum(p.returns.astype(np.float64), 0)
    feats = np.stack([cum[anchors] - cum[anchors - l] for l in (5, 10, 20, 60)], -1)
    half = len(anchors) // 2
    Z = feats[:half].reshape(-1, 4); Yt = y[:half].reshape(-1, 1)
    W = np.linalg.solve(Z.T @ Z + 1e-2 * np.eye(4), Z.T @ Yt)
    pred = (feats[half:].reshape(-1, 4) @ W).reshape(len(anchors) - half, N)
    p7 = cross_sectional_ic(pred, y[half:]).mean()
    assert full > 0.05                       # relational signal is strong at 120 bps
    assert p7 < 0.5 * full                   # own-window leak is a minor fraction


def test_curriculum_ladder_is_paired():
    from crossfolio.synth import make_relational_panel

    sh = make_shocks(400, N, seed=9)
    p_hi, _ = make_relational_panel(500, sh)
    p_lo, _ = make_relational_panel(120, sh)
    diff = np.abs(p_hi.returns - p_lo.returns)
    assert diff.max() > 0 and diff.max() < 0.05  # same shocks, drift-scale only
