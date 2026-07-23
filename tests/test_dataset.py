import dataclasses

import numpy as np
import torch

from crossfolio.config import DataCfg
from crossfolio.data.dataset import AnchorDataset, Panel
from crossfolio.data.splits import valid_anchors

CFG = DataCfg(T=10, H=5, normalize=True)


def _ds(panel):
    anchors = valid_anchors(panel.D, CFG.T, CFG.H)
    return AnchorDataset(panel, anchors, CFG), anchors


def test_shapes_and_dtypes(panel):
    ds, _ = _ds(panel)
    X, y, y_spy = ds[0]
    assert X.shape == (panel.N, CFG.T) and X.dtype == torch.float32
    assert y.shape == (panel.N,) and y.dtype == torch.float32
    assert y_spy.shape == () and y_spy.dtype == torch.float32


def test_no_lookahead_probe(panel):
    """THE leakage test: garbage after the anchor must not change X, must change y."""
    ds, anchors = _ds(panel)
    i = len(anchors) // 2
    d = int(anchors[i])
    X_before, y_before, spy_before = ds[i]

    poisoned = Panel(
        dates=panel.dates,
        returns=panel.returns.copy(),
        spy=panel.spy.copy(),
        tickers=panel.tickers,
    )
    poisoned.returns[d + 1 :] = 9.9
    poisoned.spy[d + 1 :] = 9.9
    ds_p = AnchorDataset(poisoned, anchors, CFG)
    X_after, y_after, spy_after = ds_p[i]

    assert torch.equal(X_before, X_after), "features leaked post-anchor data"
    assert not torch.equal(y_before, y_after), "targets ignored post-anchor data"
    assert spy_before != spy_after


def test_window_is_trailing_inclusive(panel):
    """X's last column is day d itself; first column is day d-T+1."""
    cfg = dataclasses.replace(CFG, normalize=False)
    anchors = valid_anchors(panel.D, cfg.T, cfg.H)
    ds = AnchorDataset(panel, anchors, cfg)
    i, d = 3, int(anchors[3])
    X, _, _ = ds[i]
    np.testing.assert_allclose(X[:, -1].numpy(), panel.returns[d], rtol=1e-6)
    np.testing.assert_allclose(X[:, 0].numpy(), panel.returns[d - cfg.T + 1], rtol=1e-6)


def test_targets_are_forward_simple_returns(panel):
    cfg = dataclasses.replace(CFG, normalize=False)
    anchors = valid_anchors(panel.D, cfg.T, cfg.H)
    ds = AnchorDataset(panel, anchors, cfg)
    i, d = 0, int(anchors[0])
    _, y, y_spy = ds[i]
    expected = np.expm1(panel.returns[d + 1 : d + 1 + cfg.H].sum(axis=0))
    np.testing.assert_allclose(y.numpy(), expected, rtol=1e-5)
    np.testing.assert_allclose(
        float(y_spy), np.expm1(panel.spy[d + 1 : d + 1 + cfg.H].sum()), rtol=1e-5
    )


def test_normalization_uses_window_only(panel):
    """Perturbing data OUTSIDE the window must not change normalized X."""
    ds, anchors = _ds(panel)
    i = 10
    d = int(anchors[i])
    X_before, _, _ = ds[i]
    poisoned = Panel(
        dates=panel.dates,
        returns=panel.returns.copy(),
        spy=panel.spy,
        tickers=panel.tickers,
    )
    poisoned.returns[: d - CFG.T + 1] = 5.5   # before the window
    ds_p = AnchorDataset(poisoned, anchors, CFG)
    X_after, _, _ = ds_p[i]
    assert torch.equal(X_before, X_after)
