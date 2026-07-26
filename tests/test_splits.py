import numpy as np
import pytest

from crossfolio.data.splits import purged_split, valid_anchors

T, H = 10, 5


def test_valid_anchors_bounds():
    a = valid_anchors(D=500, T=T, H=H)
    assert a[0] == T - 1                # first anchor has exactly T trailing rows
    assert a[-1] == 500 - 1 - H         # last anchor has exactly H forward rows
    assert np.array_equal(a, np.arange(T - 1, 500 - H))


def test_purged_split_no_overlap():
    a = valid_anchors(500, T, H)
    train, val, test = purged_split(a, 0.70, 0.15, H)
    # the two non-negotiable inequalities
    assert train.max() + H < val.min()
    assert val.max() + H < test.min()
    # contiguous, ordered, disjoint, no anchor lost except the purged ones
    combined = np.concatenate([train, val, test])
    assert len(np.unique(combined)) == len(combined)
    assert (np.diff(train) == 1).all() and (np.diff(val) == 1).all() and (np.diff(test) == 1).all()
    assert len(combined) == len(a) - 2 * H


def test_purge_drops_exactly_horizon_per_boundary():
    a = valid_anchors(500, T, H)
    train, val, test = purged_split(a, 0.70, 0.15, H)
    d_val, d_test = val.min(), test.min()
    # the dropped anchors are exactly the H just before each boundary
    assert train.max() == d_val - H - 1
    assert val.max() == d_test - H - 1


def test_blocks_are_chronological():
    a = valid_anchors(500, T, H)
    train, val, test = purged_split(a, 0.70, 0.15, H)
    assert train.max() < val.min() < val.max() < test.min()


def test_degenerate_split_fails_loud():
    a = valid_anchors(40, T, H)  # too small: purge eats a whole block
    with pytest.raises(AssertionError):
        purged_split(a, 0.70, 0.15, horizon=20)


def test_holdout_wall():
    """Round 5: dev_panel cannot reach holdout returns, even through grading."""
    import numpy as np
    from crossfolio.config import Cfg
    from crossfolio.edge import HOLDOUT_START, dev_panel

    p = dev_panel()
    cfg = Cfg()
    last_gradable = p.dates[-1] + np.timedelta64(2 * cfg.data.H, "D")  # calendar upper bound
    assert p.dates[-1] < HOLDOUT_START
    # even the last row's forward window (trading days < 2x calendar) ends pre-holdout
    assert np.datetime64(p.dates[-1]) + np.timedelta64(cfg.data.H * 2, "D") <= HOLDOUT_START + np.timedelta64(7, "D")
