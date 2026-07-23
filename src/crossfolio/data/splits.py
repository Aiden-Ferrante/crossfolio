"""Anchor enumeration and the purged chronological split — the correctness core.

An anchor d is graded on days d+1 .. d+H. Random shuffling across time blocks
leaks those grading windows and is the one failure mode this project must not
have; the split therefore purges every anchor whose window crosses a boundary
(embargo = H) and asserts its own invariants at runtime, not only in tests.

Deliberate asymmetry: a val/test anchor's INPUT window reaching back into the
prior block is fine (that's past data); only forward grading windows are purged.
"""

from __future__ import annotations

import numpy as np


def valid_anchors(D: int, T: int, H: int) -> np.ndarray:
    """Panel row indices d with T trailing returns (rows d-T+1..d) and H forward rows."""
    return np.arange(T - 1, D - H, dtype=np.int64)


def purged_split(
    anchors: np.ndarray, train_frac: float, val_frac: float, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological train/val/test over `anchors`, purging each boundary.

    The cut points are taken on the anchor array; an anchor survives its block
    only if its grading window [d+1, d+horizon] ends before the next block
    starts: d + horizon < first anchor day of the next block.
    """
    n = len(anchors)
    d_val = anchors[int(n * train_frac)]
    d_test = anchors[int(n * (train_frac + val_frac))]

    train = anchors[(anchors < d_val) & (anchors + horizon < d_val)]
    val = anchors[(anchors >= d_val) & (anchors < d_test) & (anchors + horizon < d_test)]
    test = anchors[anchors >= d_test]

    assert len(train) and len(val) and len(test), "empty split block"
    assert train.max() + horizon < val.min(), "train grading window reaches into val"
    assert val.max() + horizon < test.min(), "val grading window reaches into test"
    # anchors from valid_anchors() are consecutive integers, so each boundary
    # drops exactly `horizon` anchors; pass consecutive anchors only.
    assert len(train) + len(val) + len(test) == n - 2 * horizon, (
        "purge should drop exactly `horizon` anchors per boundary (consecutive anchors required)"
    )
    return train, val, test
