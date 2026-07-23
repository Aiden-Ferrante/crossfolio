import numpy as np
import pytest

from crossfolio.data.dataset import Panel


@pytest.fixture
def panel() -> Panel:
    """Tiny deterministic synthetic panel: D=500 days, N=8 stocks."""
    rng = np.random.default_rng(7)
    D, N = 500, 8
    return Panel(
        dates=np.datetime64("2020-01-01") + np.arange(D),
        returns=rng.normal(0, 0.01, (D, N)).astype(np.float32),
        spy=rng.normal(0, 0.008, D).astype(np.float32),
        tickers=[f"T{i}" for i in range(N)],
    )
