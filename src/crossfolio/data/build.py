"""The ONLY module that touches the stocklake lake.

Reads per-ticker parquet for the frozen universe + SPY, aligns on the inner
intersection of trading days, computes daily log returns of adj_close, and
writes data/panel.npz. Fail-loud asserts turn silent data problems into errors.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np

from ..config import PANEL, lake
from ..universe import BENCHMARK, N, TICKERS

MAX_GAP_DAYS = 7          # calendar gap between consecutive panel days
                          # (7 legitimately occurs: the post-9/11 closure, 2001-09-10 -> 09-17)
MAX_DROP_FRAC = 0.02      # of SPY trading days lost to the intersection


class PanelError(ValueError):
    """The lake produced a panel that would silently corrupt every result."""


def build_panel(out: Path = PANEL) -> dict:
    LAKE = lake()
    con = duckdb.connect()
    all_tickers = TICKERS + [BENCHMARK]
    files = ", ".join(f"'{LAKE / f'{t}.parquet'}'" for t in all_tickers)
    con.execute(
        f"CREATE VIEW prices AS SELECT date, ticker, adj_close FROM read_parquet([{files}])"
    )
    inception = con.execute(
        "SELECT max(first_bar) FROM (SELECT ticker, min(date) AS first_bar FROM prices GROUP BY ticker)"
    ).fetchone()[0]

    # wide adj_close matrix on the inner intersection of trading days
    cols = ", ".join(
        f'max(adj_close) FILTER (ticker = \'{t}\') AS "{t}"' for t in all_tickers
    )
    wide = con.execute(
        f"""
        SELECT date, {cols}
        FROM prices
        WHERE date >= ?
        GROUP BY date
        HAVING count(DISTINCT ticker) = {len(all_tickers)}
        ORDER BY date
        """,
        [inception],
    ).fetchnumpy()

    spy_total = con.execute(
        "SELECT count(*) FROM prices WHERE ticker = ? AND date >= ?",
        [BENCHMARK, inception],
    ).fetchone()[0]

    dates = wide["date"].astype("datetime64[D]")
    px = np.stack([wide[t].astype(np.float64) for t in all_tickers], axis=1)

    if np.isnan(px).any():
        raise PanelError("NaN adj_close in aligned panel")
    if not (px > 0).all():
        raise PanelError("non-positive adj_close in aligned panel")
    dropped = 1 - len(dates) / spy_total
    if dropped > MAX_DROP_FRAC:
        raise PanelError(
            f"intersection dropped {dropped:.1%} of {BENCHMARK} days (> {MAX_DROP_FRAC:.0%}) — "
            "a universe ticker has holes; re-run freeze_universe or refresh the lake"
        )
    gaps = np.diff(dates).astype(int)
    if gaps.max() > MAX_GAP_DAYS:
        raise PanelError(
            f"gap of {gaps.max()} calendar days in panel at {dates[gaps.argmax()]}"
        )

    logret = np.log(px[1:] / px[:-1]).astype(np.float32)
    dates = dates[1:]

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        dates=dates.astype("datetime64[D]").astype(np.int64),  # epoch days
        returns=logret[:, :N],
        spy=logret[:, N],
        tickers=np.array(TICKERS),
        provenance=np.array(
            f"lake={LAKE} inception={inception} built_rows={len(dates)}"
        ),
    )
    return dict(
        path=str(out), D=len(dates), N=N,
        start=str(dates[0]), end=str(dates[-1]),
        dropped_frac=float(dropped),
    )
