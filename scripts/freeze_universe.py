"""One-time universe selection: scan the stocklake parquet lake, pick the top
liquid equities with full history since 2000, and write the frozen list (with
sector labels) into src/crossfolio/universe.py, which is committed.

Dev-time script. Run with stocklake's venv (it has duckdb + yfinance + pandas):

    ~/Desktop/code/stocklake/.venv/bin/python scripts/freeze_universe.py --candidates
    # refresh the printed tickers via stocklake ingest, then:
    ~/Desktop/code/stocklake/.venv/bin/python scripts/freeze_universe.py --freeze

The crossfolio runtime never imports this; yfinance is allowed here only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

import duckdb

LAKE = Path("~/Desktop/code/stocklake/lake/raw/prices").expanduser()
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "src" / "crossfolio" / "universe.py"
SECTOR_CACHE = REPO / "data" / "sector_cache.json"

INCEPTION_CUTOFF = "2000-01-03"
MIN_COVERAGE = 0.98      # fraction of SPY trading days since cutoff
FRESHNESS_DAYS = 7       # last bar must be within this many days of lake max
TOP_K = 120              # buffer above the 100-minimum

# Not equities: indices, FX, crypto, futures, foreign listings.
NON_EQUITY_PATTERNS = [r"^\^", r"=X$", r"-USD$", r"=F$", r"\.\w+$"]
# Equities-only universe: exclude funds explicitly (lake mixes them in).
ETF_DENYLIST = {
    "SPY", "QQQ", "DIA", "IWM", "MDY", "RSP", "VTI", "VOO", "IVV", "IJH", "IJR",
    "GLD", "SLV", "USO", "UNG", "TLT", "HYG", "LQD", "AGG", "BND", "SHY", "IEF",
    "EEM", "EFA", "EWJ", "EWZ", "FXI", "VNQ", "IYR", "GDX", "SMH", "SOXX", "IBB",
    "XBI", "KRE", "ARKK", "VXX", "UVXY", "SQQQ", "TQQQ", "XLB", "XLC", "XLE",
    "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
}


def scan(con: duckdb.DuckDBPyConnection):
    """Return (rows, lake_max_date, spy_days). rows = per-ticker stats passing
    the equity/inception/coverage filters, ranked by median dollar volume."""
    con.execute(
        f"CREATE OR REPLACE VIEW prices AS SELECT * FROM read_parquet('{LAKE}/*.parquet')"
    )
    lake_max = con.execute("SELECT max(date) FROM prices").fetchone()[0].date()
    spy_days = con.execute(
        "SELECT count(*) FROM prices WHERE ticker='SPY' AND date >= ?",
        [INCEPTION_CUTOFF],
    ).fetchone()[0]

    rows = con.execute(
        """
        WITH stats AS (
            SELECT
                ticker,
                min(date)                                    AS first_bar,
                max(date)                                    AS last_bar,
                count(*) FILTER (date >= CAST(? AS DATE))    AS bars_since,
                median(adj_close * volume)
                    FILTER (date >= CAST(? AS DATE) - INTERVAL 5 YEAR) AS med_dollar_vol
            FROM prices
            GROUP BY ticker
        )
        SELECT ticker, first_bar, last_bar, bars_since, med_dollar_vol
        FROM stats
        WHERE first_bar <= ?
        ORDER BY med_dollar_vol DESC
        """,
        [INCEPTION_CUTOFF, str(lake_max), INCEPTION_CUTOFF],
    ).fetchall()

    out = []
    for ticker, first_bar, last_bar, bars_since, mdv in rows:
        if ticker in ETF_DENYLIST:
            continue
        if any(re.search(p, ticker) for p in NON_EQUITY_PATTERNS):
            continue
        if bars_since < MIN_COVERAGE * spy_days:
            continue
        if mdv is None:
            continue
        out.append(
            dict(ticker=ticker, first_bar=str(first_bar.date()),
                 last_bar=str(last_bar.date()), med_dollar_vol=float(mdv))
        )
    return out, lake_max, spy_days


def fetch_sectors(tickers: list[str]) -> dict[str, str]:
    import yfinance as yf

    cache: dict[str, str] = {}
    if SECTOR_CACHE.exists():
        cache = json.loads(SECTOR_CACHE.read_text())
    missing = [t for t in tickers if t not in cache]
    for i, t in enumerate(missing):
        try:
            sector = yf.Ticker(t).info.get("sector") or "UNKNOWN"
        except Exception as e:  # noqa: BLE001 - any network flake -> UNKNOWN, retryable via cache
            print(f"  sector fetch failed for {t}: {e}", file=sys.stderr)
            sector = "UNKNOWN"
        cache[t] = sector
        if (i + 1) % 10 == 0:
            print(f"  sectors {i + 1}/{len(missing)}")
        SECTOR_CACHE.parent.mkdir(parents=True, exist_ok=True)
        SECTOR_CACHE.write_text(json.dumps(cache, indent=1, sort_keys=True))
        time.sleep(0.5)
    return {t: cache[t] for t in tickers}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--candidates", action="store_true",
                      help="print top-K candidate tickers (ignores freshness) for lake refresh")
    mode.add_argument("--freeze", action="store_true",
                      help="apply all filters incl. freshness, fetch sectors, write universe.py")
    args = ap.parse_args()

    con = duckdb.connect()
    rows, lake_max, spy_days = scan(con)

    if args.candidates:
        cands = [r["ticker"] for r in rows[: TOP_K + 20]]
        print(" ".join(cands))
        print(f"# {len(cands)} candidates; lake max date {lake_max}; SPY days since cutoff {spy_days}",
              file=sys.stderr)
        return

    fresh = [r for r in rows
             if (lake_max - date.fromisoformat(r["last_bar"])).days <= FRESHNESS_DAYS]
    dropped = len(rows) - len(fresh)
    selected = fresh[:TOP_K]
    if len(selected) < 100:
        sys.exit(f"only {len(selected)} tickers pass all filters (need >= 100); "
                 f"{dropped} dropped for staleness — refresh the lake first")

    tickers = [r["ticker"] for r in selected]
    print(f"selected {len(tickers)} tickers ({dropped} candidates dropped as stale); fetching sectors...")
    sectors = fetch_sectors(tickers)

    entries = "\n".join(
        f'    ("{r["ticker"]}", "{sectors[r["ticker"]]}"),' for r in selected
    )
    OUT.write_text(f'''"""FROZEN universe — generated by scripts/freeze_universe.py, do not edit by hand.

Selection (frozen {date.today()}, lake max date {lake_max}):
  equities only (index/FX/crypto/futures patterns + ETF denylist excluded),
  first bar <= {INCEPTION_CUTOFF}, coverage >= {MIN_COVERAGE:.0%} of SPY trading days
  since then, last bar within {FRESHNESS_DAYS} days of lake max,
  top {TOP_K} by median daily dollar volume (adj_close * volume) over the trailing 5y.

Survivorship bias is baked in: membership reflects today's lake and today's
liquidity. Fine for a learning project; fatal if forgotten when interpreting.
"""

BENCHMARK = "SPY"

# (ticker, sector) — column order of every panel/tensor in this repo.
UNIVERSE: list[tuple[str, str]] = [
{entries}
]

TICKERS: list[str] = [t for t, _ in UNIVERSE]
SECTORS: dict[str, str] = dict(UNIVERSE)
N: int = len(UNIVERSE)
''')
    print(f"wrote {OUT} with N={len(selected)}")


if __name__ == "__main__":
    main()
