# crossfolio

A neural portfolio allocator built as a learning instrument: the network outputs
an **allocation** (softmax over 120 stocks); **alpha is the grade, not the
output** — the loss computes it from the allocation and the next 21 trading
days of realized returns. There is no return-prediction head, and portfolio
dollar size is never an input (it's a scale factor at reporting time).

Born from one idea — *"what if the network's weights were the dollar
allocations?"* — grown into a ladder:

| stage | architecture | what it teaches |
|---|---|---|
| `stage0` | no input; the allocation vector IS the parameters | optimization wearing NN clothes; the all-in-on-the-ex-post-winner failure mode |
| `linear` | flatten(N×T) → linear → softmax (864k params) | the bag-of-words baseline; guaranteed overfitting vs ~230 effective months |
| `attention` | shared per-stock encoder + self-attention **across stocks** (23k params) | "a stock is a token, the universe at a moment is the sentence"; parameter sharing |

The wider frame: an architecture is a testable hypothesis about how markets
generate returns. This repo implements the first three rows of that program
(null / linear-factor / cross-sectional-relational) on a **shared harness** —
same data, same purged splits, same Sharpe loss — that future architectures
plug into, adjudicated by internal evidence (e.g. do attention heads recover
sector structure?) rather than leaderboard Sharpe.

## Quickstart

```bash
uv sync                                  # torch comes from the cu130 index (see pyproject)
uv run crossfolio build-dataset          # reads the stocklake parquet -> data/panel.npz
uv run crossfolio inspect                # panel + purged split summary
uv run crossfolio stage0                 # the original idea, both losses side by side
uv run crossfolio train --model linear
uv run crossfolio train --model attention
uv run crossfolio evaluate --compare runs/<eq> runs/<linear> runs/<attention>
uv run pytest                            # leakage tests are the correctness core
```

Data comes from the local [stocklake](../stocklake) lake (keyless, yfinance-fed).
The frozen 120-name universe is committed in `src/crossfolio/universe.py`,
generated once by `scripts/freeze_universe.py` (re-run only deliberately —
it redefines the panel). Refresh the lake first:
`cd ../stocklake && .venv/bin/python -m stocklake ingest --tickers <universe> SPY`.

## The correctness core

Daily-anchored examples overlap heavily (adjacent anchors share 59/60 input
days), so **random train/test splits leak catastrophically**. Splits are
chronological blocks with a 21-trading-day purge/embargo at each boundary
(`data/splits.py` asserts its own invariants at runtime), and
`tests/test_dataset.py` contains the probe that matters: poison every panel row
after the anchor and assert features are unchanged.

## Honest caveats — read before interpreting any number

- **No edge.** ~48 independent monthly test points ⇒ Sharpe standard error
  ≈ ±0.5. Every model here is a statistical tie with SPY and with equal-weight.
- **The headline finding is negative and real:** early stopping keeps the
  *untrained* weights for both learned models — nothing learnable from
  2000–2018 transferred to later blocks at this data budget. The `last.pt`
  checkpoint (trained/overfit) is saved alongside `best.pt` for interp
  comparisons.
- **Survivorship bias**: the universe is today's lake ranked by today's
  liquidity; every backtest row is inflated by it, roughly equally.
- **Spinoff distortion**: yfinance adj_close handles splits+dividends, not
  spinoff value (GE breakup etc.).
- **Batch-Sharpe** mixes market regimes within a shuffled batch — its std is
  cross-regime, not sequential; loss values are batch-size dependent.
- Long-only, fully invested, no transaction costs, no shorting, frozen universe.

## Repo map

See [ARCHITECTURE.md](ARCHITECTURE.md). One-line version: `config ← data ←
{models, losses} ← train ← evaluate/plots ← cli`; only `data/build.py` touches
the lake; only `train.py`/`evaluate.py` write `runs/`.
