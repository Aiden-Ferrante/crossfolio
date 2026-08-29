# crossfolio

A neural portfolio allocator built as a learning instrument: the network outputs
an **allocation** (softmax over the 405-name universe); **alpha is the grade, not
the output** — the loss computes it from the allocation and the next 5 trading
days of realized returns. There is no return-prediction head, and portfolio
dollar size is never an input (it's a scale factor at reporting time).

Current config (`src/crossfolio/config.py`, campaign v2): **N = 405** names,
**T = 120**-day trailing window, **H = 5** (weekly grading), window
normalization off. Campaign v1 ran N = 120 / T = 60 / H = 21; older reports
under `docs/power/` were produced at those settings.

Born from one idea — *"what if the network's weights were the dollar
allocations?"* — grown into a ladder. Parameter counts at the current config:

| stage | architecture | params | what it teaches |
|---|---|---|---|
| `stage0` | no input; the allocation vector IS the parameters | 405 | optimization wearing NN clothes; the all-in-on-the-ex-post-winner failure mode |
| `linear` | flatten(N×T) → linear → softmax | 19.7M | the bag-of-words baseline; guaranteed overfitting vs a few hundred effective months |
| `attention` | shared per-stock encoder + self-attention **across stocks** | 36k | "a stock is a token, the universe at a moment is the sentence"; parameter sharing |
| `p7` | the same backbone with **no** cross-stock interaction | 18k | the probe verdict: attention never earned its place (Round 3) |

Parameter sharing is the counter-lesson to the linear baseline: 36k shared
weights learn what 19.7M unshared ones cannot, and 18k of attention-free
encoder beat both.

The wider frame: an architecture is a testable hypothesis about how markets
generate returns. Every arm runs on a **shared harness** — same data, same
purged splits, same loss — and is adjudicated by internal evidence (do
attention heads recover sector structure?) rather than leaderboard Sharpe.

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

**On reproducibility:** the price panel is *not* included — `data/` and `runs/`
are gitignored, and the daily bars come from `stocklake`, a separate (private,
keyless, yfinance-fed) local lake of mine. What *is* committed is everything
needed to check the reasoning rather than re-run it: the frozen 405-name
universe (`src/crossfolio/universe.py`, generated once by
`scripts/freeze_universe.py`), the pre-registrations, the full 80-row trial
ledger, and the results files behind the tables below. Substituting any daily
adjusted-close source for the same tickers reproduces the panel;
`build-dataset` is the only code that touches the lake.

Note that per-run artifacts in `runs/` are **not** kept, so tables here cite
the committed reports under `docs/`, which are the durable record.

## The correctness core

Daily-anchored examples overlap heavily (adjacent anchors share most of their
input window), so **random train/test splits leak catastrophically**. Splits are
chronological blocks with an H-anchor purge/embargo at each boundary
(`data/splits.py` asserts at runtime that no grading window crosses a boundary,
and that each boundary drops exactly H anchors), and `tests/test_dataset.py`
contains the probe that matters: poison every panel row after the anchor and
assert features are unchanged.

Development code may load the panel only through `edge.dev_panel()`, which
truncates it so no development anchor's grading window can reach the holdout.
`scripts/holdout_once.py` is the single permitted reader of the full panel.

## Honest caveats — read before interpreting any number

- **No demonstrated edge.** The best holdout result is +0.94 net excess Sharpe
  at t = 1.35 over 25 months — suggestive, not significant. See the verdicts
  below.
- **A zero-parameter baseline is competitive with every model here.** Plain
  5-day reversal scores +0.913 net @10bps against the best learned arm's
  +0.856 (`docs/edge/ledger.jsonl`). It is a single unconfirmed run, disclosed
  as measured before its pre-registration — but no learned result should be
  read without it.
- **Attention is causally harmful to signal extraction** on this data: flattening
  it to uniform *improves* detection (`docs/probes/PROBES.md`). Deleting it
  outright reproduced the whole sim2real campaign's gain for free
  (`docs/power/v3/ROUND3.md`).
- **Pretraining bought nothing.** Walk-forward 2014–2026: scratch +0.0056,
  full-finetune +0.0065, head-only +0.0049, against 2·SE of ±0.014–0.017 — a
  clean tie (`docs/campaign/D_REPORT.md`).
- **Survivorship bias**: the universe is today's lake ranked by today's
  liquidity; every backtest row is inflated by it, roughly equally.
- **Spinoff distortion**: yfinance adj_close handles splits+dividends, not
  spinoff value (GE breakup etc.).
- **The later power rounds' null runs cross the detection threshold too often.**
  v1 and v2 were clean (0/12, 0/8), but v3 logged 2/9 and 3/12 and v4 logged
  5/15 null runs crossing 2·SE, each marked INVESTIGATE in its own report and
  unresolved. At a 2σ threshold ~5% is expected, so a "detection" in Rounds 3–4
  is weaker evidence than the word implies. Treat those detection floors as
  provisional.
- **Batch-Sharpe** mixes market regimes within a shuffled batch — its std is
  cross-regime, not sequential; loss values are batch-size dependent.
- Long-only, fully invested, no shorting, frozen universe. Costs are a linear
  10 bps each way on turnover, with no market impact.

## The power test (docs/power/)

Before iterating on real-data training, we planted signals of known strength in
synthetic data (`crossfolio power`) and measured what the harness can recover.
Results ([curve](docs/power/power_curve.png), [full report](docs/power/POWER.md),
3 paired seeds, pre-registered thresholds, 0/12 false positives at γ=0):

| arm | minimum detected signal |
|---|---|
| attention / IC loss | **40 bps/month** (~80% oracle recovery at 80 bps) |
| attention / Sharpe loss | 80 bps/month |
| linear / either loss | never — blind at all strengths |

Three conclusions carry the project: **(1)** densified cross-sectional
supervision (IC loss) roughly halves the detectable signal strength — the
supervision-starvation diagnosis is confirmed and quantified; **(2)** parameter
sharing, not capacity, is what separates architectures here; **(3)** time-axis
window normalization provably erases momentum-level signals (`test_synth.py`),
so it is off by default. Later rounds sharpened (2): the attention-free `p7`
encoder detects at 20 bps and hits 70% oracle recovery at 40 bps, beating the
full attention model and the entire sim2real pipeline
([Round 3](docs/power/v3/ROUND3.md)); a correlation-biased attention variant is
the first arm to break the per-stock ceiling
([Round 4](docs/power/v4/ROUND4.md)).

For calibration: real cross-sectional anomalies are believed to live in the
~10–40 bps/month range — at the edge of, and mostly below, this harness's
detection floor. That is the honest context for any real-data result.

Two standing qualifications on this section: later reports run at the v2 config,
so numbers are not directly comparable across campaign versions; and the
Round 3–4 detections carry the unresolved null-crossing rate noted in the
caveats above.

## Repo map

See [ARCHITECTURE.md](ARCHITECTURE.md). One-line version: `config ← data ←
{models, losses} ← train ← evaluate/plots ← cli`; only `data/build.py` touches
the lake; only `train.py`/`evaluate.py` write `runs/`.

## Is there an actual edge? (docs/edge/)

Under pre-registered discipline — an untouched 2-year holdout, every trial
ledgered, net-of-costs only:

| round | dev net @10bps | holdout net @10bps | t (25 mo) |
|---|---|---|---|
| [5](docs/edge/VERDICT.md) — p7 + position smoothing | +0.62 | **+0.94** (+1.01/+0.79 at 5/20 bps) | 1.35 |
| [6](docs/edge/ROUND6.md) — + SPD/log-Euclidean regime feature | +0.856 | **+0.72** (+0.80/+0.58) | 1.04 |

Round 5 met its registered thresholds, but the development claim deflates past
significance under the 10-valid-trial Bonferroni from the ledger, and a
true-zero strategy produces a holdout like that roughly one time in ten.

Round 6 is the more instructive result: the regime feature added +0.24 on dev
and came back **lower** on the holdout. At 25 months the two reads are
noise-compatible, and the holdout has now been consumed twice, which degrades
its remaining evidentiary value. Round 6 also returned two confirmed nulls —
KL-to-EW, HHI and no penalty agree to three decimals, and no optimizer beats
Adam.

Verdict: **no demonstrated edge.** A frozen, paper-tracked champion evaluated in
true walk-forward time is the registered path to a real answer. Not investment
advice; not a basis for deploying money.

Process note: the v1 evaluation bug (graded against SPY rather than the
pre-registered equal-weight benchmark) was caught, ledgered as an amendment,
and the invalid rows were kept rather than deleted — see
[`ledger.jsonl`](docs/edge/ledger.jsonl).
