# crossfolio

[![CI](https://github.com/Aiden-Ferrante/crossfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/Aiden-Ferrante/crossfolio/actions/workflows/ci.yml)

A neural portfolio allocator for a 405-stock universe, built mainly as an
**evaluation harness**: the part worth reading is how claims get checked.

**Result: honest negative.** The best holdout was +0.94 net excess Sharpe @10bps
at t = 1.35 over 25 months, not significant (`docs/edge/VERDICT.md`). A
zero-parameter 5-day reversal baseline (+0.913 net on dev) beat the best learned
arm (+0.856) (`docs/edge/ledger.jsonl`, round 7 vs round 6).

## The evaluation method

- **Pre-registration.** Metric, success threshold, cost model and candidate
  budget are committed before any run (`docs/edge/PREREGISTRATION.md`); every
  trained variant goes into a ledger (`docs/edge/ledger.jsonl`, 80 rows) and the
  dev claim is Bonferroni-deflated by the ledger's trial count.
- **Leakage poison probe.** Overwrite every panel row after the anchor with
  garbage and assert the features are bit-identical (`tests/test_dataset.py`,
  `tests/test_synth.py`).
- **Purged chronological splits.** Overlapping daily anchors make random splits
  leak, so blocks are chronological with an H-anchor purge at each boundary;
  `data/splits.py` raises `LeakageError` if any grading window crosses one.
- **Holdout wall.** Dev code can only load `edge.dev_panel()`, truncated before
  2024-07-26; `scripts/holdout_once.py` is the one full-panel reader, and a test
  enforces the wall.
- **Power test first.** Before trusting a real-data null, plant signals of known
  strength in synthetic data and measure what the harness can detect
  (`docs/power/`), with null runs as false-positive controls.

## Quickstart (CPU, no private data)

```bash
uv sync --extra cpu          # or plain `uv sync` (PyPI torch); `--extra cu130` for CUDA 13
uv run pytest -q             # 48 passed, 1 skipped (holdout wall: needs the private panel)
uv run crossfolio demo       # plant a signal, see if the harness finds it (45 s on CPU)
```

The demo output from a CPU-only run (seed 0, N=405, D=5000, p7_encoder/IC loss):

```
gamma bps/mo  oracle IC  model IC    2*SE  recovery  verdict
           0    -0.0119   -0.0012  0.0082        - null (control: should be null)
          40    +0.0125   +0.0101  0.0082      81% DETECTED
          80    +0.0373   +0.0360  0.0082      96% DETECTED
```

Real data: set `CROSSFOLIO_LAKE` to a folder with one parquet per ticker in
`src/crossfolio/universe.py` plus SPY (`date, ticker, adj_close`), then
`uv run crossfolio build-dataset`. The author's lake (`stocklake`) is private.

## What went wrong / what I learned

- **No demonstrated edge**, and Round 6's "better" regime feature came back
  lower on the holdout (+0.72), which has now been consumed twice.
- **A free baseline beat the models**; it is a single unconfirmed run, disclosed as such.
- **Attention hurt**: flattening it to uniform improved detection (`docs/probes/PROBES.md`).
- **A grading bug was caught and kept**: v1 graded against SPY, not the
  registered equal-weight benchmark; ledgered as an amendment, rows kept.

Full design, every caveat (survivorship, spinoffs, null crossings in power
rounds 3–4) and the round tables: [docs/RESULTS.md](docs/RESULTS.md). Code map:
[ARCHITECTURE.md](ARCHITECTURE.md). Not investment advice.
