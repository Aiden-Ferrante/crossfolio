# crossfolio architecture

## Modules and dependency direction

```
config  <-  data  <-  models, losses  <-  train  <-  evaluate, plots  <-  cli
universe (generated, committed)  <-  everyone
```

Arrows point from depended-on to dependent. Nothing imports `cli`; `cli` does
lazy imports per subcommand. No module imports a sibling to its right.

| module | responsibility | hides |
|---|---|---|
| `universe.py` | frozen (ticker, sector) list; N derives from it, nothing hardcodes N | how the universe was selected (scripts/freeze_universe.py) |
| `config.py` | every knob as frozen dataclasses + repo paths | — |
| `data/build.py` | **the only lake reader**: parquet → aligned log-return panel → `data/panel.npz` | stocklake layout, DuckDB, alignment/gap policy |
| `data/splits.py` | anchor enumeration + purged chronological split; asserts invariants at runtime | embargo arithmetic |
| `data/dataset.py` | Panel + AnchorDataset: anchor → (X, y, y_spy) | window slicing, normalization, log→simple return conversion |
| `models/` | Allocator contract `forward(X)->(softmax w, aux)` + registry | each architecture's internals; attention exposes `aux["attn"]` deliberately |
| `losses.py` | pure fns of (w, y, y_spy): batch-Sharpe / mean-excess + HHI | the grading — models never see returns |
| `train.py` | shared harness: AdamW, val-Sharpe early stop, seeding, checkpoints | run-dir layout (`runs/<ts>-<model>-<loss>/`) |
| `evaluate.py` | test-block walk → metrics + report.md | monthly-vs-daily anchor policy |
| `plots.py` | matplotlib renderings incl. sector-sorted attention maps | sector ordering |
| `stage0.py` | the no-input original idea, both losses side by side | — |

## Where state lives

- `data/panel.npz` — the one processed dataset; rebuilt only by `build-dataset`. Gitignored.
- `runs/<ts>-<model>-<loss>/` — config.json, best.pt, last.pt, history.jsonl, report/plots. Gitignored. Written only by `train.py`/`evaluate.py`.
- `src/crossfolio/universe.py` — generated but **committed**; the single source of N and column order. Regenerating it invalidates panel and all runs.
- Everything else is stateless; configs are frozen dataclasses.

## Load-bearing invariants

1. Allocation out, alpha as loss — no return-prediction head anywhere.
2. Features at anchor d use only rows ≤ d (probe-tested); grading windows never
   cross split boundaries (21-day purge, runtime-asserted).
3. Time blocks never mix in a batch; shuffling happens within train only.
4. Runtime is keyless/local (yfinance only inside the dev-time freeze script).
5. The harness is model-agnostic: a new architecture = one file + one registry
   line, inheriting data, splits, loss, training, evaluation, and plots.
