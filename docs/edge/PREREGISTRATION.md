# Round 5 pre-registration (committed before any experiment)

- **Holdout**: all data from 2024-07-26 onward. Development code loads panels
  only via `edge.dev_panel()` (truncated before the boundary minus H).
  `scripts/holdout_once.py` is the single permitted full-panel reader, run once
  in Phase 3 on the single pre-selected champion. Tests enforce the wall.
- **Headline metric**: NET-of-costs excess Sharpe vs daily-rebalanced
  equal-weight; costs = 10 bps each way x weekly turnover Σ|Δw| (sensitivity
  reported at 5/20 bps). Monthly aggregation = calendar-month sums of weekly
  net excess (stated approximation). No gross number may appear in a conclusion.
- **Success threshold**: holdout net excess Sharpe > 0 AND development net
  excess Sharpe > 0.5 with trial-count-deflated significance (every trained
  variant logged to docs/edge/ledger.jsonl; the final report deflates by ledger
  count — Bonferroni on the development claim; the holdout is a single test).
- **Candidate budget**: p7 tuning grid, reversal parameterization, corr-bias
  hybrid, ensembles, ≤2 wildcards. Anything losing net to equal-weight in
  development is killed, not tuned.
- **Known biases stated on every table**: survivorship (present-day universe),
  no borrow/shorting (long-only), cost model is linear and ignores impact.
- Turnover control = post-hoc EMA position smoothing (alpha in the ledger),
  chosen over a loss-side penalty for trial economy; stated here as a deviation
  from the goal text's "turnover-aware loss option".
