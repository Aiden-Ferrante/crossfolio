# Round 5 verdict: pre-registered criteria met; edge suggestive, NOT demonstrated

One-shot holdout (2024-07-26 → 2026-07, 25 months, champion p7_base + smooth
0.3, trained on dev only, evaluated once):

| cost | holdout net excess Sharpe (ann.) |
|---|---|
| 5 bps | +1.01 |
| **10 bps (headline)** | **+0.94** |
| 20 bps | +0.79 |

t-stat of monthly net excess @10bps: **1.35** (one-sided p ≈ 0.10). Weekly
turnover 0.33, matching development — the mechanics generalized.

## Honest reading
- The pre-registered thresholds (holdout net > 0; dev net > 0.5) are MET.
- The evidence is NOT statistically significant: 25 holdout months put an SE of
  ~0.7 on that +0.94; the development claim (+0.62, p ≈ 0.055 one-sided)
  deflates past significance under the 10-valid-trial Bonferroni from the
  ledger. A true-zero strategy produces a holdout like this ~1 time in 10.
- Standing caveats on every number: survivorship-biased present-day universe,
  long-only, linear cost model without market impact, one 25-month regime.
- **Conclusion: suggestive positive, not a demonstrated edge — and explicitly
  not a basis for deploying real money.** The registered follow-up that would
  strengthen or kill it: paper-track the frozen champion forward out-of-sample
  (true walk-forward time, no retraining decisions), and/or extend the universe
  history back to 2000 for a second, disjoint pseudo-holdout.
- Process note: the v1 evaluation bug (graded vs SPY, near-uniform weights) was
  caught, ledgered, and amended before any conclusion — see ledger.jsonl.
