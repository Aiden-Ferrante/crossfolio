# Round 6 verdict: geometry round — one real winner (regime feature), two clean nulls

Dev leaderboard (net @10bps, identical walk-forward, ledger v3):
p7_regime **+0.856** | gated_regime +0.812 | incumbent/KL/HHI (exact tie) +0.618
| RMSprop +0.564 | SGD +0.515 | SPT p=-0.5 +0.324 (untrained bar) | SPT p=+0.5
-0.065 | EG -0.23/-0.33.

Per-arm vs registered predictions:
- **A1 CONFIRMED NULL**: KL-to-EW == HHI == no penalty, to three decimals — at
  this operating point (standardized logits + smoothing) diversification
  penalties are inert.
- **A2 CONFIRMED**: anti-cap SPT diversity earns +0.32 untrained; a permanent
  honesty bar that learned models must clear (they do).
- **A3 CONFIRMED NULL**: no optimizer beats Adam; EG online portfolios lose
  vs EW on this grading. Geometry of the *update rule* is not where the
  money was.
- **A4 CONFIRMED WINNER (dev)**: the SPD/log-Euclidean regime scalar lifts the
  champion +0.24 dev net Sharpe — the geometry of the *state space* is where
  the information was, exactly as the Round-2 probes hinted.

## Holdout (gate triggered at +0.856 >= +0.77; single shot, second lifetime read)
p7_regime: net +0.72 @10bps (+0.80/+0.58 at 5/20 bps), t=1.04 over 25 months.
Positive and cost-robust, but (a) NOT significant, (b) LOWER than the Round-5
incumbent's holdout (+0.94, t=1.35) despite the +0.24 dev advantage — at 25
months these are mutually noise-compatible, and (c) the holdout has now been
read twice, degrading its future evidentiary value. Honest conclusion: the
regime feature is the best *dev-confirmed* improvement the project has found;
the holdout neither confirms nor refutes it. The frozen paper-track (both
champions) is the remaining judge that can settle it.

Ledger note: module-level analytic arms re-logged on import (duplicate
spt/eg rows, identical values) — append-only record kept as-is.
Standing caveats: survivorship-biased universe, long-only, linear costs,
pseudo-cap SPT proxy. Not investment advice; not a basis for real money.
