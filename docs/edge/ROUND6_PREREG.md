# Round 6 pre-registration (committed before any experiment)

Inherits all Round-5 rules (holdout wall, ledger, net-of-costs, EW benchmark,
alpha=0.3 smoothing, paper-only). Incumbent reference: p7_base+smooth, dev net
Sharpe +0.62 @10bps.

**HOLDOUT GATE**: the 2024-07-26 holdout was consumed by Round 5. Round 6
touches it AT MOST once, only if a champion beats the incumbent's dev net
Sharpe by >= +0.15 on the identical dev walk-forward. Otherwise it stays
sealed and the verdict rests on dev evidence + the paper-track.

## Arms, exact definitions, registered predictions

- **A1 (KL vs HHI penalty)**: IC training loss gains a diversification term:
  loss = -mean_ic + lambda*P with P in {KL(w||uniform), HHI}, lambda=0.05,
  vs the lambda=0 incumbent. KL(w||u) = log N + sum w log w. Prediction: ~tie
  (both minimized at uniform); any gap says which geometry the data prefers.
- **A2 (SPT baselines, untrained)**: diversity-weighted w_i ∝ m_i(t)^p,
  p in {0.5, -0.5}. PROXY (registered): true market caps are unavailable in
  the panel; m_i(t) = pseudo-cap = cumulative gross return from panel start
  (equal initial caps evolving by returns — the SPT market-weight process
  under an equal-cap initialization). Prediction: competitive with EW.
- **A3 (geometry-matched updates)**: two well-defined sub-arms, replacing the
  goal text's underspecified "EMD on head params" (deviation registered here):
  (a) **EG(eta) online portfolio** — Helmbold et al. (1998) exact update
      w_{t+1,i} ∝ w_{t,i}·exp(eta·x_{t,i}/(w_t·x_t)) on weekly gross returns,
      eta in {0.05, 0.5}, analytic, no training. The literal paper as an arm.
  (b) **Optimizer-geometry probe**: incumbent architecture trained with
      Adam (incumbent) vs plain SGD vs RMSprop, same budget — does optimizer
      geometry move dev performance at all? Prediction for both: null
      (a confirmed null is the information).
- **A4 (SPD regime feature)**: r_t = z-scored log-Euclidean distance
  ||logm(C_t) - logm(C_{t-5})||_F between rolling 120d correlation matrices,
  RMT-denoised (eigenvalues below the Marchenko-Pastur edge averaged).
  Fed as (i) an extra input row appended to each stock's window for p7
  (Linear(T+1,...)) and (ii) a gate modulator g = g0 + g1*r_t for
  gated_attention. Prediction: likeliest winner — probes showed the models
  compute regime sensitivity; this hands them the canonical coordinate.

All dev comparisons: identical walk-forward (2018+, 5 seeds), ledger v3 rows.

## Phase 5 gate decision (written before the holdout run)
p7_regime dev net +0.856 >= gate (+0.77). Champion for the single holdout
shot: **p7_regime** (T+1 input, regime scalar appended). gated_regime (+0.812)
also cleared but p7_regime is simpler and higher. This is Round 6's one and
only holdout evaluation; ledger records it either way.
