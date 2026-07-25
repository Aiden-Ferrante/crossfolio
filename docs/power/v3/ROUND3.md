# Round 3 verdicts: attention never earned its place

Paired contrasts, 3 seeds, small-arch arms (d32), IC loss. Full tables in POWER*.md.

## R1 — momentum signal
- **p7_encoder (attention-free) detects at 20 bps (+0.0057) and hits 70% oracle
  recovery at 40 bps (+0.0175)** — beating the full attention model (40-bps floor,
  31%) AND matching/beating the entire sim2real pretrained pipeline (20-bps floor,
  62%). Deleting attention reproduced the whole campaign's gain for free.
- gated_attention == p7_encoder to 4 decimals; gates stayed at ±0.006 ≈ 0.

## R2 — relational signal (sector-mates lead-lag)
- The two-oracle device self-corrected as designed: the 4-feature p7-oracle said
  the own-window leak was tiny (+0.004 @ 120 bps), but the p7_encoder MLP reached
  +0.0189 — the sector-factor leak reachable by a NONLINEAR per-stock model is
  ~29% of the full oracle (+0.0662). Ridge oracles lower-bound leaks; MLPs collect them.
- **The genuinely-relational remainder (~70% of oracle) was captured by NO arm.**
  Gates stayed shut (±0.01); plain attention underperformed p7 again (+0.0093).

## R3 — real data
- p7 vs gated: statistical tie (val rank-IC ≈ +0.010..+0.015 both). Gates opened
  slightly on real data ({+0.038, −0.002, +0.051} vs ±0.006 synthetic) — an order
  of magnitude more pull than pure per-stock synthetic signals, but tiny in
  absolute terms and buying no measurable validation improvement.

## Reading
The double dissociation FAILED informatively: gates stayed shut even when
context held 70% of the signal. Combined with R2's uncaptured gap, the verdict
is a **trainability failure of SGD-through-attention at this scale**, not
evidence the relational structure is unlearnable (the oracle proves it's
there). Cross-sectional attention, as built, has now lost on every field:
causally harmful when trained un-gated, unopened when gated, matched by its
own encoder everywhere. The P7 per-stock encoder is the reigning architecture
of this harness — 22k params, no pretraining required.

Next candidates if the relational gap is ever to be captured: attention with a
relational inductive bias (e.g. attention over CORRELATION-derived graphs, P12),
auxiliary losses that supervise attention directly, or curriculum where
relational signal strength starts huge and anneals.
