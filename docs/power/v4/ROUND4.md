# Round 4 verdicts: the relational gap is capturable — with the right inductive bias

Paired contrasts, relational signal, 3 seeds. Gap share = (arm−p7)/(oracle−p7).

- **corr_bias_attention: +20% gap share at γ80, +35% at γ120** — the first
  architecture to break the per-stock ceiling. Dials: gate 0.1→0.33; λ split
  head-wise: one head λ=+0.96 (attend to correlated peers — the designed
  mechanism), three heads λ≈−0.9 (attend to ANTI-correlated stocks — the same
  preference the probes found in the old model, now demonstrably load-bearing).
  Slight cost below γ40 (−3..−10%) where the machinery is noise.
- **gated_curriculum: failed decisively.** +6% at best, and gates stayed ≈0
  even DURING the 500-bps phase — loud gradients were never the bottleneck.
  H-SNR rejected; H-bias confirmed.
- Controls held: gated_attention == p7 exactly; p7-oracle row unchanged.

## Real data (r4-real-corrbias-s{0,1,2})
Gates open to 0.17–0.22 on real weekly equities (vs ±0.04 for unbiased gating
in R3) with |λ| ≈ 0.3–0.65, anti-correlation slightly dominant — **real markets
pull correlation-biased attention open**. But best val rank-IC (+0.007–0.008)
remains slightly below p7's (+0.010–0.015): the context is wanted by the
optimizer, and does not yet pay on held-out data at this budget. The dial and
the scoreboard disagree — the project's recurring theme, now measured at its
sharpest.

## Standing
SGD-through-attention was the bottleneck and a data-derived correlation bias
removes it. P7 remains the real-data champion; corr-bias attention is the
synthetic-relational champion and the first genuine P6 mechanism the harness
has produced. Next candidates: hybrid (corr-bias only where it pays), longer
real histories, and the anti-correlation heads as an interp target.
