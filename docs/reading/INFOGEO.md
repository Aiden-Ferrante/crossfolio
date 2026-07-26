# Finance × Information Geometry: verified reading list + strategy synthesis

Compiled 2026-07-26 via adversarially-verified deep research (106 agents, 3-vote
citation verification). Tracks 1–2 and the finance↔geometry bridge are
**web-verified against primary sources**; Track 3 entries are from the
assistant's knowledge and marked ◇ = real to the best of its knowledge but NOT
web-verified this session — check before citing.

## Track 1 — Finance on the simplex (verified ✓)

1. ✓ **Cover, T.M., "Universal Portfolios," *Mathematical Finance* 1(1):1–29 (1991).**
   Wealth-weighted average over all simplex portfolios asymptotically matches the
   best constant-rebalanced portfolio in hindsight. → The theoretical ceiling any
   simplex allocator is chasing; the canonical benchmark arm.
2. ✓ **Helmbold, Schapire, Singer & Warmuth, "On-Line Portfolio Selection Using
   Multiplicative Updates," *Mathematical Finance* 8(4):325–347 (1998).**
   EG(η) = maximize linearized log-wealth minus KL to previous weights — *exactly
   entropic mirror descent*; O(√(log N / T)) regret, linear cost per period.
   → The single most direct bridge: your softmax allocator's update geometry,
   with a √(N/ln N) dimension advantage over Euclidean methods at N≈400.
3. ✓ **Vervuurt & Karatzas, "Diversity-weighted portfolios with negative
   parameter," *Annals of Finance* 11:411–432 (2015).**
   Weighting inversely to market weight yields relative arbitrage vs the cap-
   weighted market under volatility non-degeneracy. → An implementable SPT
   baseline arm with actual theory behind it.
4. ✓ **Cuchiero, Schachermayer & Wong, "Cover's universal portfolio, stochastic
   portfolio theory and the numéraire portfolio," *Mathematical Finance*
   29(3):773–803 (2019).** Unifies Cover, Fernholz SPT, and the numéraire
   portfolio; extends universality to functionally generated portfolios.
   → The modern synthesis paper; read after 1–3.
5. ✓ **Wong, T.-K.L., "Information Geometry in Portfolio Theory," in *Geometric
   Structures of Information* (Nielsen ed.), Springer 2019, pp. 105–136.**
   The explicit survey of the finance↔information-geometry bridge (exponential
   concavity, free-energy duality, relative arbitrage as geometry). → The map of
   the whole intersection; the paper to read FIRST.

## Track 2 — Information geometry core (verified ✓)

6. ✓ **Amari, "Natural Gradient Works Efficiently in Learning," *Neural
   Computation* 10(2):251–276 (1998).** The Fisher-metric gradient is the true
   steepest descent on a statistical manifold; online natural gradient is
   asymptotically Fisher-efficient. → Why "vanilla Adam on simplex outputs" is
   geometrically naive — the training-rule hypothesis for Round 6.
7. ✓ **Amari & Nagaoka, *Methods of Information Geometry*, AMS Translations of
   Mathematical Monographs 191 (2000).** Fisher metric + dual α-connections; the
   canonical textbook. → The foundation text of your long-run info-geometry track.
8. ✓ **Beck & Teboulle, "Mirror descent and nonlinear projected subgradient
   methods for convex optimization," *Operations Research Letters* 31:167–175
   (2003).** Mirror descent = projected subgradient with a Bregman distance;
   defines Entropic Mirror Descent on the simplex with √(2 ln N) efficiency.
   → The optimization-theory bridge from EG to modern ML.
9. ✓ **Martens & Grosse, "Optimizing Neural Networks with Kronecker-factored
   Approximate Curvature," ICML 2015.** Practical natural gradient via
   Kronecker-factored Fisher blocks. → The implementable natural-gradient
   trainer if the mirror-descent arm shows promise.
10. ✓ **Martens, "New Insights and Perspectives on the Natural Gradient Method,"
    *JMLR* 21(146):1–76 (2020).** Natural gradient as second-order optimization;
    approximate parameterization invariance. → The deep-dive before building
    anything K-FAC-shaped.

## Track 3 — Geometry applied to markets (◇ = from model knowledge, verify before citing)

11. ◇ Laloux, Cizeau, Bouchaud & Potters, "Noise Dressing of Financial
    Correlation Matrices," *PRL* 83:1467 (1999) — already load-bearing in your
    probe suite; the RMT noise band.
12. ◇ Plerou, Gopikrishnan, Rosenow, Amaral & Stanley, "Universal and
    Nonuniversal Properties of Cross Correlations in Financial Time Series,"
    *PRL* 83:1471 (1999) — the companion RMT result; sector eigenmodes.
13. ◇ Mantegna, "Hierarchical structure in financial markets," *Eur. Phys. J. B*
    11:193 (1999) — correlation-distance MSTs; the ancestor of your
    attention-vs-correlation probes.
14. ◇ Onnela, Chakraborti, Kaski, Kertész & Kanto, "Dynamics of market
    correlations," *Phys. Rev. E* 68:056110 (2003) — time-varying correlation
    trees through crashes; the regime-dynamics reference.
15. ◇ Pennec, Fillard & Ayache, "A Riemannian Framework for Tensor Computing,"
    *IJCV* 66(1):41–66 (2006) — the affine-invariant SPD metric; the natural
    distance for rolling correlation matrices.
16. ◇ Arsigny, Fillard, Pennec & Ayache, "Log-Euclidean metrics" (*Magn. Reson.
    Med.* 56:411 (2006) / *SIAM J. Matrix Anal.* 29:328 (2007)) — the cheap SPD
    metric you'd actually compute at scale.
17. ◇ Bera & Park, "Optimal Portfolio Diversification Using the Maximum Entropy
    Principle," *Econometric Reviews* 27(4-6):484–512 (2008) — entropy-based
    diversification; the direct ancestor of KL-to-EW regularization.
18. ◇ Pal & Wong, "Exponentially concave functions and a new information
    geometry," *Annals of Probability* 46(2):1070–1113 (2018) — the research
    core that Wong's survey (#5) reviews; L-divergences and relative arbitrage.

## Synthesis: the Round 6 this literature points at

The exact result to build on: **your allocator already lives in this geometry.**
EG (#2) is entropic mirror descent (#8) on the simplex, which is the natural-
gradient geometry (#6) of a softmax output. Four concrete, cheap, pre-registrable
arms fall out, in order of expected information per GPU-hour:

1. **KL-to-equal-weight replaces HHI** as the diversification penalty — it is
   the Bregman divergence this geometry says you should have been using
   (Σw² was the Euclidean guess). One-line change, direct A/B.
2. **SPT baseline arms** (#3): diversity-weighted p∈{0.5, −0.5} portfolios join
   equal-weight as untrained references with relative-arbitrage theory attached.
   If learned models can't beat *these*, that's the finding.
3. **Entropic-mirror-descent training** of the allocator head vs Adam (#2, #8),
   with K-FAC (#9) as the escalation if the geometry-matched optimizer moves
   val rank-IC.
4. **SPD-manifold regime features** (#15/#16 on the rolling correlation matrix,
   with #11's noise band as the denoiser): log-Euclidean distance between
   consecutive correlation matrices as a gate input — the info-geometric
   upgrade of the regime gating that Round-2 probes showed the models want.

Caveats stated once: none of this manufactures signal — geometry changes the
*inductive bias and the optimizer*, not the information content of prices; every
arm goes through the same pre-registered net-of-costs discipline as Round 5, and
the same honest prior applies.
