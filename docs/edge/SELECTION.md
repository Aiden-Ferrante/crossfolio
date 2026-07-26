# Phase 3 champion selection (written BEFORE the holdout run)

Development leaderboard (v2, net@10bps): p7_big/0.3 +0.628, corr_bias/0.3
+0.627, p7_base/0.3 +0.624, p7_stride1/0.3 +0.616 — a four-way statistical tie
(79 dev months => Sharpe SE ~0.4). Threshold (net > 0.5) passed by all four.

**Champion: p7_base with smooth_alpha=0.3.** Rationale: within 0.004 of the
nominal leader (deep inside noise, where picking the leader means picking
noise); the simplest and smallest of the tie; cost-robust (+0.515 at 20 bps);
and consistent with rounds 3-4, where the plain per-stock encoder was the
real-data champion everywhere. Smoothing is load-bearing: unsmoothed turnover
(~1.25/wk) destroys the signal at realistic costs.

Deflation context recorded in advance: the ledger holds 21 trial rows (10 v1
invalid + 1 amendment + 10 v2); the development claim is deflated by the 10
valid v2 trials (Bonferroni). The holdout is a single pre-registered test.
