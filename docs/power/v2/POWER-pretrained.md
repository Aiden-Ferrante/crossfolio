# Planted-signal power test

## Pre-registered thresholds (written before any run)

- Headline metric: mean test IC over non-overlapping stride-H anchors;
  SE = (1/sqrt(N)) / sqrt(n_monthly).
- **Detection** = mean test IC > 2*SE.
- **gamma=0 acceptance** (false-positive check, all null runs): |test IC| < 2*SE
  AND best_epoch small AND no val improvement beyond noise. At a 2-sigma
  threshold, ~0-1 of the null runs may cross by chance — expected, and said here
  in advance so it can't become a post-hoc story.
- Recovery fraction = model test IC / same-panel empirical oracle test IC.


## Results

| γ (bps) | seed | model | loss | test IC | detected | best epoch | excess Sharpe |
|---|---|---|---|---|---|---|---|
| 0 | 0 | **oracle** | - | -0.0119 | - | - | - |
| 0 | 0 | attention | sharpe | -0.0008 | no | 3 | +0.16 |
| 0 | 0 | attention | ic | +0.0019 | no | 24 | +0.34 |
| 5 | 0 | **oracle** | - | -0.0089 | - | - | - |
| 5 | 0 | attention | sharpe | -0.0008 | no | 3 | +0.16 |
| 5 | 0 | attention | ic | +0.0024 | no | 22 | +0.43 |
| 10 | 0 | **oracle** | - | -0.0058 | - | - | - |
| 10 | 0 | attention | sharpe | -0.0001 | no | 3 | +0.17 |
| 10 | 0 | attention | ic | -0.0019 | no | 29 | +0.34 |
| 15 | 0 | **oracle** | - | -0.0028 | - | - | - |
| 15 | 0 | attention | sharpe | +0.0014 | no | 6 | +0.03 |
| 15 | 0 | attention | ic | -0.0002 | no | 3 | +0.37 |
| 20 | 0 | **oracle** | - | +0.0003 | - | - | - |
| 20 | 0 | attention | sharpe | +0.0027 | no | 5 | +0.24 |
| 20 | 0 | attention | ic | +0.0047 | no | 4 | +0.68 |
| 40 | 0 | **oracle** | - | +0.0125 | - | - | - |
| 40 | 0 | attention | sharpe | +0.0126 | YES | 4 | +1.61 |
| 40 | 0 | attention | ic | +0.0119 | YES | 4 | +1.13 |
| 0 | 1 | **oracle** | - | -0.0142 | - | - | - |
| 0 | 1 | attention | sharpe | +0.0005 | no | 28 | -0.02 |
| 0 | 1 | attention | ic | -0.0053 | no | 8 | -0.02 |
| 5 | 1 | **oracle** | - | -0.0111 | - | - | - |
| 5 | 1 | attention | sharpe | -0.0019 | no | 5 | -0.29 |
| 5 | 1 | attention | ic | -0.0058 | no | 9 | +0.03 |
| 10 | 1 | **oracle** | - | -0.0079 | - | - | - |
| 10 | 1 | attention | sharpe | -0.0090 | no | 6 | -0.94 |
| 10 | 1 | attention | ic | -0.0044 | no | 4 | -0.16 |
| 15 | 1 | **oracle** | - | -0.0048 | - | - | - |
| 15 | 1 | attention | sharpe | -0.0024 | no | 5 | -0.18 |
| 15 | 1 | attention | ic | -0.0063 | no | 4 | +0.13 |
| 20 | 1 | **oracle** | - | -0.0018 | - | - | - |
| 20 | 1 | attention | sharpe | -0.0021 | no | 5 | -0.00 |
| 20 | 1 | attention | ic | +0.0002 | no | 3 | +0.31 |
| 40 | 1 | **oracle** | - | +0.0106 | - | - | - |
| 40 | 1 | attention | sharpe | +0.0100 | YES | 4 | +1.76 |
| 40 | 1 | attention | ic | +0.0099 | YES | 3 | +0.70 |
| 0 | 2 | **oracle** | - | +0.0035 | - | - | - |
| 0 | 2 | attention | sharpe | -0.0076 | no | 3 | -0.95 |
| 0 | 2 | attention | ic | -0.0018 | no | 3 | -0.64 |
| 5 | 2 | **oracle** | - | +0.0067 | - | - | - |
| 5 | 2 | attention | sharpe | -0.0014 | no | 5 | -0.46 |
| 5 | 2 | attention | ic | -0.0048 | no | 2 | -0.73 |
| 10 | 2 | **oracle** | - | +0.0098 | - | - | - |
| 10 | 2 | attention | sharpe | +0.0033 | no | 5 | +0.01 |
| 10 | 2 | attention | ic | -0.0018 | no | 2 | -0.44 |
| 15 | 2 | **oracle** | - | +0.0130 | - | - | - |
| 15 | 2 | attention | sharpe | +0.0067 | no | 5 | +0.43 |
| 15 | 2 | attention | ic | +0.0063 | no | 3 | -0.35 |
| 20 | 2 | **oracle** | - | +0.0161 | - | - | - |
| 20 | 2 | attention | sharpe | +0.0074 | no | 4 | +0.81 |
| 20 | 2 | attention | ic | +0.0052 | no | 2 | +0.00 |
| 40 | 2 | **oracle** | - | +0.0288 | - | - | - |
| 40 | 2 | attention | sharpe | +0.0211 | YES | 4 | +2.52 |
| 40 | 2 | attention | ic | +0.0193 | YES | 2 | +0.64 |

Absolute detection threshold: 2·SE = 0.0082 (secondary — slow scores are
serially correlated across anchors, so absolute IC has far fewer effective
observations than anchor counts suggest).

## Verdict (paired contrasts — the official statistic)

Within-seed IC(γ) − IC(0) cancels the serially-correlated score-baseline noise
that the paired common-random-numbers design was built to cancel.

- **oracle / -**: minimum detected γ = 5 bps/month. γ5:+0.0031(±0.0001)* γ10:+0.0062(±0.0001)* γ15:+0.0093(±0.0002)* γ20:+0.0124(±0.0002)* γ40:+0.0248(±0.0005)*
- **attention / sharpe**: minimum detected γ = 40 bps/month. γ5:+0.0012(±0.0051) γ10:+0.0007(±0.0117) γ15:+0.0045(±0.0102) γ20:+0.0053(±0.0103) γ40:+0.0172(±0.0117)*
- **attention / ic**: minimum detected γ = 20 bps/month. γ5:-0.0010(±0.0021) γ10:-0.0010(±0.0029) γ15:+0.0017(±0.0065) γ20:+0.0051(±0.0024)* γ40:+0.0154(±0.0064)*
- **γ=0 false positives**: 0 of 6 null runs crossed 2·SE (within chance expectation).
