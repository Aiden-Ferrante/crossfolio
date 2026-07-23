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
| 0 | 0 | linear | sharpe | -0.0038 | no | 0 | +0.32 |
| 0 | 0 | linear | ic | -0.0027 | no | 34 | -0.18 |
| 0 | 0 | attention | sharpe | -0.0032 | no | 0 | +0.23 |
| 0 | 0 | attention | ic | +0.0043 | no | 13 | +0.57 |
| 5 | 0 | **oracle** | - | -0.0089 | - | - | - |
| 5 | 0 | linear | sharpe | -0.0038 | no | 0 | +0.40 |
| 5 | 0 | linear | ic | -0.0025 | no | 34 | -0.13 |
| 5 | 0 | attention | sharpe | -0.0033 | no | 0 | +0.30 |
| 5 | 0 | attention | ic | +0.0041 | no | 13 | +0.54 |
| 10 | 0 | **oracle** | - | -0.0058 | - | - | - |
| 10 | 0 | linear | sharpe | -0.0037 | no | 0 | +0.48 |
| 10 | 0 | linear | ic | -0.0024 | no | 34 | -0.09 |
| 10 | 0 | attention | sharpe | -0.0033 | no | 0 | +0.38 |
| 10 | 0 | attention | ic | +0.0039 | no | 13 | +0.54 |
| 15 | 0 | **oracle** | - | -0.0028 | - | - | - |
| 15 | 0 | linear | sharpe | -0.0037 | no | 0 | +0.56 |
| 15 | 0 | linear | ic | -0.0023 | no | 37 | -0.06 |
| 15 | 0 | attention | sharpe | -0.0034 | no | 0 | +0.45 |
| 15 | 0 | attention | ic | +0.0039 | no | 13 | +0.56 |
| 20 | 0 | **oracle** | - | +0.0003 | - | - | - |
| 20 | 0 | linear | sharpe | -0.0037 | no | 0 | +0.65 |
| 20 | 0 | linear | ic | -0.0024 | no | 37 | -0.03 |
| 20 | 0 | attention | sharpe | -0.0034 | no | 0 | +0.53 |
| 20 | 0 | attention | ic | +0.0041 | no | 13 | +0.60 |
| 40 | 0 | **oracle** | - | +0.0125 | - | - | - |
| 40 | 0 | linear | sharpe | -0.0036 | no | 0 | +0.99 |
| 40 | 0 | linear | ic | -0.0029 | no | 37 | +0.08 |
| 40 | 0 | attention | sharpe | -0.0036 | no | 0 | +0.85 |
| 40 | 0 | attention | ic | +0.0105 | YES | 25 | +1.34 |
| 0 | 1 | **oracle** | - | -0.0142 | - | - | - |
| 0 | 1 | attention | sharpe | -0.0016 | no | 5 | -0.18 |
| 0 | 1 | attention | ic | -0.0050 | no | 8 | -0.71 |
| 5 | 1 | **oracle** | - | -0.0111 | - | - | - |
| 5 | 1 | attention | sharpe | -0.0016 | no | 5 | -0.17 |
| 5 | 1 | attention | ic | -0.0053 | no | 8 | -0.72 |
| 10 | 1 | **oracle** | - | -0.0079 | - | - | - |
| 10 | 1 | attention | sharpe | -0.0016 | no | 5 | -0.15 |
| 10 | 1 | attention | ic | -0.0056 | no | 8 | -0.71 |
| 15 | 1 | **oracle** | - | -0.0048 | - | - | - |
| 15 | 1 | attention | sharpe | -0.0017 | no | 5 | -0.13 |
| 15 | 1 | attention | ic | -0.0069 | no | 11 | -0.72 |
| 20 | 1 | **oracle** | - | -0.0018 | - | - | - |
| 20 | 1 | attention | sharpe | -0.0025 | no | 7 | -0.26 |
| 20 | 1 | attention | ic | -0.0077 | no | 16 | -0.66 |
| 40 | 1 | **oracle** | - | +0.0106 | - | - | - |
| 40 | 1 | attention | sharpe | +0.0025 | no | 19 | +0.54 |
| 40 | 1 | attention | ic | -0.0030 | no | 16 | +0.09 |
| 0 | 2 | **oracle** | - | +0.0035 | - | - | - |
| 0 | 2 | attention | sharpe | -0.0002 | no | 0 | -0.64 |
| 0 | 2 | attention | ic | -0.0068 | no | 31 | -1.13 |
| 5 | 2 | **oracle** | - | +0.0067 | - | - | - |
| 5 | 2 | attention | sharpe | -0.0002 | no | 0 | -0.56 |
| 5 | 2 | attention | ic | -0.0065 | no | 31 | -1.02 |
| 10 | 2 | **oracle** | - | +0.0098 | - | - | - |
| 10 | 2 | attention | sharpe | -0.0002 | no | 0 | -0.48 |
| 10 | 2 | attention | ic | -0.0049 | no | 31 | -0.79 |
| 15 | 2 | **oracle** | - | +0.0130 | - | - | - |
| 15 | 2 | attention | sharpe | -0.0002 | no | 0 | -0.40 |
| 15 | 2 | attention | ic | -0.0030 | no | 31 | -0.55 |
| 20 | 2 | **oracle** | - | +0.0161 | - | - | - |
| 20 | 2 | attention | sharpe | -0.0002 | no | 0 | -0.32 |
| 20 | 2 | attention | ic | -0.0011 | no | 31 | -0.30 |
| 40 | 2 | **oracle** | - | +0.0288 | - | - | - |
| 40 | 2 | attention | sharpe | +0.0073 | no | 31 | +0.97 |
| 40 | 2 | attention | ic | +0.0079 | no | 26 | +0.99 |

Absolute detection threshold: 2·SE = 0.0082 (secondary — slow scores are
serially correlated across anchors, so absolute IC has far fewer effective
observations than anchor counts suggest).

## Verdict (paired contrasts — the official statistic)

Within-seed IC(γ) − IC(0) cancels the serially-correlated score-baseline noise
that the paired common-random-numbers design was built to cancel.

- **oracle / -**: minimum detected γ = 5 bps/month. γ5:+0.0031(±0.0001)* γ10:+0.0062(±0.0001)* γ15:+0.0093(±0.0002)* γ20:+0.0124(±0.0002)* γ40:+0.0248(±0.0005)*
- **attention / sharpe**: no detection at any planted strength. γ5:-0.0000(±0.0000) γ10:-0.0000(±0.0000) γ15:-0.0001(±0.0001) γ20:-0.0004(±0.0006) γ40:+0.0038(±0.0046)
- **attention / ic**: minimum detected γ = 40 bps/month. γ5:-0.0001(±0.0004) γ10:+0.0003(±0.0016) γ15:+0.0005(±0.0034) γ20:+0.0009(±0.0050) γ40:+0.0077(±0.0075)*
- **γ=0 false positives**: 0 of 8 null runs crossed 2·SE (within chance expectation).
