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
| 0 | 0 | p7_encoder | ic | -0.0012 | no | 2 | +0.32 |
| 0 | 0 | attention | ic | +0.0043 | no | 13 | +0.57 |
| 0 | 0 | gated_attention | ic | -0.0012 | no | 2 | +0.32 |
| 5 | 0 | **oracle** | - | -0.0089 | - | - | - |
| 5 | 0 | p7_encoder | ic | -0.0030 | no | 1 | +0.37 |
| 5 | 0 | attention | ic | +0.0041 | no | 13 | +0.54 |
| 5 | 0 | gated_attention | ic | -0.0030 | no | 1 | +0.37 |
| 10 | 0 | **oracle** | - | -0.0058 | - | - | - |
| 10 | 0 | p7_encoder | ic | -0.0026 | no | 2 | +0.44 |
| 10 | 0 | attention | ic | +0.0039 | no | 13 | +0.54 |
| 10 | 0 | gated_attention | ic | -0.0026 | no | 2 | +0.44 |
| 15 | 0 | **oracle** | - | -0.0028 | - | - | - |
| 15 | 0 | p7_encoder | ic | -0.0018 | no | 2 | +0.54 |
| 15 | 0 | attention | ic | +0.0039 | no | 13 | +0.56 |
| 15 | 0 | gated_attention | ic | -0.0018 | no | 2 | +0.54 |
| 20 | 0 | **oracle** | - | +0.0003 | - | - | - |
| 20 | 0 | p7_encoder | ic | -0.0004 | no | 3 | +0.65 |
| 20 | 0 | attention | ic | +0.0041 | no | 13 | +0.60 |
| 20 | 0 | gated_attention | ic | -0.0004 | no | 3 | +0.65 |
| 40 | 0 | **oracle** | - | +0.0125 | - | - | - |
| 40 | 0 | p7_encoder | ic | +0.0101 | YES | 7 | +1.24 |
| 40 | 0 | attention | ic | +0.0105 | YES | 25 | +1.34 |
| 40 | 0 | gated_attention | ic | +0.0101 | YES | 7 | +1.23 |
| 0 | 1 | **oracle** | - | -0.0142 | - | - | - |
| 0 | 1 | p7_encoder | ic | -0.0102 | no | 0 | -0.01 |
| 0 | 1 | attention | ic | -0.0050 | no | 8 | -0.71 |
| 0 | 1 | gated_attention | ic | -0.0102 | no | 0 | -0.01 |
| 5 | 1 | **oracle** | - | -0.0111 | - | - | - |
| 5 | 1 | p7_encoder | ic | -0.0006 | no | 1 | +0.04 |
| 5 | 1 | attention | ic | -0.0053 | no | 8 | -0.72 |
| 5 | 1 | gated_attention | ic | -0.0006 | no | 1 | +0.04 |
| 10 | 1 | **oracle** | - | -0.0079 | - | - | - |
| 10 | 1 | p7_encoder | ic | -0.0036 | no | 53 | -0.13 |
| 10 | 1 | attention | ic | -0.0056 | no | 8 | -0.71 |
| 10 | 1 | gated_attention | ic | -0.0044 | no | 48 | -0.16 |
| 15 | 1 | **oracle** | - | -0.0048 | - | - | - |
| 15 | 1 | p7_encoder | ic | -0.0050 | no | 32 | -0.09 |
| 15 | 1 | attention | ic | -0.0069 | no | 11 | -0.72 |
| 15 | 1 | gated_attention | ic | -0.0045 | no | 22 | -0.10 |
| 20 | 1 | **oracle** | - | -0.0018 | - | - | - |
| 20 | 1 | p7_encoder | ic | -0.0039 | no | 35 | +0.01 |
| 20 | 1 | attention | ic | -0.0077 | no | 16 | -0.66 |
| 20 | 1 | gated_attention | ic | -0.0036 | no | 38 | -0.00 |
| 40 | 1 | **oracle** | - | +0.0106 | - | - | - |
| 40 | 1 | p7_encoder | ic | +0.0077 | no | 20 | +0.96 |
| 40 | 1 | attention | ic | -0.0030 | no | 16 | +0.09 |
| 40 | 1 | gated_attention | ic | +0.0077 | no | 20 | +0.96 |
| 0 | 2 | **oracle** | - | +0.0035 | - | - | - |
| 0 | 2 | p7_encoder | ic | +0.0036 | no | 11 | -0.57 |
| 0 | 2 | attention | ic | -0.0068 | no | 31 | -1.13 |
| 0 | 2 | gated_attention | ic | +0.0036 | no | 11 | -0.58 |
| 5 | 2 | **oracle** | - | +0.0067 | - | - | - |
| 5 | 2 | p7_encoder | ic | +0.0049 | no | 11 | -0.46 |
| 5 | 2 | attention | ic | -0.0065 | no | 31 | -1.02 |
| 5 | 2 | gated_attention | ic | +0.0049 | no | 11 | -0.47 |
| 10 | 2 | **oracle** | - | +0.0098 | - | - | - |
| 10 | 2 | p7_encoder | ic | +0.0072 | no | 11 | -0.33 |
| 10 | 2 | attention | ic | -0.0049 | no | 31 | -0.79 |
| 10 | 2 | gated_attention | ic | +0.0072 | no | 11 | -0.33 |
| 15 | 2 | **oracle** | - | +0.0130 | - | - | - |
| 15 | 2 | p7_encoder | ic | +0.0103 | YES | 11 | -0.17 |
| 15 | 2 | attention | ic | -0.0030 | no | 31 | -0.55 |
| 15 | 2 | gated_attention | ic | +0.0103 | YES | 11 | -0.17 |
| 20 | 2 | **oracle** | - | +0.0161 | - | - | - |
| 20 | 2 | p7_encoder | ic | +0.0135 | YES | 11 | -0.01 |
| 20 | 2 | attention | ic | -0.0011 | no | 31 | -0.30 |
| 20 | 2 | gated_attention | ic | +0.0135 | YES | 11 | -0.01 |
| 40 | 2 | **oracle** | - | +0.0288 | - | - | - |
| 40 | 2 | p7_encoder | ic | +0.0269 | YES | 11 | +0.68 |
| 40 | 2 | attention | ic | +0.0079 | no | 26 | +0.99 |
| 40 | 2 | gated_attention | ic | +0.0269 | YES | 11 | +0.68 |

Absolute detection threshold: 2·SE = 0.0082 (secondary — slow scores are
serially correlated across anchors, so absolute IC has far fewer effective
observations than anchor counts suggest).

## Verdict (paired contrasts — the official statistic)

Within-seed IC(γ) − IC(0) cancels the serially-correlated score-baseline noise
that the paired common-random-numbers design was built to cancel.

- **oracle / -**: minimum detected γ = 5 bps/month. γ5:+0.0031(±0.0001)* γ10:+0.0062(±0.0001)* γ15:+0.0093(±0.0002)* γ20:+0.0124(±0.0002)* γ40:+0.0248(±0.0005)*
- **attention / ic**: minimum detected γ = 40 bps/month. γ5:-0.0001(±0.0004) γ10:+0.0003(±0.0016) γ15:+0.0005(±0.0034) γ20:+0.0009(±0.0050) γ40:+0.0077(±0.0075)*
- **gated_attention / ic**: minimum detected γ = 20 bps/month. γ5:+0.0030(±0.0068) γ10:+0.0027(±0.0042) γ15:+0.0039(±0.0046) γ20:+0.0058(±0.0053)* γ40:+0.0175(±0.0069)*
- **p7_encoder / ic**: minimum detected γ = 20 bps/month. γ5:+0.0030(±0.0068) γ10:+0.0029(±0.0047) γ15:+0.0038(±0.0045) γ20:+0.0057(±0.0053)* γ40:+0.0175(±0.0070)*
- **γ=0 false positives**: 2 of 9 null runs crossed 2·SE — INVESTIGATE.
