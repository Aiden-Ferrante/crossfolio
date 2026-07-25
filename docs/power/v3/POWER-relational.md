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
| 0 | 0 | **oracle** | - | -0.0029 | - | - | - |
| 0 | 0 | **p7_oracle** | - | +0.0027 | - | - | - |
| 0 | 0 | p7_encoder | ic | -0.0012 | no | 2 | +0.32 |
| 0 | 0 | attention | ic | +0.0043 | no | 13 | +0.57 |
| 0 | 0 | gated_attention | ic | -0.0012 | no | 2 | +0.32 |
| 20 | 0 | **oracle** | - | +0.0075 | - | - | - |
| 20 | 0 | **p7_oracle** | - | +0.0051 | - | - | - |
| 20 | 0 | p7_encoder | ic | -0.0005 | no | 2 | +0.48 |
| 20 | 0 | attention | ic | +0.0050 | no | 13 | +0.66 |
| 20 | 0 | gated_attention | ic | -0.0005 | no | 2 | +0.48 |
| 40 | 0 | **oracle** | - | +0.0180 | - | - | - |
| 40 | 0 | **p7_oracle** | - | +0.0080 | - | - | - |
| 40 | 0 | p7_encoder | ic | +0.0018 | no | 3 | +0.69 |
| 40 | 0 | attention | ic | +0.0024 | no | 39 | +0.12 |
| 40 | 0 | gated_attention | ic | +0.0018 | no | 3 | +0.69 |
| 80 | 0 | **oracle** | - | +0.0380 | - | - | - |
| 80 | 0 | **p7_oracle** | - | +0.0149 | - | - | - |
| 80 | 0 | p7_encoder | ic | +0.0086 | YES | 19 | +1.33 |
| 80 | 0 | attention | ic | +0.0090 | YES | 24 | +1.16 |
| 80 | 0 | gated_attention | ic | +0.0086 | YES | 19 | +1.33 |
| 120 | 0 | **oracle** | - | +0.0606 | - | - | - |
| 120 | 0 | **p7_oracle** | - | +0.0234 | - | - | - |
| 120 | 0 | p7_encoder | ic | +0.0161 | YES | 19 | +2.00 |
| 120 | 0 | attention | ic | +0.0159 | YES | 13 | +1.84 |
| 120 | 0 | gated_attention | ic | +0.0161 | YES | 19 | +2.00 |
| 0 | 1 | **oracle** | - | -0.0097 | - | - | - |
| 0 | 1 | **p7_oracle** | - | +0.0146 | - | - | - |
| 0 | 1 | p7_encoder | ic | -0.0102 | no | 0 | -0.01 |
| 0 | 1 | attention | ic | -0.0050 | no | 8 | -0.71 |
| 0 | 1 | gated_attention | ic | -0.0102 | no | 0 | -0.01 |
| 20 | 1 | **oracle** | - | +0.0010 | - | - | - |
| 20 | 1 | **p7_oracle** | - | +0.0124 | - | - | - |
| 20 | 1 | p7_encoder | ic | -0.0015 | no | 1 | +0.08 |
| 20 | 1 | attention | ic | -0.0052 | no | 8 | -0.67 |
| 20 | 1 | gated_attention | ic | -0.0015 | no | 1 | +0.08 |
| 40 | 1 | **oracle** | - | +0.0110 | - | - | - |
| 40 | 1 | **p7_oracle** | - | +0.0099 | - | - | - |
| 40 | 1 | p7_encoder | ic | -0.0030 | no | 9 | +0.10 |
| 40 | 1 | attention | ic | -0.0064 | no | 11 | -0.64 |
| 40 | 1 | gated_attention | ic | -0.0030 | no | 9 | +0.10 |
| 80 | 1 | **oracle** | - | +0.0340 | - | - | - |
| 80 | 1 | **p7_oracle** | - | +0.0036 | - | - | - |
| 80 | 1 | p7_encoder | ic | +0.0010 | no | 18 | +0.49 |
| 80 | 1 | attention | ic | -0.0060 | no | 11 | -0.45 |
| 80 | 1 | gated_attention | ic | +0.0010 | no | 18 | +0.49 |
| 120 | 1 | **oracle** | - | +0.0588 | - | - | - |
| 120 | 1 | **p7_oracle** | - | -0.0035 | - | - | - |
| 120 | 1 | p7_encoder | ic | +0.0103 | YES | 18 | +1.29 |
| 120 | 1 | attention | ic | -0.0012 | no | 16 | +0.26 |
| 120 | 1 | gated_attention | ic | +0.0103 | YES | 18 | +1.29 |
| 0 | 2 | **oracle** | - | -0.0003 | - | - | - |
| 0 | 2 | **p7_oracle** | - | +0.0070 | - | - | - |
| 0 | 2 | p7_encoder | ic | +0.0036 | no | 11 | -0.57 |
| 0 | 2 | attention | ic | -0.0068 | no | 31 | -1.13 |
| 0 | 2 | gated_attention | ic | +0.0036 | no | 11 | -0.58 |
| 20 | 2 | **oracle** | - | +0.0104 | - | - | - |
| 20 | 2 | **p7_oracle** | - | +0.0065 | - | - | - |
| 20 | 2 | p7_encoder | ic | +0.0042 | no | 11 | -0.42 |
| 20 | 2 | attention | ic | -0.0060 | no | 31 | -0.98 |
| 20 | 2 | gated_attention | ic | +0.0042 | no | 11 | -0.42 |
| 40 | 2 | **oracle** | - | +0.0212 | - | - | - |
| 40 | 2 | **p7_oracle** | - | +0.0067 | - | - | - |
| 40 | 2 | p7_encoder | ic | +0.0059 | no | 11 | -0.22 |
| 40 | 2 | attention | ic | -0.0028 | no | 57 | -0.33 |
| 40 | 2 | gated_attention | ic | +0.0059 | no | 11 | -0.22 |
| 80 | 2 | **oracle** | - | +0.0431 | - | - | - |
| 80 | 2 | **p7_oracle** | - | +0.0094 | - | - | - |
| 80 | 2 | p7_encoder | ic | +0.0125 | YES | 11 | +0.30 |
| 80 | 2 | attention | ic | -0.0050 | no | 43 | -0.58 |
| 80 | 2 | gated_attention | ic | +0.0125 | YES | 11 | +0.30 |
| 120 | 2 | **oracle** | - | +0.0662 | - | - | - |
| 120 | 2 | **p7_oracle** | - | +0.0168 | - | - | - |
| 120 | 2 | p7_encoder | ic | +0.0227 | YES | 11 | +0.97 |
| 120 | 2 | attention | ic | +0.0057 | no | 19 | +0.72 |
| 120 | 2 | gated_attention | ic | +0.0226 | YES | 11 | +0.97 |

Absolute detection threshold: 2·SE = 0.0082 (secondary — slow scores are
serially correlated across anchors, so absolute IC has far fewer effective
observations than anchor counts suggest).

## Verdict (paired contrasts — the official statistic)

Within-seed IC(γ) − IC(0) cancels the serially-correlated score-baseline noise
that the paired common-random-numbers design was built to cancel.

- **oracle / -**: minimum detected γ = 20 bps/month. γ20:+0.0106(±0.0002)* γ40:+0.0210(±0.0004)* γ80:+0.0427(±0.0018)* γ120:+0.0662(±0.0029)*
- **p7_oracle / -**: no detection at any planted strength. γ20:-0.0001(±0.0027) γ40:+0.0001(±0.0058) γ80:+0.0012(±0.0134) γ120:+0.0041(±0.0232)
- **attention / ic**: minimum detected γ = 120 bps/month. γ20:+0.0004(±0.0006) γ40:+0.0002(±0.0038) γ80:+0.0018(±0.0033) γ120:+0.0093(±0.0055)*
- **gated_attention / ic**: minimum detected γ = 40 bps/month. γ20:+0.0033(±0.0054) γ40:+0.0041(±0.0030)* γ80:+0.0100(±0.0013)* γ120:+0.0189(±0.0018)*
- **p7_encoder / ic**: minimum detected γ = 40 bps/month. γ20:+0.0033(±0.0054) γ40:+0.0041(±0.0030)* γ80:+0.0100(±0.0013)* γ120:+0.0189(±0.0018)*
- **γ=0 false positives**: 3 of 12 null runs crossed 2·SE — INVESTIGATE.
