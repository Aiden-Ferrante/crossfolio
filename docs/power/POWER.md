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
| 0 | 0 | **oracle** | - | -0.0183 | - | - | - |
| 0 | 0 | linear | sharpe | +0.0003 | no | 80 | +0.23 |
| 0 | 0 | linear | ic | -0.0115 | no | 11 | +1.16 |
| 0 | 0 | attention | sharpe | +0.0055 | no | 33 | +0.26 |
| 0 | 0 | attention | ic | +0.0058 | no | 21 | +0.79 |
| 10 | 0 | **oracle** | - | -0.0072 | - | - | - |
| 10 | 0 | linear | sharpe | -0.0044 | no | 74 | +0.12 |
| 10 | 0 | linear | ic | -0.0128 | no | 11 | +1.33 |
| 10 | 0 | attention | sharpe | -0.0173 | no | 57 | -0.46 |
| 10 | 0 | attention | ic | -0.0026 | no | 34 | +0.62 |
| 20 | 0 | **oracle** | - | +0.0039 | - | - | - |
| 20 | 0 | linear | sharpe | -0.0020 | no | 80 | +0.22 |
| 20 | 0 | linear | ic | -0.0136 | no | 11 | +1.50 |
| 20 | 0 | attention | sharpe | -0.0080 | no | 43 | -0.02 |
| 20 | 0 | attention | ic | +0.0226 | no | 20 | +1.80 |
| 40 | 0 | **oracle** | - | +0.0261 | - | - | - |
| 40 | 0 | linear | sharpe | +0.0013 | no | 80 | +0.40 |
| 40 | 0 | linear | ic | -0.0146 | no | 11 | +1.86 |
| 40 | 0 | attention | sharpe | +0.0018 | no | 18 | +0.98 |
| 40 | 0 | attention | ic | +0.0364 | YES | 15 | +2.33 |
| 80 | 0 | **oracle** | - | +0.0705 | - | - | - |
| 80 | 0 | linear | sharpe | -0.0037 | no | 0 | +2.82 |
| 80 | 0 | linear | ic | -0.0141 | no | 11 | +2.62 |
| 80 | 0 | attention | sharpe | +0.0399 | YES | 13 | +2.42 |
| 80 | 0 | attention | ic | +0.0649 | YES | 13 | +3.44 |
| 0 | 1 | **oracle** | - | +0.0027 | - | - | - |
| 0 | 1 | linear | sharpe | +0.0176 | no | 12 | +0.62 |
| 0 | 1 | linear | ic | +0.0217 | no | 1 | +0.31 |
| 0 | 1 | attention | sharpe | +0.0212 | no | 12 | +0.97 |
| 0 | 1 | attention | ic | +0.0174 | no | 12 | +0.54 |
| 10 | 1 | **oracle** | - | +0.0145 | - | - | - |
| 10 | 1 | linear | sharpe | -0.0153 | no | 0 | +0.16 |
| 10 | 1 | linear | ic | +0.0217 | no | 1 | +0.53 |
| 10 | 1 | attention | sharpe | +0.0127 | no | 22 | +0.58 |
| 10 | 1 | attention | ic | +0.0196 | no | 12 | +0.69 |
| 20 | 1 | **oracle** | - | +0.0262 | - | - | - |
| 20 | 1 | linear | sharpe | -0.0153 | no | 0 | +0.38 |
| 20 | 1 | linear | ic | +0.0217 | no | 1 | +0.74 |
| 20 | 1 | attention | sharpe | +0.0227 | no | 14 | +0.93 |
| 20 | 1 | attention | ic | +0.0250 | no | 12 | +0.94 |
| 40 | 1 | **oracle** | - | +0.0496 | - | - | - |
| 40 | 1 | linear | sharpe | -0.0152 | no | 0 | +0.84 |
| 40 | 1 | linear | ic | +0.0218 | no | 1 | +1.19 |
| 40 | 1 | attention | sharpe | +0.0396 | YES | 13 | +1.60 |
| 40 | 1 | attention | ic | +0.0367 | YES | 12 | +1.49 |
| 80 | 1 | **oracle** | - | +0.0964 | - | - | - |
| 80 | 1 | linear | sharpe | -0.0150 | no | 0 | +1.77 |
| 80 | 1 | linear | ic | +0.0221 | no | 1 | +2.13 |
| 80 | 1 | attention | sharpe | +0.0733 | YES | 11 | +3.22 |
| 80 | 1 | attention | ic | +0.0839 | YES | 8 | +3.34 |
| 0 | 2 | **oracle** | - | +0.0024 | - | - | - |
| 0 | 2 | linear | sharpe | -0.0020 | no | 18 | -0.02 |
| 0 | 2 | linear | ic | -0.0102 | no | 5 | -0.47 |
| 0 | 2 | attention | sharpe | -0.0037 | no | 5 | -0.23 |
| 0 | 2 | attention | ic | -0.0040 | no | 4 | -0.40 |
| 10 | 2 | **oracle** | - | +0.0140 | - | - | - |
| 10 | 2 | linear | sharpe | -0.0031 | no | 2 | -0.17 |
| 10 | 2 | linear | ic | -0.0098 | no | 5 | -0.28 |
| 10 | 2 | attention | sharpe | +0.0015 | no | 3 | -0.00 |
| 10 | 2 | attention | ic | -0.0043 | no | 4 | -0.26 |
| 20 | 2 | **oracle** | - | +0.0256 | - | - | - |
| 20 | 2 | linear | sharpe | -0.0025 | no | 2 | -0.02 |
| 20 | 2 | linear | ic | -0.0091 | no | 5 | -0.08 |
| 20 | 2 | attention | sharpe | +0.0015 | no | 3 | +0.06 |
| 20 | 2 | attention | ic | +0.0003 | no | 16 | -0.02 |
| 40 | 2 | **oracle** | - | +0.0486 | - | - | - |
| 40 | 2 | linear | sharpe | -0.0011 | no | 2 | +0.31 |
| 40 | 2 | linear | ic | -0.0083 | no | 5 | +0.32 |
| 40 | 2 | attention | sharpe | +0.0022 | no | 20 | +0.33 |
| 40 | 2 | attention | ic | +0.0165 | no | 10 | +0.67 |
| 80 | 2 | **oracle** | - | +0.0944 | - | - | - |
| 80 | 2 | linear | sharpe | -0.0047 | no | 0 | +1.41 |
| 80 | 2 | linear | ic | -0.0081 | no | 5 | +1.10 |
| 80 | 2 | attention | sharpe | +0.0465 | YES | 9 | +2.04 |
| 80 | 2 | attention | ic | +0.0569 | YES | 9 | +2.17 |

Detection threshold: 2·SE = 0.0264.

## Verdict

- **linear / sharpe**: no detection at any planted strength.
- **linear / ic**: no detection at any planted strength.
- **attention / sharpe**: minimum detected γ = 80 bps/month.
- **attention / ic**: minimum detected γ = 40 bps/month.
- **γ=0 false positives**: 0 of 12 null runs crossed 2·SE (within chance expectation).
