# Stage D: walk-forward real-data campaign

Ensemble (mean logits across seeds) test rank-IC per year:

| year | scratch | full | head |
|---|---|---|---|
| 2014 | -0.0056 | +0.0026 | -0.0051 |
| 2015 | +0.0005 | -0.0039 | +0.0009 |
| 2016 | -0.0032 | -0.0373 | -0.0418 |
| 2017 | +0.0218 | +0.0167 | +0.0290 |
| 2018 | +0.0279 | +0.0429 | +0.0492 |
| 2019 | +0.0312 | +0.0357 | +0.0186 |
| 2020 | +0.0573 | +0.0331 | +0.0444 |
| 2021 | +0.0262 | +0.0256 | +0.0112 |
| 2022 | -0.0444 | -0.0303 | -0.0267 |
| 2023 | +0.0177 | -0.0075 | -0.0015 |
| 2024 | +0.0199 | +0.0200 | +0.0142 |
| 2025 | -0.0258 | +0.0034 | -0.0165 |
| 2026 | -0.0506 | -0.0163 | -0.0117 |

| | mean | 2·SE over years |
|---|---|---|
| scratch | +0.0056 | ±0.0173 |
| full | +0.0065 | ±0.0140 |
| head | +0.0049 | ±0.0148 |

Per-anchor rank-IC SE ≈ 0.05/√(~50 weeks/yr) — treat single-year
cells as noise; the mean row over 13 years is the result. Survivorship
bias inflates all protocols equally; differences between columns are
the sim2real claim.
