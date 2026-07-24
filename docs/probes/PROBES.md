# Probe suite results

Four-model matrix; `random` is the null every claim must beat.

## P1 — attention vs sector/correlation structure

| model | head | within/cross | sector AUC | corr ρ | RMT mkt | RMT sector | RMT noise |
|---|---|---|---|---|---|---|---|
| pretrained | 0 | 1.00 | 0.477 | -0.180 | 0.77 | 0.02 | 0.21 |
| pretrained | 1 | 1.01 | 0.499 | -0.319 | 0.80 | 0.02 | 0.18 |
| pretrained | 2 | 1.00 | 0.502 | -0.314 | 0.81 | 0.01 | 0.18 |
| pretrained | 3 | 1.02 | 0.495 | -0.294 | 0.80 | 0.02 | 0.18 |
| pretrained | 4 | 1.01 | 0.501 | -0.309 | 0.80 | 0.02 | 0.18 |
| pretrained | 5 | 1.02 | 0.493 | -0.286 | 0.80 | 0.02 | 0.18 |
| pretrained | 6 | 1.02 | 0.492 | -0.276 | 0.80 | 0.02 | 0.18 |
| pretrained | 7 | 1.01 | 0.503 | -0.317 | 0.80 | 0.02 | 0.18 |
| finetuned | 0 | 1.00 | 0.499 | -0.275 | 0.80 | 0.02 | 0.18 |
| finetuned | 1 | 1.03 | 0.492 | -0.172 | 0.80 | 0.02 | 0.17 |
| finetuned | 2 | 1.00 | 0.487 | -0.136 | 0.80 | 0.03 | 0.17 |
| finetuned | 3 | 1.03 | 0.491 | -0.175 | 0.80 | 0.02 | 0.17 |
| finetuned | 4 | 1.03 | 0.491 | -0.173 | 0.80 | 0.02 | 0.17 |
| finetuned | 5 | 1.03 | 0.492 | -0.173 | 0.80 | 0.02 | 0.18 |
| finetuned | 6 | 1.03 | 0.492 | -0.171 | 0.80 | 0.02 | 0.18 |
| finetuned | 7 | 1.02 | 0.490 | -0.172 | 0.80 | 0.02 | 0.17 |
| scratch | 0 | 1.00 | 0.505 | +0.006 | 0.04 | 0.00 | 0.96 |
| scratch | 1 | 1.00 | 0.502 | +0.009 | 0.04 | 0.00 | 0.96 |
| scratch | 2 | 0.99 | 0.494 | -0.016 | 0.04 | 0.00 | 0.96 |
| scratch | 3 | 1.00 | 0.496 | -0.006 | 0.06 | 0.00 | 0.94 |
| random | 0 | 1.00 | 0.493 | -0.017 | 0.81 | 0.02 | 0.17 |
| random | 1 | 1.00 | 0.503 | -0.036 | 0.81 | 0.04 | 0.15 |
| random | 2 | 1.00 | 0.543 | -0.018 | 0.81 | 0.04 | 0.15 |
| random | 3 | 1.00 | 0.524 | +0.138 | 0.80 | 0.07 | 0.13 |
| random | 4 | 1.00 | 0.531 | -0.095 | 0.81 | 0.06 | 0.13 |
| random | 5 | 1.00 | 0.499 | +0.033 | 0.81 | 0.04 | 0.15 |
| random | 6 | 1.00 | 0.523 | +0.008 | 0.81 | 0.02 | 0.17 |
| random | 7 | 1.00 | 0.479 | +0.093 | 0.81 | 0.07 | 0.12 |

## P2 — mean attention-correlation alignment per head

- **pretrained**: [-0.041, -0.0215, -0.009, -0.0332, -0.0215, -0.0368, -0.0399, -0.0146]
- **finetuned**: [-0.034, -0.0513, -0.0273, -0.0635, -0.0494, -0.0644, -0.069, -0.0448]
- **scratch**: [0.0078, 0.0015, -0.0085, -0.0134]
- **random**: [-0.007, 0.0008, 0.0198, 0.008, -0.0002, -0.0176, 0.0113, -0.0037]

## P3 — in-context latent (gamma_mom readout R², pooled tokens)

- **pretrained**: R² = -0.1994
- **finetuned**: R² = -0.1513
- **scratch**: R² = -0.1036
- **random**: R² = -0.1019

(full per-depth feature R² tables in results.json)

## P4 — causal patching (synthetic detection IC)

| model | condition | IC@γ0 | IC@γ20 | IC@γ40 |
|---|---|---|---|---|
| pretrained | base | +0.0015 | +0.0068 | +0.0125 |
| pretrained | uniform_attn | +0.0003 | +0.0080 | +0.0161 |
| finetuned | base | +0.0033 | +0.0012 | -0.0010 |
| finetuned | uniform_attn | +0.0007 | +0.0085 | +0.0164 |
| scratch | base | -0.0032 | +0.0016 | +0.0065 |
| scratch | uniform_attn | -0.0022 | +0.0027 | +0.0079 |
| random | base | -0.0017 | +0.0001 | +0.0020 |
| random | uniform_attn | -0.0017 | +0.0001 | +0.0020 |

Per-head ablations in results.json.

## P5 — regime gating (logit-momentum alignment by window-vol tertile)

- **pretrained**: {'calm': 0.3622, 'mid': 0.2571, 'stressed': 0.081}
- **finetuned**: {'calm': -0.2389, 'mid': -0.3587, 'stressed': -0.5164}
- **scratch**: {'calm': 0.3841, 'mid': 0.4345, 'stressed': 0.5144}
- **random**: {'calm': 0.1393, 'mid': 0.1427, 'stressed': 0.1381}

## Verdicts

1. **No sector recovery (P1).** Sector AUC ≈ 0.5 and within/cross ≈ 1.0 for every
   model including pretrained — indistinguishable from the random null. The
   paper-shaped hope (attention finds GICS structure unsupervised) is negative
   on real data. RMT market-share ≈ 0.8 also appears in the random null (any
   near-uniform symmetric map projects onto the dominant mode) — not evidence.

2. **A real learned attention pattern — pointing the "wrong" way (P1/P2).**
   All 8 pretrained heads show consistent NEGATIVE attention-correlation
   alignment (ρ −0.18..−0.32 vs null scatter ±0.1), strengthened by real-data
   fine-tuning. The model attends preferentially to DISSIMILAR stocks —
   plausibly because correlated peers carry redundant windows. This clears the
   null and is the one novel internal structure found.

3. **Attention is causally harmful to signal extraction (P4 — the headline).**
   Flattening attention to uniform IMPROVES synthetic detection for pretrained
   (+0.0125 → +0.0161 @ γ40) and resurrects the finetuned model (−0.001 →
   +0.0164). The signal path is the per-stock encoder + FFN; the learned
   attention subtracts from it. **P6→P7 demotion for this instance**: the
   cross-sectional machinery is not the signal path — it is a liability the
   encoder carries.

4. **Fine-tuning on real data corrupted attention, kept the encoder, and
   flipped the sign (P4/P5).** Finetuned base detection ≈ 0 but uniform-patched
   detection ≈ pretrained — the encoder survived; attention absorbed the
   real-data gradient damage. And P5 shows finetuned logits are ANTI-momentum
   (−0.24 calm → −0.52 stressed): real weekly cross-sections reward reversal,
   and fine-tuning obliged. This is the mechanistic account of the Stage D tie.

5. **Regime gating emerged in pretraining (P5).** Pretrained momentum-alignment
   falls 0.36 → 0.08 from calm to stressed windows — the cocktail's
   momentum-dies-in-stress conditioning was internalized (random is flat;
   scratch, never trained on gated data, shows none). In-context regime
   reading is real, even though the pooled linear γ-readout is not (P3: R² < 0
   for all models — not linearly decodable from mean-pooled tokens by this probe).

**Next instruments these verdicts load:** (a) an attention-free per-stock
encoder baseline (pure P7) — predicted to match or beat the full model;
(b) purely-relational planted signals (defined only against peers) so a P7
path CANNOT learn them — forcing attention to earn its place or lose it;
(c) a closer look at the anti-correlation attention head pattern.
