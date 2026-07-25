"""The planted-signal power test: can the harness recover a signal we KNOW
exists, and at what minimum strength? Sweeps gamma x seed x model x loss with
paired panels (common random numbers per seed) and reports a power curve.

Success is a measurement, not a win: "the sharpe arm detects nothing at any
gamma" is a clean verdict (supervision starvation confirmed), and so is the
opposite. Pre-registered thresholds are written to POWER.md before any run.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch

from .config import RUNS, Cfg, DataCfg, LossCfg, ModelCfg, TrainCfg
from .data.dataset import AnchorDataset, Panel
from .data.splits import purged_split, valid_anchors
from .models import REGISTRY
from .synth import (cross_sectional_ic, forward_returns, make_panel,
                    make_relational_panel, make_shocks)

SIGNALS = {"momentum": (make_panel, [0, 5, 10, 15, 20, 40]),
           # relational z has a smaller per-bps IC slope (L=20 mates-mean); grid shifted up
           "relational": (make_relational_panel, [0, 20, 40, 80, 120])}
ROUND3_ARMS = ["p7_encoder", "attention", "gated_attention"]
ROUND4_ARMS = ["p7_encoder", "gated_attention", "corr_bias_attention", "gated_curriculum"]
CURRICULUM_LADDER = (500.0, 250.0, 120.0)  # bps, strong -> weaker, then the target panel
from .train import train

from .universe import N as UNIVERSE_N

GAMMAS = [0, 5, 10, 15, 20, 40]  # bps/month (v2 grid; v1 was {0,10,20,40,80} at N=120)
SEEDS = [0, 1, 2]
MODELS = ["linear", "attention"]
LOSSES = ["sharpe", "ic"]
# 1-seed linear control: linear was blind at every strength in v1; keep it as
# a control arm without paying 3 seeds for it.
ARM_SEEDS = {"attention": SEEDS, "linear": [0]}
N, D_FULL, D_QUICK = UNIVERSE_N, 5000, 2000

THRESHOLDS = """## Pre-registered thresholds (written before any run)

- Headline metric: mean test IC over non-overlapping stride-H anchors;
  SE = (1/sqrt(N)) / sqrt(n_monthly).
- **Detection** = mean test IC > 2*SE.
- **gamma=0 acceptance** (false-positive check, all null runs): |test IC| < 2*SE
  AND best_epoch small AND no val improvement beyond noise. At a 2-sigma
  threshold, ~0-1 of the null runs may cross by chance — expected, and said here
  in advance so it can't become a post-hoc story.
- Recovery fraction = model test IC / same-panel empirical oracle test IC.
"""


def _ranks(a: np.ndarray) -> np.ndarray:
    """Per-row ranks along the last axis; feeding these to cross_sectional_ic
    gives Spearman rank-IC — the fat-tail-robust headline metric at H=5."""
    return a.argsort(-1).argsort(-1).astype(np.float64)


def _arm_cfg(loss_name: str, seed: int, pretrained: bool = False) -> Cfg:
    ic = loss_name == "ic"
    if pretrained:
        # must match the Stage C pretraining architecture (campaign.PretrainCfg);
        # id_embed is fresh (absent from the checkpoint), everything else loads.
        model = ModelCfg(d_model=256, heads=8, enc_hidden=512, n_blocks=4,
                         use_id_embed=True, id_embed_init_scale=0.0,
                         head_init_scale=1e-2)
        lr = 1e-4  # full fine-tune at low lr
    else:
        model = ModelCfg(head_init_scale=1e-2)
        lr = 1e-3
    return Cfg(
        data=DataCfg(normalize=False),  # time-axis z-score erases the signal
        loss=LossCfg(name=loss_name, hhi_lambda=0.0 if ic else 0.05),
        model=model,
        train=TrainCfg(seed=1000 + seed, lr=lr, train_stride=2, max_epochs=80,
                       weight_decay=0.0 if ic else 1e-2),
    )


@torch.no_grad()
def _eval_test(run_dir: Path, panel: Panel, cfg: Cfg) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model = REGISTRY[ckpt["model"]](panel.N, cfg.data.T, cfg.model).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    anchors = valid_anchors(panel.D, cfg.data.T, cfg.data.H)
    _, _, test = purged_split(anchors, cfg.split.train_frac, cfg.split.val_frac, cfg.data.H)
    ds = AnchorDataset(panel, test, cfg.data)
    logits, weights = [], []
    for i in range(len(ds)):
        X, _, _ = ds[i]
        w, aux = model(X.unsqueeze(0).to(device))
        logits.append(aux["logits"].squeeze(0).cpu().numpy())
        weights.append(w.squeeze(0).cpu().numpy())
    logits, weights = np.array(logits), np.array(weights)

    H = cfg.data.H
    y, y_spy = forward_returns(panel, test, H)
    ic_daily = cross_sectional_ic(_ranks(logits), _ranks(y))  # rank-IC (Spearman)
    ic_monthly = ic_daily[::H]
    excess_m = (weights[::H] * y[::H]).sum(-1) - y_spy[::H]
    se = (1 / np.sqrt(panel.N)) / np.sqrt(len(ic_monthly))
    return {
        "test_ic_monthly": float(ic_monthly.mean()),
        "test_ic_daily": float(ic_daily.mean()),
        "ic_se": float(se),
        "detected": bool(ic_monthly.mean() > 2 * se),
        "test_excess_sharpe_ann": float(
            excess_m.mean() / (excess_m.std(ddof=1) + 1e-12) * np.sqrt(252 / H)
        ),
        "best_epoch": int(ckpt["epoch"]),
        "n_monthly": int(len(ic_monthly)),
    }


def _p7_oracle_ic(panel, anchors, H: int) -> float:
    """Best own-window-only ridge (trailing cum returns at 4 lookbacks) —
    measures the per-stock leak of a relational signal. In-sample fit on the
    first half of anchors, rank-IC on the second half."""
    cum = np.cumsum(panel.returns.astype(np.float64), 0)
    y, _ = forward_returns(panel, anchors, H)
    feats = np.stack([cum[anchors] - cum[anchors - l] for l in (5, 10, 20, 60)], -1)
    half = len(anchors) // 2
    Z = feats[:half].reshape(-1, 4)
    W = np.linalg.solve(Z.T @ Z + 1e-2 * np.eye(4), Z.T @ y[:half].reshape(-1, 1))
    pred = (feats[half:].reshape(-1, 4) @ W).reshape(-1, panel.N)
    return float(cross_sectional_ic(_ranks(pred), _ranks(y[half:])).mean())


def run_sweep(quick: bool = False, pretrained: Path | None = None,
              signal: str = "momentum", arms: str = "classic") -> Path:
    make, grid = SIGNALS[signal]
    gammas = [0, grid[-1]] if quick else grid
    seeds = [0] if quick else SEEDS
    D = D_QUICK if quick else D_FULL
    base = Cfg()
    T, H = base.data.T, base.data.H
    init_state = None
    if pretrained is not None:
        ck = torch.load(pretrained, map_location="cpu", weights_only=False)
        init_state = ck["ema"]  # EMA weights are the pretrained artifact
        print(f"B': fine-tuning from {pretrained} (step {ck['step']})")
    models = {"round3": ROUND3_ARMS, "round4": ROUND4_ARMS}.get(arms, MODELS)
    arm_seeds = {m: SEEDS for m in models} if arms != "classic" else ARM_SEEDS
    losses = ["ic"] if arms != "classic" else LOSSES  # sharpe is starved; rounds are ic-only
    out = RUNS / (f"power-{time.strftime('%Y%m%d-%H%M%S')}"
                  + (f"-{signal}" if signal != "momentum" else "")
                  + (f"-{arms}" if arms != "classic" else "")
                  + ("-pretrained" if pretrained else "")
                  + ("-quick" if quick else ""))
    out.mkdir(parents=True)
    (out / "POWER.md").write_text(f"# Planted-signal power test\n\n{THRESHOLDS}\n")
    results_f = (out / "results.jsonl").open("w")
    results: list[dict] = []
    t_start = time.time()

    for seed in seeds:
        shocks = make_shocks(D, N, seed)
        for gamma in gammas:
            panel, meta = make(gamma, shocks)
            anchors = valid_anchors(panel.D, T, H)
            _, _, test = purged_split(anchors, 0.70, 0.15, H)
            y_orc, _ = forward_returns(panel, test[::H], H)
            orc = cross_sectional_ic(_ranks(meta["z"][test[::H]]), _ranks(y_orc))
            orc_row = {"gamma": gamma, "seed": seed, "model": "oracle", "loss": "-",
                       "test_ic_monthly": float(orc.mean()),
                       "ic_se": float((1 / np.sqrt(N)) / np.sqrt(len(orc)))}
            results.append(orc_row)
            results_f.write(json.dumps(orc_row) + "\n")
            results_f.flush()
            print(f"[g={gamma:2} s={seed}] oracle IC {orc.mean():+.4f}")
            if signal == "relational":
                p7row = {"gamma": gamma, "seed": seed, "model": "p7_oracle",
                         "loss": "-", "test_ic_monthly": _p7_oracle_ic(panel, test[::H], H),
                         "ic_se": orc_row["ic_se"]}
                results.append(p7row)
                results_f.write(json.dumps(p7row) + "\n")
                print(f"[g={gamma:2} s={seed}] p7-oracle IC {p7row['test_ic_monthly']:+.4f}")

            for model_name in models:
                if pretrained is not None and model_name != "attention":
                    continue  # the checkpoint is an attention model
                if seed not in arm_seeds[model_name]:
                    continue
                for loss_name in losses:
                    cfg = _arm_cfg(loss_name, seed, pretrained=pretrained is not None)
                    name = f"{out.name}/g{gamma:03d}-s{seed}-{model_name}-{loss_name}"
                    t0 = time.time()
                    real_model = model_name
                    ladder = None
                    if model_name == "gated_curriculum":
                        real_model = "gated_attention"
                        ladder = [make(gl, shocks)[0] for gl in CURRICULUM_LADDER]
                    run_dir = train(real_model, cfg, panel, run_name=name,
                                    init_state=init_state, curriculum_panels=ladder)
                    row = {"gamma": gamma, "seed": seed, "model": model_name,
                           "loss": loss_name, **_eval_test(run_dir, panel, cfg),
                           "wall_s": round(time.time() - t0, 1)}
                    results.append(row)
                    results_f.write(json.dumps(row) + "\n")
                    results_f.flush()
                    print(f"[g={gamma:2} s={seed}] {model_name}/{loss_name}: "
                          f"IC {row['test_ic_monthly']:+.4f} "
                          f"({'DETECTED' if row['detected'] else 'null'}), "
                          f"best_epoch {row['best_epoch']}, {row['wall_s']}s")

    results_f.close()
    _plot(results, out / "power_curve.png")
    _report(results, out / "POWER.md")
    print(f"\nsweep done in {(time.time() - t_start) / 60:.1f} min -> {out}")
    return out


def _agg(results, model, loss):
    """Per-gamma (mean, min, max) of test IC across seeds for one arm."""
    rows = [r for r in results if r["model"] == model and r.get("loss") == loss]
    gammas = sorted({r["gamma"] for r in rows})
    per_g = [[r["test_ic_monthly"] for r in rows if r["gamma"] == g] for g in gammas]
    return gammas, ([np.mean(v) for v in per_g], [np.min(v) for v in per_g],
                    [np.max(v) for v in per_g])


def _plot(results: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    g_o, (m_o, lo_o, hi_o) = _agg(results, "oracle", "-")
    ax.plot(g_o, m_o, "k--", lw=2, label="oracle (planted z)")
    ax.fill_between(g_o, lo_o, hi_o, color="k", alpha=0.1)
    if any(r["model"] == "p7_oracle" for r in results):
        g_p, (m_p, _, _) = _agg(results, "p7_oracle", "-")
        ax.plot(g_p, m_p, color="gray", ls="--", lw=1.5, label="p7-oracle (own-window leak)")
    for model, loss in sorted({(r["model"], r.get("loss")) for r in results
                               if r["model"] not in ("oracle", "p7_oracle")}):
        g, (mean, lo, hi) = _agg(results, model, loss)
        (line,) = ax.plot(g, mean, marker="o", label=f"{model} / {loss}")
        ax.fill_between(g, lo, hi, color=line.get_color(), alpha=0.15)
    se = next(r["ic_se"] for r in results if r["model"] != "oracle")
    ax.axhline(2 * se, color="red", ls=":", lw=1, label="detection (2·SE)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("planted signal strength γ (bps/month)")
    ax.set_ylabel("test IC (non-overlapping monthly anchors)")
    ax.set_title("Power curve: can the harness recover a planted signal?")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _report(results: list[dict], out: Path) -> None:
    lines = [out.read_text(), "## Results", "",
             "| γ (bps) | seed | model | loss | test IC | detected | best epoch | excess Sharpe |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        if r["model"] in ("oracle", "p7_oracle"):
            lines.append(f"| {r['gamma']} | {r['seed']} | **{r['model']}** | - | "
                         f"{r['test_ic_monthly']:+.4f} | - | - | - |")
        else:
            lines.append(
                f"| {r['gamma']} | {r['seed']} | {r['model']} | {r['loss']} | "
                f"{r['test_ic_monthly']:+.4f} | {'YES' if r['detected'] else 'no'} | "
                f"{r['best_epoch']} | {r['test_excess_sharpe_ann']:+.2f} |")

    se = next(r["ic_se"] for r in results if r["model"] != "oracle")
    lines += ["", f"Absolute detection threshold: 2·SE = {2 * se:.4f} (secondary — slow scores are",
              "serially correlated across anchors, so absolute IC has far fewer effective",
              "observations than anchor counts suggest).", "",
              "## Verdict (paired contrasts — the official statistic)", "",
              "Within-seed IC(γ) − IC(0) cancels the serially-correlated score-baseline noise",
              "that the paired common-random-numbers design was built to cancel.", ""]

    def paired(model, loss):
        out = []
        rows = {(r["gamma"], r["seed"]): r["test_ic_monthly"] for r in results
                if r["model"] == model and r.get("loss") == loss}
        gammas = sorted({g for g, _ in rows if g > 0})
        seeds = sorted({s for g, s in rows if g == 0})
        for g in gammas:
            d = [rows[(g, s)] - rows[(0, s)] for s in seeds if (g, s) in rows]
            if len(d) > 1:
                m, sem = float(np.mean(d)), float(np.std(d, ddof=1) / np.sqrt(len(d)))
                out.append((g, m, 2 * sem, m > 2 * sem and m > 0))
        return out

    arms = [("oracle", "-"), ("p7_oracle", "-")] + sorted(
        {(r["model"], r.get("loss")) for r in results
         if r["model"] not in ("oracle", "p7_oracle")})
    for model, loss in arms:
        rows_p = paired(model, loss)
        if not rows_p:
            continue
        detected = [g for g, m, thr, det in rows_p if det]
        detail = " ".join(f"γ{g}:{m:+.4f}(±{thr:.4f}){'*' if det else ''}"
                          for g, m, thr, det in rows_p)
        lines.append(f"- **{model} / {loss}**: "
                     + (f"minimum detected γ = {min(detected)} bps/month. " if detected
                        else "no detection at any planted strength. ")
                     + detail)
    null_fp = [r for r in results
               if r["model"] != "oracle" and r["gamma"] == 0
               and abs(r["test_ic_monthly"]) > 2 * se]
    lines.append(f"- **γ=0 false positives**: {len(null_fp)} of "
                 f"{len([r for r in results if r['model'] != 'oracle' and r['gamma'] == 0])} "
                 "null runs crossed 2·SE"
                 + (" (within chance expectation)." if len(null_fp) <= 1 else " — INVESTIGATE."))
    out.write_text("\n".join(lines) + "\n")
