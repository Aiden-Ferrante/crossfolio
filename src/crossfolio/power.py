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
from .synth import cross_sectional_ic, forward_returns, make_panel, make_shocks, oracle_ic
from .train import train

GAMMAS = [0, 10, 20, 40, 80]     # bps/month
SEEDS = [0, 1, 2]
MODELS = ["linear", "attention"]
LOSSES = ["sharpe", "ic"]
N, D_FULL, D_QUICK = 120, 6712, 2000

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


def _arm_cfg(loss_name: str, seed: int) -> Cfg:
    ic = loss_name == "ic"
    return Cfg(
        data=DataCfg(normalize=False),  # time-axis z-score erases the signal
        loss=LossCfg(name=loss_name, hhi_lambda=0.0 if ic else 0.05),
        model=ModelCfg(head_init_scale=1e-2),
        train=TrainCfg(seed=1000 + seed, train_stride=2, max_epochs=80,
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
    ic_daily = cross_sectional_ic(logits, y)
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


def run_sweep(quick: bool = False) -> Path:
    gammas = [0, 80] if quick else GAMMAS
    seeds = [0] if quick else SEEDS
    D = D_QUICK if quick else D_FULL
    out = RUNS / f"power-{time.strftime('%Y%m%d-%H%M%S')}{'-quick' if quick else ''}"
    out.mkdir(parents=True)
    (out / "POWER.md").write_text(f"# Planted-signal power test\n\n{THRESHOLDS}\n")
    results_f = (out / "results.jsonl").open("w")
    results: list[dict] = []
    t_start = time.time()

    for seed in seeds:
        shocks = make_shocks(D, N, seed)
        for gamma in gammas:
            panel, meta = make_panel(gamma, shocks)
            anchors = valid_anchors(panel.D, 60, 21)
            _, _, test = purged_split(anchors, 0.70, 0.15, 21)
            orc = oracle_ic(panel, meta, test[::21], 21)
            orc_row = {"gamma": gamma, "seed": seed, "model": "oracle", "loss": "-",
                       "test_ic_monthly": float(orc.mean()),
                       "ic_se": float((1 / np.sqrt(N)) / np.sqrt(len(orc)))}
            results.append(orc_row)
            results_f.write(json.dumps(orc_row) + "\n")
            results_f.flush()
            print(f"[g={gamma:2} s={seed}] oracle IC {orc.mean():+.4f}")

            for model_name in MODELS:
                for loss_name in LOSSES:
                    cfg = _arm_cfg(loss_name, seed)
                    name = f"{out.name}/g{gamma:02d}-s{seed}-{model_name}-{loss_name}"
                    t0 = time.time()
                    run_dir = train(model_name, cfg, panel, run_name=name)
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
    for model in MODELS:
        for loss in LOSSES:
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
        if r["model"] == "oracle":
            lines.append(f"| {r['gamma']} | {r['seed']} | **oracle** | - | "
                         f"{r['test_ic_monthly']:+.4f} | - | - | - |")
        else:
            lines.append(
                f"| {r['gamma']} | {r['seed']} | {r['model']} | {r['loss']} | "
                f"{r['test_ic_monthly']:+.4f} | {'YES' if r['detected'] else 'no'} | "
                f"{r['best_epoch']} | {r['test_excess_sharpe_ann']:+.2f} |")

    se = next(r["ic_se"] for r in results if r["model"] != "oracle")
    lines += ["", f"Detection threshold: 2·SE = {2 * se:.4f}.", "", "## Verdict", ""]
    for model in MODELS:
        for loss in LOSSES:
            g, (mean, _, _) = _agg(results, model, loss)
            detected = [gi for gi, mi in zip(g, mean) if gi > 0 and mi > 2 * se]
            lines.append(f"- **{model} / {loss}**: "
                         + (f"minimum detected γ = {min(detected)} bps/month."
                            if detected else "no detection at any planted strength."))
    null_fp = [r for r in results
               if r["model"] != "oracle" and r["gamma"] == 0
               and abs(r["test_ic_monthly"]) > 2 * se]
    lines.append(f"- **γ=0 false positives**: {len(null_fp)} of "
                 f"{len([r for r in results if r['model'] != 'oracle' and r['gamma'] == 0])} "
                 "null runs crossed 2·SE"
                 + (" (within chance expectation)." if len(null_fp) <= 1 else " — INVESTIGATE."))
    out.write_text("\n".join(lines) + "\n")
