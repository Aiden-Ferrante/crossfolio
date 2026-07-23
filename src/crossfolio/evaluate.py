"""Test-block evaluation: honest numbers over the untouched final block.

Headline metrics use non-overlapping (stride-H) monthly anchors; the
overlapping-daily Sharpe is reported as a footnoted secondary. With only ~48
monthly test points the Sharpe standard error is huge — the report says so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import Cfg
from .data.dataset import AnchorDataset, Panel, load_panel
from .data.splits import purged_split, valid_anchors
from .models import REGISTRY
from .universe import SECTORS


@dataclass
class EvalResult:
    name: str
    dates: np.ndarray        # (n_daily,) anchor dates
    weights: np.ndarray      # (n_daily, N)
    excess_daily: np.ndarray # (n_daily,) overlapping H-day excess returns
    port_m: np.ndarray       # (n_monthly,) non-overlapping H-day portfolio returns
    spy_m: np.ndarray        # (n_monthly,)
    attn: np.ndarray | None  # (heads, N, N) mean attention, or None

    @property
    def excess_m(self) -> np.ndarray:
        return self.port_m - self.spy_m

    def metrics(self, H: int) -> dict:
        ann = np.sqrt(252 / H)
        e, hhi = self.excess_m, (self.weights**2).sum(-1).mean()
        curve = np.cumprod(1 + self.port_m)
        dd = 1 - curve / np.maximum.accumulate(curve)
        return {
            "sharpe_monthly_ann": float(e.mean() / (e.std(ddof=1) + 1e-12) * ann),
            "sharpe_daily_ann": float(
                self.excess_daily.mean() / (self.excess_daily.std(ddof=1) + 1e-12) * ann
            ),
            "cum_alpha": float(curve[-1] - np.cumprod(1 + self.spy_m)[-1]),
            "max_drawdown": float(dd.max()),
            "mean_hhi": float(hhi),
            "eff_n": float(1 / hhi),
            "n_monthly": len(e),
        }


@torch.no_grad()
def evaluate_run(run_dir: Path, cfg: Cfg | None = None, panel: Panel | None = None) -> EvalResult:
    cfg = cfg or Cfg()
    panel = panel or load_panel()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model = REGISTRY[ckpt["model"]](panel.N, cfg.data.T, cfg.model).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    anchors = valid_anchors(panel.D, cfg.data.T, cfg.data.H)
    _, _, test = purged_split(anchors, cfg.split.train_frac, cfg.split.val_frac, cfg.data.H)
    ds = AnchorDataset(panel, test, cfg.data)

    weights, excess, attn_sum, port = [], [], None, []
    for i in range(len(ds)):
        X, y, y_spy = ds[i]
        w, aux = model(X.unsqueeze(0).to(device))
        w = w.squeeze(0).cpu()
        weights.append(w.numpy())
        port.append(float((w * y).sum()))
        excess.append(float((w * y).sum() - y_spy))
        if "attn" in aux:
            a = aux["attn"].squeeze(0).cpu().numpy()
            attn_sum = a if attn_sum is None else attn_sum + a

    H = cfg.data.H
    spy_all = np.array([float(ds[i][2]) for i in range(len(ds))])
    return EvalResult(
        name=ckpt["model"],
        dates=panel.dates[test],
        weights=np.array(weights),
        excess_daily=np.array(excess),
        port_m=np.array(port)[::H],
        spy_m=spy_all[::H],
        attn=None if attn_sum is None else attn_sum / len(ds),
    )


def write_report(results: list[EvalResult], cfg: Cfg, out: Path) -> str:
    H = cfg.data.H
    rows = [(r.name, r.metrics(H)) for r in results]
    lines = [
        "# crossfolio evaluation",
        "",
        f"Test block: {results[0].dates[0]} -> {results[0].dates[-1]} "
        f"({rows[0][1]['n_monthly']} non-overlapping {H}-day periods).",
        "",
        "| model | Sharpe (monthly, ann.) | Sharpe (daily, ann.)* | cum. alpha vs SPY | max DD | mean HHI | eff. N |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in rows:
        lines.append(
            f"| {name} | {m['sharpe_monthly_ann']:+.2f} | {m['sharpe_daily_ann']:+.2f} "
            f"| {m['cum_alpha']:+.1%} | {m['max_drawdown']:.1%} "
            f"| {m['mean_hhi']:.3f} | {m['eff_n']:.1f} |"
        )
    lines += [
        "",
        "\\* overlapping daily anchors — correlated samples, secondary evidence only.",
        "",
        f"**Uncertainty:** with ~{rows[0][1]['n_monthly']} independent monthly points, the standard error "
        f"of an annualized Sharpe is roughly ±{np.sqrt(12 / rows[0][1]['n_monthly']):.1f}. "
        "Differences between these models are mostly statistical ties; this repo is a "
        "learning instrument, not an edge. Survivorship bias inflates every row equally.",
        "",
        f"Config: loss={cfg.loss.name}, hhi_lambda={cfg.loss.hhi_lambda}, "
        f"T={cfg.data.T}, H={H}, seed={cfg.train.seed}.",
    ]
    text = "\n".join(lines)
    out.write_text(text)
    return text


def compare(run_dirs: list[Path], out_dir: Path | None = None) -> None:
    from .plots import plot_allocations, plot_attention, plot_equity

    cfg = Cfg()
    panel = load_panel()
    results = [evaluate_run(d, cfg, panel) for d in run_dirs]
    out_dir = out_dir or run_dirs[-1]

    print(write_report(results, cfg, out_dir / "report.md"))
    plot_equity(results, cfg, out_dir / "equity_curves.png")
    for r in results:
        if r.name != "equal_weight":
            plot_allocations(r, panel.tickers, out_dir / f"allocations_{r.name}.png")
        if r.attn is not None:
            plot_attention(r, panel.tickers, SECTORS, out_dir / "attention.png")
    print(f"\nplots -> {out_dir}")
