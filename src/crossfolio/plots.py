"""Matplotlib outputs: equity curves, allocation-over-time, attention heatmaps."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _sector_order(tickers: list[str], sectors: dict[str, str]) -> tuple[np.ndarray, list[int], list[str]]:
    """Indices sorted by (sector, ticker), sector boundary positions, labels."""
    order = np.array(sorted(range(len(tickers)), key=lambda i: (sectors[tickers[i]], tickers[i])))
    sorted_secs = [sectors[tickers[i]] for i in order]
    bounds = [i for i in range(1, len(sorted_secs)) if sorted_secs[i] != sorted_secs[i - 1]]
    return order, bounds, sorted_secs


def plot_equity(results, cfg, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    spy_curve = np.cumprod(1 + results[0].spy_m)
    x = results[0].dates[:: cfg.data.H][: len(spy_curve)]
    ax.plot(x, spy_curve, label="SPY", color="black", lw=2, ls="--")
    for r in results:
        ax.plot(x, np.cumprod(1 + r.port_m), label=r.name, lw=1.5)
    ax.set_yscale("log")
    ax.set_ylabel("growth of $1 (log scale)")
    ax.set_title("Test block: monthly-rebalanced portfolios vs SPY")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_allocations(result, tickers: list[str], out: Path) -> None:
    from .universe import SECTORS

    order, bounds, _ = _sector_order(tickers, SECTORS)
    W = result.weights[:, order].T  # (N, time)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(W, aspect="auto", cmap="viridis",
                   extent=[0, W.shape[1], W.shape[0], 0])
    for b in bounds:
        ax.axhline(b, color="white", lw=0.5, alpha=0.6)
    ax.set_yticks(np.arange(len(order)) + 0.5)
    ax.set_yticklabels([tickers[i] for i in order], fontsize=4)
    n_x = W.shape[1]
    xt = np.linspace(0, n_x - 1, 6, dtype=int)
    ax.set_xticks(xt)
    ax.set_xticklabels([str(result.dates[i]) for i in xt], fontsize=7)
    ax.set_title(f"{result.name}: allocation over test block (tickers grouped by sector)")
    fig.colorbar(im, ax=ax, label="weight")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_attention(result, tickers: list[str], sectors: dict[str, str], out: Path) -> None:
    """Per-head mean attention over the test block, sector-sorted with boundary
    gridlines — the interp question: did heads recover sector structure?"""
    order, bounds, sorted_secs = _sector_order(tickers, sectors)
    heads = result.attn.shape[0]
    fig, axes = plt.subplots(1, heads, figsize=(5 * heads, 5.4))
    sec_ticks = {}
    for i, s in enumerate(sorted_secs):
        sec_ticks.setdefault(s, []).append(i)
    tick_pos = [np.mean(v) for v in sec_ticks.values()]
    tick_lab = [s[:12] for s in sec_ticks]
    for h, ax in enumerate(np.atleast_1d(axes)):
        A = result.attn[h][np.ix_(order, order)]
        ax.imshow(A, cmap="magma")
        for b in bounds:
            ax.axhline(b - 0.5, color="cyan", lw=0.4, alpha=0.7)
            ax.axvline(b - 0.5, color="cyan", lw=0.4, alpha=0.7)
        ax.set_title(f"head {h}", fontsize=9)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab, rotation=90, fontsize=6)
        if h == 0:
            ax.set_yticks(tick_pos)
            ax.set_yticklabels(tick_lab, fontsize=6)
        else:
            ax.set_yticks([])
    fig.suptitle("mean attention over test block, sorted by sector (query row -> key col)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
