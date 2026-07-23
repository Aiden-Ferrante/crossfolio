"""Panel loading and the anchor dataset.

One example per anchor day d:
  X     (N, T) float32 — trailing daily log returns, rows d-T+1 .. d (nothing after d)
  y     (N,)   float32 — forward simple returns over d+1 .. d+H
  y_spy ()     float32 — same for the benchmark
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import PANEL, Cfg, DataCfg
from .splits import purged_split, valid_anchors

EPS = 1e-8


@dataclass(frozen=True)
class Panel:
    dates: np.ndarray    # (D,) datetime64[D]
    returns: np.ndarray  # (D, N) float32 daily log returns
    spy: np.ndarray      # (D,) float32
    tickers: list[str]

    @property
    def D(self) -> int:
        return len(self.dates)

    @property
    def N(self) -> int:
        return self.returns.shape[1]

    def summary(self, cfg: Cfg | None = None) -> str:
        cfg = cfg or Cfg()
        anchors = valid_anchors(self.D, cfg.data.T, cfg.data.H)
        train, val, test = purged_split(
            anchors, cfg.split.train_frac, cfg.split.val_frac, cfg.data.H
        )
        fmt = lambda idx: f"{len(idx):>5} anchors  {self.dates[idx[0]]} -> {self.dates[idx[-1]]}"
        return "\n".join([
            f"panel: D={self.D} days x N={self.N} stocks, {self.dates[0]} -> {self.dates[-1]}",
            f"anchors (T={cfg.data.T}, H={cfg.data.H}): {len(anchors)}",
            f"  train {fmt(train)}",
            f"  val   {fmt(val)}",
            f"  test  {fmt(test)}",
            f"  (purged {len(anchors) - len(train) - len(val) - len(test)} anchors at block boundaries)",
        ])


def load_panel(path: Path = PANEL) -> Panel:
    with np.load(path, allow_pickle=False) as z:
        return Panel(
            dates=z["dates"].astype("datetime64[D]"),
            returns=z["returns"],
            spy=z["spy"],
            tickers=[str(t) for t in z["tickers"]],
        )


class AnchorDataset(Dataset):
    def __init__(self, panel: Panel, anchors: np.ndarray, cfg: DataCfg):
        self.panel = panel
        self.anchors = anchors
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, i: int):
        d = int(self.anchors[i])
        T, H = self.cfg.T, self.cfg.H
        X = self.panel.returns[d - T + 1 : d + 1].T          # (N, T), rows <= d only
        if self.cfg.normalize:
            mu = X.mean(axis=1, keepdims=True)
            sd = X.std(axis=1, keepdims=True)
            X = (X - mu) / (sd + EPS)
        fwd = self.panel.returns[d + 1 : d + 1 + H]           # (H, N)
        y = np.expm1(fwd.sum(axis=0))                          # simple returns
        y_spy = np.expm1(self.panel.spy[d + 1 : d + 1 + H].sum())
        return (
            torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
            torch.from_numpy(y.astype(np.float32)),
            torch.tensor(float(y_spy), dtype=torch.float32),
        )
