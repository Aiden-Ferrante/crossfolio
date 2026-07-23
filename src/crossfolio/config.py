"""All knobs in one place, as frozen dataclasses. No yaml, no hydra."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

REPO = Path(__file__).resolve().parent.parent.parent
LAKE = Path("~/Desktop/code/stocklake/lake/raw/prices").expanduser()
PANEL = REPO / "data" / "panel.npz"
RUNS = REPO / "runs"


@dataclass(frozen=True)
class DataCfg:
    T: int = 60            # trailing window, trading days
    H: int = 21            # grading horizon, trading days (also purge embargo)
    normalize: bool = True # per-window z-score of X (window-only stats, no leakage)


@dataclass(frozen=True)
class SplitCfg:
    train_frac: float = 0.70
    val_frac: float = 0.15  # test gets the remainder


@dataclass(frozen=True)
class LossCfg:
    name: str = "sharpe"    # "sharpe" | "mean_excess"
    hhi_lambda: float = 0.05


@dataclass(frozen=True)
class ModelCfg:
    d_model: int = 32
    heads: int = 4
    enc_hidden: int = 64


@dataclass(frozen=True)
class TrainCfg:
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-2
    max_epochs: int = 200
    patience: int = 15
    seed: int = 1337
    train_stride: int = 1   # 1 = every daily anchor; 5 thins near-duplicates


@dataclass(frozen=True)
class Cfg:
    data: DataCfg = field(default_factory=DataCfg)
    split: SplitCfg = field(default_factory=SplitCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)

    def dump(self) -> str:
        return json.dumps(asdict(self), indent=2)
