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
    # Campaign v2 defaults (v1 was T=60, H=21, normalize=True — see docs/power/).
    T: int = 120            # trailing window: Stage C plants lookbacks up to 120d
    H: int = 5              # weekly grading: 4x less target overlap, embargo 5
    # Time-axis window z-scoring provably erases momentum-level signals
    # (tests/test_synth.py regression) — off by default since the power test.
    normalize: bool = False


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
    n_blocks: int = 1
    # Off during sim2real pretraining: synthetic stocks have no persistent
    # identity, so an ID embedding is a pure memorization channel.
    use_id_embed: bool = True
    # Scale on the fresh id_embed init. MUST be 0.0 when loading a checkpoint
    # pretrained without the embedding: a default N(0,1) embedding added to the
    # token stream drowns the pretrained representations (found the hard way).
    id_embed_init_scale: float = 1.0
    # scale on the final head layer's default init; ~1e-2 starts the model at
    # (near) equal weight so it must earn deviations. NOT exact zero: constant
    # logits make Pearson corr 0/0 and can zero the Sharpe std.
    head_init_scale: float = 1.0


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
