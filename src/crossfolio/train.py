"""Shared training harness — every model (including equal_weight) runs through
it, so all runs produce identical artifacts under runs/<ts>-<model>-<loss>/:
config.json, best.pt (state_dict + epoch + val_sharpe), history.jsonl.

Early stopping on validation Sharpe over ALL daily val anchors (the val block
is too short for monthly-only: ~10 non-overlapping points). Train anchors are
shuffled within the train block each epoch — blocks never mix.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import RUNS, Cfg
from .data.dataset import AnchorDataset, Panel, load_panel
from .data.splits import purged_split, valid_anchors
from .losses import make_loss
from .models import REGISTRY


def _seed_all(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


@torch.no_grad()
def eval_metrics(model, loader, device) -> dict[str, float]:
    """Deterministic val metrics over every anchor in the loader: Sharpe of
    excess returns (portfolio grade) and mean per-anchor IC (cross-sectional
    grade). Both always computed; early stopping picks the loss-matched one."""
    from .losses import mean_ic

    model.eval()
    excess, ics = [], []
    for X, y, y_spy in loader:
        X, y, y_spy = X.to(device), y.to(device), y_spy.to(device)
        w, aux = model(X)
        excess.append((w * y).sum(-1) - y_spy)
        ics.append(mean_ic(aux["logits"], y) * len(X))
    e = torch.cat(excess)
    return {
        "sharpe": float(e.mean() / (e.std(unbiased=True) + 1e-8)),
        "ic": float(torch.stack(ics).sum() / len(loader.dataset)),
    }


def train(model_name: str, cfg: Cfg | None = None, panel: Panel | None = None,
          run_name: str | None = None) -> Path:
    cfg = cfg or Cfg()
    panel = panel or load_panel()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g = _seed_all(cfg.train.seed)

    anchors = valid_anchors(panel.D, cfg.data.T, cfg.data.H)
    tr, va, _ = purged_split(anchors, cfg.split.train_frac, cfg.split.val_frac, cfg.data.H)
    tr = tr[:: cfg.train.train_stride]
    dl_kw = dict(batch_size=cfg.train.batch_size, num_workers=0)
    train_dl = DataLoader(AnchorDataset(panel, tr, cfg.data), shuffle=True,
                          drop_last=True, generator=g, **dl_kw)
    val_dl = DataLoader(AnchorDataset(panel, va, cfg.data), shuffle=False, **dl_kw)

    model = REGISTRY[model_name](panel.N, cfg.data.T, cfg.model).to(device)
    loss_fn = make_loss(cfg.loss.name, cfg.loss.hhi_lambda)
    n_params = sum(p.numel() for p in model.parameters())

    run_dir = RUNS / (run_name or f"{time.strftime('%Y%m%d-%H%M%S')}-{model_name}-{cfg.loss.name}")
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(cfg.dump())
    history = (run_dir / "history.jsonl").open("w")

    def checkpoint(epoch: int, val_metric: float, name: str = "best.pt") -> None:
        torch.save(
            {"model": model_name, "state_dict": model.state_dict(),
             "epoch": epoch, "val_metric": val_metric, "stop_metric": stop_on},
            run_dir / name,
        )

    # stop on the metric the loss can actually move — stopping the IC arm on
    # val Sharpe would select on mismatched noise and recreate the epoch-0 bug
    stop_on = "ic" if cfg.loss.name == "ic" else "sharpe"
    max_epochs = 0 if n_params == 0 else cfg.train.max_epochs
    metrics = eval_metrics(model, val_dl, device)
    best, best_epoch = metrics[stop_on], 0
    checkpoint(0, best)
    print(f"{model_name}: {n_params} params on {device}, "
          f"{len(tr)} train / {len(va)} val anchors, loss={cfg.loss.name}")
    print(f"  epoch   0: val_{stop_on} {best:+.4f} (untrained)")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay) if max_epochs else None
    for epoch in range(1, max_epochs + 1):
        model.train()
        for X, y, y_spy in train_dl:
            X, y, y_spy = X.to(device), y.to(device), y_spy.to(device)
            opt.zero_grad()
            w, aux = model(X)
            loss = loss_fn(w, y, y_spy, logits=aux["logits"])
            assert torch.isfinite(loss), f"non-finite loss at epoch {epoch}"
            loss.backward()
            opt.step()

        metrics = eval_metrics(model, val_dl, device)
        vs = metrics[stop_on]
        history.write(json.dumps({"epoch": epoch, **{f"val_{k}": v for k, v in metrics.items()}}) + "\n")
        checkpoint(epoch, vs, "last.pt")  # kept for interp: the overfit weights
        if vs > best:
            best, best_epoch = vs, epoch
            checkpoint(epoch, vs)
            print(f"  epoch {epoch:3}: val_{stop_on} {vs:+.4f} *")
        elif epoch - best_epoch >= cfg.train.patience:
            print(f"  epoch {epoch:3}: val_{stop_on} {vs:+.4f} — early stop "
                  f"(best {best:+.4f} @ epoch {best_epoch})")
            break

    history.close()
    print(f"  -> {run_dir}")
    return run_dir
