"""crossfolio CLI: build-dataset | inspect | stage0 | train | evaluate."""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(prog="crossfolio", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build-dataset", help="read the stocklake lake, write data/panel.npz")
    sub.add_parser("inspect", help="print panel + split summary")
    sub.add_parser("stage0", help="the original idea: direct trainable weights, both losses")

    p_train = sub.add_parser("train", help="train a model through the shared harness")
    p_train.add_argument("--model", required=True,
                         choices=["equal_weight", "linear", "attention"])
    p_train.add_argument("--loss", default=None, choices=["sharpe", "mean_excess"])
    p_train.add_argument("--stride", type=int, default=None, help="train-anchor stride")

    args = ap.parse_args()

    if args.cmd == "build-dataset":
        from .data.build import build_panel

        info = build_panel()
        print(f"panel: {info['path']}")
        print(f"  D={info['D']} days x N={info['N']} stocks, {info['start']} -> {info['end']}")
        print(f"  intersection dropped {info['dropped_frac']:.2%} of benchmark days")
    elif args.cmd == "inspect":
        from .data.dataset import load_panel

        panel = load_panel()
        print(panel.summary())
    elif args.cmd == "stage0":
        from .stage0 import run

        run()
    elif args.cmd == "train":
        import dataclasses

        from .config import Cfg, LossCfg, TrainCfg
        from .train import train

        cfg = Cfg()
        if args.loss:
            cfg = dataclasses.replace(cfg, loss=LossCfg(name=args.loss, hhi_lambda=cfg.loss.hhi_lambda))
        if args.stride:
            cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, train_stride=args.stride))
        train(args.model, cfg)


if __name__ == "__main__":
    main()
