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

    p_eval = sub.add_parser("evaluate", help="evaluate run dir(s) on the test block")
    p_eval.add_argument("--compare", nargs="+", required=True, metavar="RUN_DIR")

    p_power = sub.add_parser("power", help="planted-signal power test sweep")
    p_power.add_argument("--quick", action="store_true", help="smoke mode: 1 seed, 2 gammas, D=2000")

    p_camp = sub.add_parser("campaign", help="long-training campaign stages")
    p_camp.add_argument("--stage", required=True, choices=["C"], help="B'/D arrive after C")
    p_camp.add_argument("--hours", type=float, default=12.0)
    p_camp.add_argument("--resume", default=None,
                        help="run dir to resume, or 'auto' for the latest")

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
    elif args.cmd == "evaluate":
        from pathlib import Path

        from .evaluate import compare

        compare([Path(p) for p in args.compare])
    elif args.cmd == "power":
        from .power import run_sweep

        run_sweep(quick=args.quick)
    elif args.cmd == "campaign":
        from .campaign import run_stage_c

        run_stage_c(args.hours, args.resume)


if __name__ == "__main__":
    main()
