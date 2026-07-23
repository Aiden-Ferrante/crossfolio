"""crossfolio CLI: build-dataset | inspect | stage0 | train | evaluate."""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(prog="crossfolio", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build-dataset", help="read the stocklake lake, write data/panel.npz")
    sub.add_parser("inspect", help="print panel + split summary")

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


if __name__ == "__main__":
    main()
