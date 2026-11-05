"""Quick synthetic demo: does the harness recover a signal we planted?

Runs on CPU in a couple of minutes with no private data. It is a one-seed
slice of the planted-signal power test (`crossfolio power`, docs/power/):
synthetic panels with market, sector and idiosyncratic noise at the real
universe size (N=405), momentum planted at gamma bps/month, paired across
gammas by sharing the same shocks (common random numbers). One arm is trained
per panel through the same harness the real-data work uses: chronological
purged splits, early stopping on val, and grading on the untouched test block.

The arm is the attention-free `p7_encoder` with IC loss, the best detector in
the Round 3 power test. The verdict uses the power test's pre-registered rule:
detection = mean monthly test rank-IC > 2*SE. gamma=0 is the null control, and
"oracle" is the IC of the true planted z-score on the same test anchors, the
ceiling any model could reach.

`--check` makes the run a regression test (used in CI): it fails unless the
null is not detected and the strongest planted signal is.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from .config import RUNS
from .data.splits import purged_split, valid_anchors
from .power import _arm_cfg, _eval_test, _ranks
from .synth import cross_sectional_ic, forward_returns, make_panel, make_shocks
from .train import train
from .universe import N as UNIVERSE_N

GAMMAS = (0, 40, 80)   # bps/month; 0 is the null control
D = 5000               # trading days per synthetic panel (the power test's D_FULL)
ARM = "p7_encoder"
LOSS = "ic"


def run_demo(seed: int = 0, check: bool = False) -> list[dict]:
    # The arm is ~18k params: more than a few CPU threads is pure overhead
    # (measured 1.7 min at 20 threads vs under 1 min at 4 on the author's box).
    torch.set_num_threads(min(4, torch.get_num_threads()))
    t_start = time.time()
    out = RUNS / f"demo-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    shocks = make_shocks(D, UNIVERSE_N, seed)
    cfg = _arm_cfg(LOSS, seed)
    T, H = cfg.data.T, cfg.data.H
    rows = []
    for gamma in GAMMAS:
        panel, meta = make_panel(gamma, shocks)
        anchors = valid_anchors(panel.D, T, H)
        _, _, test = purged_split(anchors, cfg.split.train_frac, cfg.split.val_frac, H)
        y, _ = forward_returns(panel, test[::H], H)
        oracle = float(cross_sectional_ic(_ranks(meta["z"][test[::H]]), _ranks(y)).mean())
        t0 = time.time()
        run_dir = train(ARM, cfg, panel, run_name=f"{out.name}/g{gamma:03d}-s{seed}")
        res = _eval_test(run_dir, panel, cfg)
        rows.append({"gamma": gamma, "seed": seed, "model": ARM, "loss": LOSS,
                     "oracle_ic": oracle, **res, "wall_s": round(time.time() - t0, 1)})

    with (out / "results.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"\nPlanted-signal demo: N={UNIVERSE_N} synthetic stocks, D={D} days, "
          f"seed {seed}, {ARM}/{LOSS} loss")
    print("detection rule (pre-registered): mean monthly test rank-IC > 2*SE\n")
    print(f"{'gamma bps/mo':>12} {'oracle IC':>10} {'model IC':>9} {'2*SE':>7} "
          f"{'recovery':>9}  verdict")
    for r in rows:
        rec = (f"{r['test_ic_monthly'] / r['oracle_ic']:>8.0%}"
               if r["gamma"] and r["oracle_ic"] > 0 else f"{'-':>8}")
        verdict = "DETECTED" if r["detected"] else "null"
        if r["gamma"] == 0:
            verdict += " (control: should be null)"
        print(f"{r['gamma']:>12} {r['oracle_ic']:>+10.4f} {r['test_ic_monthly']:>+9.4f} "
              f"{2 * r['ic_se']:>7.4f} {rec} {verdict}")
    minutes = (time.time() - t_start) / 60
    print(f"\n{minutes:.1f} min -> {out}")

    if check:
        null, strong = rows[0], rows[-1]
        problems = []
        if null["detected"]:
            problems.append("the gamma=0 null was 'detected' (false positive)")
        if not strong["detected"]:
            problems.append(f"the gamma={strong['gamma']} planted signal was missed")
        if not np.isfinite([r["test_ic_monthly"] for r in rows]).all():
            problems.append("non-finite test IC")
        if problems:
            raise SystemExit("demo check failed: " + "; ".join(problems))
        print("check passed: null not detected, strongest planted signal detected")
    return rows
