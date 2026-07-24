"""The long-training campaign runner (Stage C: sim2real pretraining).

Design (see plan + docs): infinite stream of randomized cocktail panels feeds a
scaled cross-sectional attention model (no ID embedding — it's N-agnostic and
permutation-invariant). The stream is stateless-by-index: panel i is fully
determined by rng([base_seed, i]), so resume only needs `next_panel_index`.

Budget-driven (--hours), checkpoint every 15 min + on SIGTERM (atomic saves),
eval every 30 min against a frozen synthetic val suite (known gammas — the live
detection floor) plus zero-shot real-data val rank-IC. Statistical continuity
on resume, not bitwise (bf16/cuBLAS nondeterminism accepted).
"""

from __future__ import annotations

import copy
import json
import os
import queue
import signal
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

from .config import RUNS, Cfg, DataCfg, ModelCfg
from .data.splits import purged_split, valid_anchors
from .losses import mean_ic, rank_ic
from .models import REGISTRY
from .synth import make_cocktail_panel, make_panel, make_shocks


@dataclass(frozen=True)
class PretrainCfg:
    N: int = 405              # match the real universe geometry
    D: int = 1500
    T: int = 120
    H: int = 5
    d_model: int = 256
    heads: int = 8
    n_blocks: int = 4
    enc_hidden: int = 512
    batch: int = 512
    lr: float = 3e-4
    warmup_steps: int = 1000
    ema_decay: float = 0.999
    weight_decay: float = 0.01
    base_seed: int = 777
    queue_panels: int = 32
    workers: int = 2
    ckpt_every_s: int = 900
    eval_every_s: int = 1800
    keep_last: int = 3
    val_gammas: tuple = (0, 10, 20, 40)
    val_panels_per_gamma: int = 8


def _model_cfg(p: PretrainCfg) -> ModelCfg:
    return ModelCfg(d_model=p.d_model, heads=p.heads, enc_hidden=p.enc_hidden,
                    n_blocks=p.n_blocks, use_id_embed=False, head_init_scale=1e-2)


class PanelStream:
    """Background workers generate cocktail panels by index; statelessly resumable."""

    def __init__(self, p: PretrainCfg, start_index: int):
        self.p = p
        self.next_index = start_index
        self._idx_q: queue.Queue = queue.Queue()
        self._out_q: queue.Queue = queue.Queue(maxsize=p.queue_panels)
        self._stop = threading.Event()
        for i in range(p.queue_panels):
            self._idx_q.put(start_index + i)
        self.next_index = start_index + p.queue_panels
        self._threads = [threading.Thread(target=self._work, daemon=True)
                         for _ in range(p.workers)]
        for t in self._threads:
            t.start()

    def _work(self):
        while not self._stop.is_set():
            try:
                i = self._idx_q.get(timeout=0.5)
            except queue.Empty:
                continue
            panel = make_cocktail_panel(i, self.p.base_seed, self.p.N, self.p.D)
            self._out_q.put((i, panel.returns))

    def get(self) -> tuple[int, np.ndarray]:
        i, r = self._out_q.get()
        self._idx_q.put(self.next_index)
        self.next_index += 1
        return i, r

    def stop(self):
        self._stop.set()


def _windows_and_targets(r: torch.Tensor, T: int, H: int):
    """r (D, N) on device -> X (A, N, T), y (A, N) for all valid anchors."""
    D = r.shape[0]
    anchors = torch.arange(T - 1, D - H, device=r.device)
    Xall = r.T.unfold(1, T, 1)                  # (N, D-T+1, T)
    X = Xall[:, anchors - (T - 1)].transpose(0, 1)  # (A, N, T)
    cum = torch.cumsum(r, dim=0)
    y = torch.expm1(cum[anchors + H] - cum[anchors])  # (A, N)
    return X, y


class Pretrainer:
    def __init__(self, p: PretrainCfg, run_dir: Path):
        self.p = p
        self.run_dir = run_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = REGISTRY["attention"](p.N, p.T, _model_cfg(p)).to(self.device)
        self.ema = copy.deepcopy(self.model).eval()
        for q in self.ema.parameters():
            q.requires_grad_(False)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=p.lr,
                                     weight_decay=p.weight_decay)
        self.step = 0
        self.examples = 0
        self.next_panel_index = 0
        self._stop_requested = False
        self._real = self._load_real_val()
        self._val_suite = self._build_val_suite()

    # -- persistence ---------------------------------------------------------

    def checkpoint(self, tag: str = "last") -> None:
        path = self.run_dir / f"ckpt-{tag}.pt"
        tmp = path.with_suffix(".tmp")
        torch.save({
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "opt": self.opt.state_dict(),
            "step": self.step,
            "examples": self.examples,
            "next_panel_index": self.next_panel_index,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if self.device == "cuda" else None,
            "cfg": asdict(self.p),
        }, tmp)
        os.replace(tmp, path)
        if tag == "last":  # rolling history for crash safety
            hist = self.run_dir / f"ckpt-step{self.step}.pt"
            torch.save(torch.load(path, weights_only=False, map_location="cpu"), hist)
            old = sorted(self.run_dir.glob("ckpt-step*.pt"),
                         key=lambda f: int(f.stem.split("step")[1]))
            for f in old[: -self.p.keep_last]:
                f.unlink()

    def resume(self, path: Path) -> None:
        # CPU: RNG states must remain ByteTensors on CPU; state_dict loads
        # copy into device-resident params regardless
        ck = torch.load(path, map_location="cpu", weights_only=False)
        # normalize through JSON: tuples become lists on either path
        same = json.loads(json.dumps(ck["cfg"], default=list)) == \
               json.loads(json.dumps(asdict(self.p), default=list))
        assert same, "config changed since checkpoint — refuse resume"
        self.model.load_state_dict(ck["model"])
        self.ema.load_state_dict(ck["ema"])
        self.opt.load_state_dict(ck["opt"])
        self.step, self.examples = ck["step"], ck["examples"]
        self.next_panel_index = ck["next_panel_index"]
        torch.set_rng_state(ck["torch_rng"])
        if ck["cuda_rng"] is not None and self.device == "cuda":
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        print(f"resumed from {path.name}: step {self.step}, panel {self.next_panel_index}")

    # -- evaluation ----------------------------------------------------------

    def _build_val_suite(self):
        """Frozen synthetic panels at known gammas, regenerated deterministically."""
        suite = []
        for g in self.p.val_gammas:
            for k in range(self.p.val_panels_per_gamma):
                shocks = make_shocks(self.p.D, self.p.N, seed=9000 + k)
                panel, _ = make_panel(g, shocks, T=60)
                r = torch.from_numpy(panel.returns).to(self.device)
                X, y = _windows_and_targets(r, self.p.T, self.p.H)
                suite.append((g, X[:: self.p.H], y[:: self.p.H]))  # non-overlapping
        return suite

    def _load_real_val(self):
        from .config import PANEL
        from .data.dataset import load_panel

        if not PANEL.exists():
            return None
        panel = load_panel()
        anchors = valid_anchors(panel.D, self.p.T, self.p.H)
        cfg = Cfg()
        _, va, _ = purged_split(anchors, cfg.split.train_frac, cfg.split.val_frac, self.p.H)
        r = torch.from_numpy(panel.returns).to(self.device)
        X, y = _windows_and_targets(r, self.p.T, self.p.H)
        offset = self.p.T - 1
        idx = torch.from_numpy(va - offset).to(self.device)
        return X[idx][:: self.p.H], y[idx][:: self.p.H]

    @torch.no_grad()
    def evaluate(self) -> dict:
        out = {}
        per_gamma: dict = {}
        for g, X, y in self._val_suite:
            ics = []
            for i in range(0, len(X), self.p.batch):
                _, aux = self.ema(X[i : i + self.p.batch])
                ics.append(float(rank_ic(aux["logits"], y[i : i + self.p.batch])))
            per_gamma.setdefault(g, []).append(float(np.mean(ics)))
        for g, v in per_gamma.items():
            out[f"synth_ic_g{g}"] = float(np.mean(v))
        if self._real is not None:
            X, y = self._real
            ics = [float(rank_ic(self.ema(X[i : i + self.p.batch])[1]["logits"],
                                 y[i : i + self.p.batch]))
                   for i in range(0, len(X), self.p.batch)]
            out["real_val_rank_ic"] = float(np.mean(ics))
        return out

    # -- training ------------------------------------------------------------

    def run(self, hours: float) -> None:
        p = self.p
        stream = PanelStream(p, self.next_panel_index)
        log_f = (self.run_dir / "log.jsonl").open("a")
        deadline = time.time() + hours * 3600
        last_ckpt = last_eval = time.time()

        def on_sigterm(*_):
            self._stop_requested = True

        signal.signal(signal.SIGTERM, on_sigterm)
        signal.signal(signal.SIGINT, on_sigterm)
        print(f"pretraining on {self.device}: {sum(q.numel() for q in self.model.parameters())} "
              f"params, budget {hours}h, starting at step {self.step}")

        while time.time() < deadline and not self._stop_requested:
            self.next_panel_index_pending, r_np = stream.get()
            r = torch.from_numpy(r_np).to(self.device)
            perm = torch.randperm(r.shape[1], device=self.device)  # shuffle stock order
            X, y = _windows_and_targets(r[:, perm], p.T, p.H)
            order = torch.randperm(len(X), device=self.device)
            for i in range(0, len(order) - p.batch + 1, p.batch):
                idx = order[i : i + p.batch]
                with torch.autocast(self.device, dtype=torch.bfloat16,
                                    enabled=self.device == "cuda"):
                    _, aux = self.model(X[idx])
                    loss = -mean_ic(aux["logits"].float(), y[idx].float())
                assert torch.isfinite(loss), f"non-finite loss at step {self.step}"
                self.opt.zero_grad()
                loss.backward()
                # warmup
                lr = p.lr * min(1.0, (self.step + 1) / p.warmup_steps)
                for grp in self.opt.param_groups:
                    grp["lr"] = lr
                self.opt.step()
                with torch.no_grad():
                    for e, m in zip(self.ema.parameters(), self.model.parameters()):
                        e.lerp_(m, 1 - p.ema_decay)
                    for e, m in zip(self.ema.buffers(), self.model.buffers()):
                        e.copy_(m)
                self.step += 1
                self.examples += p.batch
                if self.step % 100 == 0:
                    log_f.write(json.dumps({"step": self.step, "examples": self.examples,
                                            "loss": float(loss)}) + "\n")
                    log_f.flush()
                if self._stop_requested:
                    break
            self.next_panel_index = stream.next_index

            now = time.time()
            if now - last_eval >= p.eval_every_s or self._stop_requested or now >= deadline:
                metrics = self.evaluate()
                line = {"step": self.step, "examples": self.examples,
                        "hours": round((hours * 3600 - (deadline - now)) / 3600, 2), **metrics}
                log_f.write(json.dumps({"eval": line}) + "\n")
                log_f.flush()
                with (self.run_dir / "CAMPAIGN.md").open("a") as f:
                    f.write(f"| {time.strftime('%m-%d %H:%M')} | {self.step} "
                            f"| {self.examples:,} | "
                            + " | ".join(f"{metrics.get(f'synth_ic_g{g}', float('nan')):+.4f}"
                                         for g in p.val_gammas)
                            + f" | {metrics.get('real_val_rank_ic', float('nan')):+.4f} |\n")
                print(f"step {self.step}: " + " ".join(f"{k}={v:+.4f}" for k, v in metrics.items()))
                self.checkpoint("best" if self._is_best(metrics) else "last")
                last_eval = now
            if now - last_ckpt >= p.ckpt_every_s:
                self.checkpoint()
                last_ckpt = now

        stream.stop()
        self.checkpoint()
        log_f.close()
        print(f"stopped at step {self.step} ({'SIGTERM' if self._stop_requested else 'budget'})")

    def _is_best(self, metrics: dict) -> bool:
        key = "synth_ic_g10"  # the detection-floor metric the campaign chases
        best_file = self.run_dir / "best_metric.json"
        prev = json.loads(best_file.read_text())[key] if best_file.exists() else -9e9
        if metrics.get(key, -9e9) > prev:
            best_file.write_text(json.dumps(metrics))
            self.checkpoint("best")
            return True
        return False


def run_stage_c(hours: float, resume: str | None) -> None:
    if resume == "auto":
        runs = sorted(RUNS.glob("campaign-c-*"))
        resume_dir = runs[-1] if runs else None
    else:
        resume_dir = Path(resume) if resume else None

    if resume_dir and (resume_dir / "ckpt-last.pt").exists():
        run_dir = resume_dir
        p = PretrainCfg(**json.loads((run_dir / "pretrain_cfg.json").read_text()))
        trainer = Pretrainer(p, run_dir)
        trainer.resume(run_dir / "ckpt-last.pt")
    else:
        run_dir = RUNS / f"campaign-c-{time.strftime('%Y%m%d-%H%M%S')}"
        run_dir.mkdir(parents=True)
        p = PretrainCfg()
        (run_dir / "pretrain_cfg.json").write_text(json.dumps(asdict(p)))
        (run_dir / "CAMPAIGN.md").write_text(
            "# Stage C pretraining\n\n| time | step | examples | "
            + " | ".join(f"IC@g{g}" for g in p.val_gammas) + " | real val rank-IC |\n"
            + "|---|---|---|" + "---|" * len(p.val_gammas) + "---|\n")
        trainer = Pretrainer(p, run_dir)
    trainer.run(hours)


# ---------------------------------------------------------------------------
# Stage D: walk-forward real-data campaign.
# 13 yearly folds (2014->2026), expanding train window, purged boundaries;
# per fold x seed x protocol {scratch, full, head} fine-tune the Stage C
# architecture on real data, early-stopped on fold-val rank-IC; ensemble =
# mean logits across seeds. All tensors GPU-resident per fold — no DataLoader.
# ---------------------------------------------------------------------------

def walkforward_folds(panel, T: int, H: int, start_year: int = 2014):
    anchors = valid_anchors(panel.D, T, H)
    adates = panel.dates[anchors]
    years = adates.astype("datetime64[Y]").astype(int) + 1970
    folds = []
    for y in range(start_year, int(years.max()) + 1):
        test = anchors[years == y]
        if not len(test):
            continue
        trainval = anchors[adates < np.datetime64(f"{y}-01-01")]
        trainval = trainval[trainval + H < test.min()]          # purge vs test
        cut = int(len(trainval) * 0.85)
        tr, va = trainval[:cut], trainval[cut:]
        tr = tr[tr + H < va.min()]                              # purge vs val
        assert tr.max() + H < va.min() and va.max() + H < test.min()
        folds.append((y, tr, va, test))
    return folds


def _rank_ic_t(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float(rank_ic(logits, y))


def _fold_tensors(panel, device, T, H):
    r = torch.from_numpy(panel.returns).to(device)
    X, y = _windows_and_targets(r, T, H)
    return X, y, T - 1  # offset: panel row -> X row


def run_stage_d(pretrained_ckpt: Path, seeds: int = 5, out_name: str | None = None) -> Path:
    from .config import RUNS
    from .data.dataset import load_panel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    panel = load_panel()
    p = PretrainCfg(N=panel.N)
    T, H = p.T, p.H
    ck = torch.load(pretrained_ckpt, map_location="cpu", weights_only=False)
    pre_state = ck["ema"]
    out = RUNS / (out_name or f"campaign-d-{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True)
    results_f = (out / "results.jsonl").open("w")
    X_all, y_all, off = _fold_tensors(panel, device, T, H)
    folds = walkforward_folds(panel, T, H)
    print(f"stage D: {len(folds)} folds x {seeds} seeds x 3 protocols on {device}")

    for year, tr, va, te in folds:
        Xtr, ytr = X_all[tr - off], y_all[tr - off]
        Xva, yva = X_all[va - off], y_all[va - off]
        Xte, yte = X_all[te - off], y_all[te - off]
        ens: dict[str, list] = {"scratch": [], "full": [], "head": []}
        for seed in range(seeds):
            for proto in ("scratch", "full", "head"):
                torch.manual_seed(4000 + seed)
                model = REGISTRY["attention"](panel.N, T, _model_cfg_d(p)).to(device)
                if proto != "scratch":
                    model.load_state_dict(pre_state, strict=False)
                if proto == "head":
                    for name, q in model.named_parameters():
                        q.requires_grad_("head" in name or "ln" in name
                                         or "id_embed" in name)
                lr = 1e-3 if proto in ("scratch", "head") else 1e-4
                opt = torch.optim.AdamW(
                    [q for q in model.parameters() if q.requires_grad], lr=lr)
                best_ic, best_state, best_ep = -9e9, None, 0
                for epoch in range(40):
                    model.train()
                    order = torch.randperm(len(Xtr), device=device)
                    for i in range(0, len(order) - 511, 512):
                        idx = order[i : i + 512]
                        with torch.autocast(device, dtype=torch.bfloat16,
                                            enabled=device == "cuda"):
                            _, aux = model(Xtr[idx])
                            loss = -mean_ic(aux["logits"].float(), ytr[idx].float())
                        assert torch.isfinite(loss)
                        opt.zero_grad(); loss.backward(); opt.step()
                    model.eval()
                    with torch.no_grad():
                        vic = _rank_ic_t(model(Xva)[1]["logits"], yva)
                    if vic > best_ic:
                        best_ic, best_ep = vic, epoch
                        best_state = {k: v.detach().clone()
                                      for k, v in model.state_dict().items()}
                    elif epoch - best_ep >= 8:
                        break
                model.load_state_dict(best_state)
                model.eval()
                with torch.no_grad():
                    logits_te = model(Xte)[1]["logits"]
                ens[proto].append(logits_te)
                row = {"year": year, "seed": seed, "proto": proto,
                       "val_ic": round(best_ic, 5), "best_epoch": best_ep,
                       "test_rank_ic": round(_rank_ic_t(logits_te[::H], yte[::H]), 5)}
                results_f.write(json.dumps(row) + "\n"); results_f.flush()
        for proto, ls in ens.items():
            mean_logits = torch.stack(ls).mean(0)
            row = {"year": year, "seed": "ensemble", "proto": proto,
                   "test_rank_ic": round(_rank_ic_t(mean_logits[::H], yte[::H]), 5)}
            results_f.write(json.dumps(row) + "\n"); results_f.flush()
            print(f"{year} {proto:8} ensemble rank-IC {row['test_rank_ic']:+.4f}")
    results_f.close()
    _stage_d_report(out)
    return out


def _model_cfg_d(p: PretrainCfg) -> ModelCfg:
    return ModelCfg(d_model=p.d_model, heads=p.heads, enc_hidden=p.enc_hidden,
                    n_blocks=p.n_blocks, use_id_embed=True, head_init_scale=1e-2)


def _stage_d_report(out: Path) -> None:
    rows = [json.loads(l) for l in (out / "results.jsonl").open()]
    years = sorted({r["year"] for r in rows})
    lines = ["# Stage D: walk-forward real-data campaign", "",
             "Ensemble (mean logits across seeds) test rank-IC per year:", "",
             "| year | scratch | full | head |", "|---|---|---|---|"]
    agg = {p: [] for p in ("scratch", "full", "head")}
    for y in years:
        cells = []
        for proto in ("scratch", "full", "head"):
            v = next(r["test_rank_ic"] for r in rows
                     if r["year"] == y and r["proto"] == proto and r["seed"] == "ensemble")
            agg[proto].append(v)
            cells.append(f"{v:+.4f}")
        lines.append(f"| {y} | " + " | ".join(cells) + " |")
    lines += ["", "| | mean | 2·SE over years |", "|---|---|---|"]
    for proto, vs in agg.items():
        m = float(np.mean(vs))
        se = float(np.std(vs, ddof=1) / np.sqrt(len(vs)))
        lines.append(f"| {proto} | {m:+.4f} | ±{2 * se:.4f} |")
    lines += ["", "Per-anchor rank-IC SE ≈ 0.05/√(~50 weeks/yr) — treat single-year",
              "cells as noise; the mean row over 13 years is the result. Survivorship",
              "bias inflates all protocols equally; differences between columns are",
              "the sim2real claim."]
    (out / "D_REPORT.md").write_text("\n".join(lines) + "\n")
