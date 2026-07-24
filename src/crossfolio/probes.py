"""Interpretability probe suite: what did the model actually learn?

Internal-evidence adjudication (see plan): P1 attention vs sector/correlation
structure + RMT modes, P2 alignment dynamics, P3 linear feature probes (incl.
the panel-level in-context latent), P4 causal patching (P6-vs-P7), P5 regime
gating. Every probe runs on a four-model matrix; `random` is the null control
every result must beat. Forward passes + linear algebra only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .campaign import PretrainCfg, _model_cfg, _windows_and_targets
from .config import RUNS, Cfg, ModelCfg
from .data.dataset import load_panel
from .data.splits import purged_split, valid_anchors
from .losses import mean_ic
from .models import REGISTRY
from .power import _ranks
from .synth import cross_sectional_ic, make_cocktail_panel, make_panel, make_shocks
from .universe import SECTORS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -- model matrix -----------------------------------------------------------

def load_matrix(panel) -> dict[str, torch.nn.Module]:
    p = PretrainCfg(N=panel.N)
    big = _model_cfg(p)  # use_id_embed=False: N-agnostic, matches pretraining
    out = {}

    m = REGISTRY["attention"](panel.N, p.T, big).to(DEVICE).eval()
    ck = torch.load(sorted(RUNS.glob("campaign-c-*/ckpt-best.pt"))[-1],
                    map_location="cpu", weights_only=False)
    m.load_state_dict(ck["ema"])
    out["pretrained"] = m

    out["finetuned"] = _finetuned(panel, ck["ema"], p)

    scratch_ck = sorted(RUNS.glob("power-*[0-9]/g40-s0-attention-ic/best.pt"))
    if scratch_ck:
        sm = REGISTRY["attention"](panel.N, p.T, ModelCfg(head_init_scale=1e-2)).to(DEVICE).eval()
        sm.load_state_dict(torch.load(scratch_ck[-1], map_location="cpu",
                                      weights_only=True)["state_dict"])
        out["scratch"] = sm

    torch.manual_seed(31337)
    out["random"] = REGISTRY["attention"](panel.N, p.T, big).to(DEVICE).eval()
    return out


def _finetuned(panel, pre_state, p: PretrainCfg) -> torch.nn.Module:
    """Full-protocol fine-tune on the standard purged split; cached to disk."""
    cache = RUNS / "probes-shared-finetuned.pt"
    cfg = Cfg()
    model = REGISTRY["attention"](panel.N, p.T, _model_cfg(p)).to(DEVICE)
    if cache.exists():
        model.load_state_dict(torch.load(cache, map_location="cpu", weights_only=True))
        return model.eval()
    model.load_state_dict(pre_state)
    anchors = valid_anchors(panel.D, p.T, p.H)
    tr, va, _ = purged_split(anchors, cfg.split.train_frac, cfg.split.val_frac, p.H)
    r = torch.from_numpy(panel.returns).to(DEVICE)
    X, y = _windows_and_targets(r, p.T, p.H)
    off = p.T - 1
    Xtr, ytr = X[tr - off], y[tr - off]
    Xva, yva = X[va - off], y[va - off]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    best, best_state, best_ep = -9e9, None, 0
    for epoch in range(40):
        model.train()
        order = torch.randperm(len(Xtr), device=DEVICE)
        for i in range(0, len(order) - 511, 512):
            idx = order[i : i + 512]
            with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                _, aux = model(Xtr[idx])
                loss = -mean_ic(aux["logits"].float(), ytr[idx].float())
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vic = float(mean_ic(model(Xva)[1]["logits"], yva))
        if vic > best:
            best, best_ep = vic, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_ep >= 8:
            break
    model.load_state_dict(best_state)
    torch.save(best_state, cache)
    return model.eval()


# -- P1: attention vs structure ---------------------------------------------

@torch.no_grad()
def mean_attention(model, X, batch=64) -> np.ndarray:
    """(heads, N, N) attention averaged over anchors."""
    acc, n = None, 0
    for i in range(0, len(X), batch):
        _, aux = model(X[i : i + batch])
        a = aux["attn"].sum(0)
        acc = a if acc is None else acc + a
        n += aux["attn"].shape[0]
    return (acc / n).float().cpu().numpy()


def _offdiag(M: np.ndarray) -> np.ndarray:
    return M[~np.eye(len(M), dtype=bool)]


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC without sklearn."""
    order = scores.argsort().argsort().astype(np.float64) + 1  # 1-based ranks
    pos = labels.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((order[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def sector_comembership(tickers) -> np.ndarray:
    sec = np.array([SECTORS[t] for t in tickers])
    return sec[:, None] == sec[None, :]


def p1_metrics(A_heads: np.ndarray, S: np.ndarray, C: np.ndarray) -> list[dict]:
    out = []
    off = ~np.eye(S.shape[0], dtype=bool)
    for h, A in enumerate(A_heads):
        As = (A + A.T) / 2
        within = A[S & off].mean()
        cross = A[~S & off].mean()
        out.append({
            "head": h,
            "within_cross_ratio": float(within / (cross + 1e-12)),
            "sector_auc": auc(As[off], S[off]),
            "corr_spearman": spearman(As[off], C[off]),
            **rmt_shares(As, C),
        })
    return out


def rmt_shares(A_sym: np.ndarray, C: np.ndarray, n_days: int = 250) -> dict:
    """Energy of (centered) attention in the correlation eigenbasis:
    market mode / other above-Marchenko-Pastur modes / noise band."""
    N = len(C)
    evals, V = np.linalg.eigh(C)
    evals, V = evals[::-1], V[:, ::-1]
    lam_plus = (1 + np.sqrt(N / n_days)) ** 2
    k_sig = int((evals > lam_plus).sum())
    Ac = A_sym - A_sym.mean()
    coef = V.T @ Ac @ V
    E = coef**2
    total = E.sum() + 1e-24
    market = E[0, :].sum() + E[:, 0].sum() - E[0, 0]
    sector = E[:k_sig, :k_sig].sum() - E[0, :k_sig].sum() - E[:k_sig, 0].sum() + E[0, 0]
    return {"rmt_market_share": float(market / total),
            "rmt_sector_share": float(max(sector, 0) / total),
            "rmt_noise_share": float(1 - (market + max(sector, 0)) / total),
            "rmt_n_signal_modes": k_sig}


def trailing_corr(panel, end_row: int, days: int = 250) -> np.ndarray:
    r = panel.returns[end_row - days : end_row]
    return np.corrcoef(r.T)


# -- P2: alignment dynamics -------------------------------------------------

@torch.no_grad()
def p2_alignment_series(model, panel, X, test_anchors, off, H) -> tuple[list, np.ndarray]:
    dates, series = [], []
    for d in test_anchors[::H]:
        _, aux = model(X[d - off].unsqueeze(0))
        A = aux["attn"].squeeze(0).float().cpu().numpy()
        C = trailing_corr(panel, d + 1)
        offm = ~np.eye(len(C), dtype=bool)
        series.append([spearman(((a + a.T) / 2)[offm], C[offm]) for a in A])
        dates.append(panel.dates[d])
    return dates, np.array(series)  # (t, heads)


# -- P3: ridge feature probes -----------------------------------------------

def ridge_r2(Z_tr, y_tr, Z_te, y_te, lam=1e-2) -> float:
    Z_tr = np.concatenate([Z_tr, np.ones((len(Z_tr), 1))], 1)
    Z_te = np.concatenate([Z_te, np.ones((len(Z_te), 1))], 1)
    G = Z_tr.T @ Z_tr + lam * np.eye(Z_tr.shape[1])
    W = np.linalg.solve(G, Z_tr.T @ y_tr)
    pred = Z_te @ W
    ss_res = ((y_te - pred) ** 2).sum()
    ss_tot = ((y_te - y_te.mean(0)) ** 2).sum() + 1e-24
    return float(1 - ss_res / ss_tot)


@torch.no_grad()
def token_reps(model, X, batch=64) -> dict[str, np.ndarray]:
    """Representations at depths: post-encoder, after each block. (A, N, d)."""
    grabs: dict[str, list] = {}
    hooks = []

    def grab(name):
        def fn(_m, _i, o):
            out = o[0] if isinstance(o, tuple) else o
            grabs.setdefault(name, []).append(out.detach().float().cpu())
        return fn

    hooks.append(model.encoder.register_forward_hook(grab("encoder")))
    for bi, block in enumerate(model.blocks):
        hooks.append(block.register_forward_hook(grab(f"block{bi}")))
    for i in range(0, len(X), batch):
        model(X[i : i + batch])
    for h in hooks:
        h.remove()
    return {k: torch.cat(v).numpy() for k, v in grabs.items()}


def cross_z(v: np.ndarray) -> np.ndarray:
    return (v - v.mean(-1, keepdims=True)) / (v.std(-1, keepdims=True) + 1e-12)


def real_features(panel, anchors) -> dict[str, np.ndarray]:
    """Per-(anchor, stock) targets, cross-sectionally z-scored. (A, N) each."""
    cum = np.cumsum(panel.returns, 0)
    def mom(L):
        return cross_z(np.stack([cum[d] - cum[d - L] for d in anchors]))
    r = panel.returns
    vol = cross_z(np.stack([r[d - 20 : d].std(0) for d in anchors]))
    spy = panel.spy
    beta = cross_z(np.stack([
        (r[d - 60 : d] * spy[d - 60 : d, None]).mean(0) / (spy[d - 60 : d].var() + 1e-12)
        for d in anchors]))
    return {"mom20": mom(20), "mom60": mom(60), "mom120": mom(120),
            "rev5": mom(5), "vol20": vol, "beta60": beta}


def p3_real(model, panel, X, anchors, off, max_anchors=300) -> dict:
    sub = anchors[:: max(1, len(anchors) // max_anchors)]
    reps = token_reps(model, X[sub - off])
    feats = real_features(panel, sub)
    half = len(sub) // 2
    out = {}
    for depth, Z in reps.items():
        Zf = Z.reshape(Z.shape[0], Z.shape[1], -1)
        for fname, F in feats.items():
            r2 = ridge_r2(Zf[:half].reshape(-1, Zf.shape[-1]), F[:half].reshape(-1, 1),
                          Zf[half:].reshape(-1, Zf.shape[-1]), F[half:].reshape(-1, 1))
            out[f"{depth}/{fname}"] = round(r2, 4)
    return out


def p3_latent(model, N, n_panels=120, D=800, anchors_per=4, seed0=50_000) -> float:
    """Decode the panel-level latent gamma_mom from mean-pooled tokens.
    The in-context variable pretraining was supposed to create."""
    Zs, ys = [], []
    for i in range(n_panels):
        rng = np.random.default_rng([9898, i])
        g_mom = float(rng.uniform(0, 40))
        panel, _ = make_panel(g_mom, make_shocks(D, N, seed=seed0 + i), T=60)
        r = torch.from_numpy(panel.returns).to(DEVICE)
        X, _ = _windows_and_targets(r, 120, 5)
        idx = torch.linspace(len(X) // 2, len(X) - 1, anchors_per).long()
        reps = token_reps(model, X[idx])
        deepest = sorted(reps)[-1] if any(k.startswith("block") for k in reps) else "encoder"
        Zs.append(reps[deepest].mean(axis=(0, 1)))  # pool anchors+stocks -> (d,)
        ys.append(g_mom)
    Z, y = np.array(Zs), np.array(ys)[:, None]
    half = len(Z) // 2
    return round(ridge_r2(Z[:half], y[:half], Z[half:], y[half:]), 4)


# -- P4: causal patching ----------------------------------------------------

@torch.no_grad()
def _suite_ic(model, suite) -> dict[int, float]:
    out: dict[int, list] = {}
    for g, X, y in suite:
        ics = []
        for i in range(0, len(X), 128):
            _, aux = model(X[i : i + 128])
            ics.append(cross_sectional_ic(
                _ranks(aux["logits"].float().cpu().numpy()),
                _ranks(y[i : i + 128].float().cpu().numpy())).mean())
        out.setdefault(g, []).append(float(np.mean(ics)))
    return {g: float(np.mean(v)) for g, v in out.items()}


def build_suite(N, gammas=(0, 20, 40), panels=4, D=1500):
    suite = []
    for g in gammas:
        for k in range(panels):
            panel, _ = make_panel(g, make_shocks(D, N, seed=9000 + k), T=60)
            r = torch.from_numpy(panel.returns).to(DEVICE)
            X, y = _windows_and_targets(r, 120, 5)
            suite.append((g, X[::5], y[::5]))
    return suite


def p4_patching(model, suite) -> dict:
    base = _suite_ic(model, suite)
    mhsas = [b.mhsa for b in model.blocks]
    for m in mhsas:
        m.patch_uniform = True
    uniform = _suite_ic(model, suite)
    for m in mhsas:
        m.patch_uniform = False
    heads = model.blocks[0].mhsa.heads
    per_head = {}
    for h in range(heads):
        for m in mhsas:
            m.ablate_heads = [h]
        per_head[h] = _suite_ic(model, suite)
        for m in mhsas:
            m.ablate_heads = ()
    return {"base": base, "uniform_attn": uniform, "head_ablation": per_head}


# -- P5: regime gating (proxy: realized window vol tertiles) ----------------

@torch.no_grad()
def p5_regime(model, N, n_panels=24, D=1500) -> dict:
    """On cocktail panels: corr(logits, momentum z) grouped by window-vol
    tertile. Regime gating emerged iff alignment drops in high-vol windows
    (the cocktail sets momentum gamma -> 0 when stressed)."""
    per_bucket: dict[str, list] = {"calm": [], "mid": [], "stressed": []}
    for i in range(n_panels):
        panel = make_cocktail_panel(60_000 + i, 777, N, D)
        r = torch.from_numpy(panel.returns).to(DEVICE)
        X, _ = _windows_and_targets(r, 120, 5)
        X5 = X[::5]
        cum = np.cumsum(panel.returns, 0)
        anchors = np.arange(119, D - 5)[::5]
        momz = cross_z(np.stack([cum[d] - cum[d - 60] for d in anchors]))
        wvol = X5.std(dim=(1, 2)).cpu().numpy()
        _, aux = model(X5)
        logits = aux["logits"].float().cpu().numpy()
        align = cross_sectional_ic(logits, momz)
        t1, t2 = np.quantile(wvol, [1 / 3, 2 / 3])
        per_bucket["calm"].extend(align[wvol < t1])
        per_bucket["mid"].extend(align[(wvol >= t1) & (wvol < t2)])
        per_bucket["stressed"].extend(align[wvol >= t2])
    return {k: round(float(np.mean(v)), 4) for k, v in per_bucket.items()}


# -- orchestrator -----------------------------------------------------------

def run_probes(out_dir: Path | None = None) -> Path:
    panel = load_panel()
    cfg = Cfg()
    T, H = cfg.data.T, cfg.data.H
    anchors = valid_anchors(panel.D, T, H)
    _, _, test = purged_split(anchors, cfg.split.train_frac, cfg.split.val_frac, H)
    r = torch.from_numpy(panel.returns).to(DEVICE)
    X, _ = _windows_and_targets(r, T, H)
    off = T - 1
    S = sector_comembership(panel.tickers)
    C = trailing_corr(panel, int(test[-1]) + 1)

    out = out_dir or RUNS / f"probes-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    models = load_matrix(panel)
    suite = build_suite(panel.N)
    results: dict = {}

    for name, model in models.items():
        print(f"== {name}")
        A = mean_attention(model, X[test - off][::H])
        res = {"p1": p1_metrics(A, S, C)}
        np.save(out / f"attn_{name}.npy", A)
        dates, series = p2_alignment_series(model, panel, X, test, off, H)
        res["p2_mean_alignment_per_head"] = [round(float(v), 4) for v in series.mean(0)]
        np.save(out / f"p2_series_{name}.npy", series)
        res["p3_real"] = p3_real(model, panel, X, test, off)
        res["p3_latent_gamma_r2"] = p3_latent(model, panel.N)
        res["p4"] = p4_patching(model, suite)
        res["p5"] = p5_regime(model, panel.N)
        results[name] = res
        (out / "results.json").write_text(json.dumps(results, indent=1))
        print(json.dumps({k: v for k, v in res.items() if k != "p3_real"}, indent=1)[:600])

    _write_report(results, out)
    return out


def _write_report(results: dict, out: Path) -> None:
    L = ["# Probe suite results", "",
         "Four-model matrix; `random` is the null every claim must beat.", ""]
    L += ["## P1 — attention vs sector/correlation structure", "",
          "| model | head | within/cross | sector AUC | corr ρ | RMT mkt | RMT sector | RMT noise |",
          "|---|---|---|---|---|---|---|---|"]
    for m, res in results.items():
        for row in res["p1"]:
            L.append(f"| {m} | {row['head']} | {row['within_cross_ratio']:.2f} "
                     f"| {row['sector_auc']:.3f} | {row['corr_spearman']:+.3f} "
                     f"| {row['rmt_market_share']:.2f} | {row['rmt_sector_share']:.2f} "
                     f"| {row['rmt_noise_share']:.2f} |")
    L += ["", "## P2 — mean attention-correlation alignment per head", ""]
    for m, res in results.items():
        L.append(f"- **{m}**: {res['p2_mean_alignment_per_head']}")
    L += ["", "## P3 — in-context latent (gamma_mom readout R², pooled tokens)", ""]
    for m, res in results.items():
        L.append(f"- **{m}**: R² = {res['p3_latent_gamma_r2']}")
    L += ["", "(full per-depth feature R² tables in results.json)", "",
          "## P4 — causal patching (synthetic detection IC)", "",
          "| model | condition | IC@γ0 | IC@γ20 | IC@γ40 |", "|---|---|---|---|---|"]
    for m, res in results.items():
        for cond in ("base", "uniform_attn"):
            d = res["p4"][cond]
            L.append(f"| {m} | {cond} | {d.get(0, float('nan')):+.4f} "
                     f"| {d.get(20, float('nan')):+.4f} | {d.get(40, float('nan')):+.4f} |")
    L += ["", "Per-head ablations in results.json.", "",
          "## P5 — regime gating (logit-momentum alignment by window-vol tertile)", ""]
    for m, res in results.items():
        L.append(f"- **{m}**: {res['p5']}")
    (out / "PROBES.md").write_text("\n".join(L) + "\n")
