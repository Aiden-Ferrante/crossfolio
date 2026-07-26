"""Round 6 Phase 3 (A4): SPD-manifold regime feature -> augmented-input models.
Regime r_t = trailing-z-scored log-Euclidean distance between consecutive
RMT-denoised rolling 120d correlation matrices. Registered deviation: the
feature enters BOTH models as an appended input column (T+1); the separate
gate-modulator variant is dropped for trial economy."""
import json, sys
import numpy as np, torch
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from crossfolio.campaign import _windows_and_targets, walkforward_folds
from crossfolio.config import ModelCfg
from crossfolio.edge import dev_panel, log_trial, net_monthly_excess
from crossfolio.models import REGISTRY
from crossfolio.synth import forward_returns
from phase_r6 import finetune

DEV = "cuda" if torch.cuda.is_available() else "cpu"
panel = dev_panel(); H, T = 5, 120
N, D = panel.N, panel.D
ret = panel.returns.astype(np.float64)

# regime series at stride-H grid, causal throughout
grid = np.arange(T - 1, D, H)
logms, dists = {}, np.full(D, np.nan)
prev = None
for d in grid:
    Rw = ret[d - T + 1 : d + 1]
    C = np.corrcoef(Rw.T)
    C = np.nan_to_num(C, nan=0.0); np.fill_diagonal(C, 1.0)
    ev, V = np.linalg.eigh(C)
    lam_plus = (1 + np.sqrt(N / T)) ** 2
    noise = ev < lam_plus
    if noise.any():
        ev[noise] = max(ev[noise].mean(), 1e-4)   # RMT clip -> full rank
    logC = (V * np.log(ev)) @ V.T
    if prev is not None:
        dists[d] = np.linalg.norm(logC - prev)
    prev = logC
# trailing z-score (expanding, min 20 obs) then forward-fill to every day
r_series = np.zeros(D)
vals, idxs = [], []
for d in grid:
    if not np.isnan(dists[d]):
        vals.append(dists[d]); idxs.append(d)
        if len(vals) >= 20:
            hist = np.array(vals[:-1])
            r_series[d] = (dists[d] - hist.mean()) / (hist.std() + 1e-9)
for i in range(1, D):
    if r_series[i] == 0 and i - 1 >= 0:
        r_series[i] = r_series[i - 1]
print(f"regime series: std {r_series[2000:].std():.2f}, max |z| {np.abs(r_series).max():.1f}")

r_t = torch.from_numpy(panel.returns).to(DEV)
X_all, y_all = _windows_and_targets(r_t, T, H); off = T - 1
reg = torch.from_numpy(np.clip(r_series, -4, 4).astype(np.float32)).to(DEV)
anchors_all = torch.arange(T - 1, D - H, device=DEV)
Xa = torch.cat([X_all, reg[anchors_all].view(-1, 1, 1).expand(-1, N, 1)], dim=2)  # (A,N,T+1)

folds = walkforward_folds(panel, T, H, start_year=2018)
for model_name, tag in (("p7_encoder", "p7_regime"), ("gated_attention", "gated_regime")):
    W, Y, DT = [], [], []
    for year, tr, va, te in folds:
        tr = tr[::2]; Ls = []
        for s in range(5):
            torch.manual_seed(7000 + s)
            m = REGISTRY[model_name](N, T + 1, ModelCfg(head_init_scale=1e-2)).to(DEV)
            m = finetune(m, Xa[tr-off], y_all[tr-off], Xa[va-off], y_all[va-off])
            with torch.no_grad():
                Ls.append(m(Xa[te-off])[1]["logits"].float().cpu().numpy())
        L = np.mean(Ls, 0); L = (L - L.mean(1, keepdims=True)) / (L.std(1, keepdims=True) + 1e-9)
        w = np.exp(L - L.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
        y_np, _ = forward_returns(panel, te[::H], H)
        W.append(w[::H][:len(y_np)]); Y.append(y_np); DT.append(panel.dates[te[::H]])
    W, Y, DT = map(np.concatenate, (W, Y, DT))
    res = net_monthly_excess(W, Y, Y.mean(1), DT, 10.0, 0.3)
    entry = {"round": 6, "arm": tag, "net_sharpe@10bps": round(res["net_sharpe_ann"], 3),
             "turnover": round(res["mean_weekly_turnover"], 3), "n_months": res["n_months"]}
    log_trial(entry); print(json.dumps(entry))
print("R6 PHASE 3 DONE")
