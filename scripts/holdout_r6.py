"""Round 6 Phase 5: the single holdout shot for p7_regime."""
import json, sys
import numpy as np, torch
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from crossfolio.campaign import _windows_and_targets
from crossfolio.config import ModelCfg
from crossfolio.data.dataset import load_panel
from crossfolio.data.splits import valid_anchors
from crossfolio.edge import HOLDOUT_START, log_trial, net_monthly_excess
from crossfolio.models import REGISTRY
from crossfolio.synth import forward_returns
from phase_r6 import finetune

DEV = "cuda" if torch.cuda.is_available() else "cpu"
panel = load_panel(); H, T = 5, 120
N, D = panel.N, panel.D
ret = panel.returns.astype(np.float64)

grid = np.arange(T - 1, D, H)
dists = np.full(D, np.nan); prev = None
for d in grid:
    C = np.corrcoef(ret[d - T + 1 : d + 1].T)
    C = np.nan_to_num(C, nan=0.0); np.fill_diagonal(C, 1.0)
    ev, V = np.linalg.eigh(C)
    noise = ev < (1 + np.sqrt(N / T)) ** 2
    if noise.any(): ev[noise] = max(ev[noise].mean(), 1e-4)
    logC = (V * np.log(ev)) @ V.T
    if prev is not None: dists[d] = np.linalg.norm(logC - prev)
    prev = logC
r_series = np.zeros(D); vals = []
for d in grid:
    if not np.isnan(dists[d]):
        vals.append(dists[d])
        if len(vals) >= 20:
            h = np.array(vals[:-1]); r_series[d] = (dists[d] - h.mean()) / (h.std() + 1e-9)
for i in range(1, D):
    if r_series[i] == 0: r_series[i] = r_series[i - 1]

r_t = torch.from_numpy(panel.returns).to(DEV)
X_all, y_all = _windows_and_targets(r_t, T, H); off = T - 1
reg = torch.from_numpy(np.clip(r_series, -4, 4).astype(np.float32)).to(DEV)
aidx = torch.arange(T - 1, D - H, device=DEV)
Xa = torch.cat([X_all, reg[aidx].view(-1, 1, 1).expand(-1, N, 1)], dim=2)

anchors = valid_anchors(D, T, H)
adates = panel.dates[anchors]
hold = anchors[adates >= HOLDOUT_START]
devA = anchors[adates < HOLDOUT_START]; devA = devA[devA + H < hold.min()]
cut = int(len(devA) * 0.85); tr, va = devA[:cut][::2], devA[cut:]
tr = tr[tr + H < va.min()]

Ls = []
for s in range(5):
    torch.manual_seed(7000 + s)
    m = REGISTRY["p7_encoder"](N, T + 1, ModelCfg(head_init_scale=1e-2)).to(DEV)
    m = finetune(m, Xa[tr-off], y_all[tr-off], Xa[va-off], y_all[va-off])
    with torch.no_grad(): Ls.append(m(Xa[hold-off])[1]["logits"].float().cpu().numpy())
L = np.mean(Ls, 0); L = (L - L.mean(1, keepdims=True)) / (L.std(1, keepdims=True) + 1e-9)
w = np.exp(L - L.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
y_np, _ = forward_returns(panel, hold[::H], H)
out = {}
for b in (5.0, 10.0, 20.0):
    out[f"net@{int(b)}bps"] = round(net_monthly_excess(
        w[::H][:len(y_np)], y_np, y_np.mean(1), panel.dates[hold[::H]], b, 0.3)["net_sharpe_ann"], 3)
m10 = np.array(net_monthly_excess(w[::H][:len(y_np)], y_np, y_np.mean(1),
                                  panel.dates[hold[::H]], 10.0, 0.3)["monthly"])
out["t_stat"] = round(float(m10.mean() / (m10.std(ddof=1) / np.sqrt(len(m10)))), 2)
out["n_months"] = len(m10)
log_trial({"round": 6, "arm": "HOLDOUT_p7_regime", **out})
print(json.dumps(out))
