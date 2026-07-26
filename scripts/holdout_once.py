"""Round 5 Phase 3: THE ONE SHOT. Sole permitted reader of holdout returns."""
import json, sys
import numpy as np, torch
sys.path.insert(0, "src")
from crossfolio.campaign import _windows_and_targets
from crossfolio.config import Cfg, ModelCfg
from crossfolio.data.dataset import load_panel
from crossfolio.data.splits import purged_split, valid_anchors
from crossfolio.edge import HOLDOUT_START, dev_panel, net_monthly_excess
from crossfolio.models import REGISTRY
from crossfolio.synth import forward_returns
sys.path.insert(0, "scripts")
from phase2_dev import finetune  # same training loop as development

DEV = "cuda" if torch.cuda.is_available() else "cpu"
cfg = Cfg(); T, H = cfg.data.T, cfg.data.H
full = load_panel()
r = torch.from_numpy(full.returns).to(DEV)
X_all, y_all = _windows_and_targets(r, T, H)
off = T - 1
anchors = valid_anchors(full.D, T, H)
adates = full.dates[anchors]
hold = anchors[adates >= HOLDOUT_START]
devA = anchors[adates < HOLDOUT_START]
devA = devA[devA + H < hold.min()]                       # purge
cut = int(len(devA) * 0.85)
tr, va = devA[:cut][::2], devA[cut:]
tr = tr[tr + H < va.min()]

logits = []
for s in range(5):
    torch.manual_seed(7000 + s)
    m = REGISTRY["p7_encoder"](full.N, T, ModelCfg(head_init_scale=1e-2)).to(DEV)
    m = finetune(m, X_all[tr-off], y_all[tr-off], X_all[va-off], y_all[va-off])
    with torch.no_grad():
        logits.append(m(X_all[hold-off])[1]["logits"].float().cpu().numpy())
L = np.mean(logits, 0)
L = (L - L.mean(1, keepdims=True)) / (L.std(1, keepdims=True) + 1e-9)
w = np.exp(L - L.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
h5 = np.arange(len(hold))[::H]
y_np, _ = forward_returns(full, hold[::H], H)
res = {b: net_monthly_excess(w[h5], y_np, y_np.mean(1), full.dates[hold[::H]], b, 0.3)
       for b in (5.0, 10.0, 20.0)}
m10 = np.array(res[10.0]["monthly"])
t = m10.mean() / (m10.std(ddof=1) / np.sqrt(len(m10)))
out = {"holdout_months": len(m10),
       **{f"net_sharpe@{int(b)}bps": round(res[b]["net_sharpe_ann"], 3) for b in res},
       "t_stat_monthly@10bps": round(float(t), 2),
       "turnover": round(res[10.0]["mean_weekly_turnover"], 3)}
print(json.dumps(out, indent=1))
