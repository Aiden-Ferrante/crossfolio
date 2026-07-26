"""Round 6 Phases 1-2: A1 penalty variants, A2 SPT baselines, A3 EG + optimizer probe."""
import json, sys
import numpy as np, torch
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from crossfolio.campaign import _windows_and_targets, walkforward_folds
from crossfolio.config import ModelCfg
from crossfolio.edge import dev_panel, log_trial, net_monthly_excess
from crossfolio.losses import make_loss, rank_ic
from crossfolio.models import REGISTRY
from crossfolio.synth import forward_returns

DEV = "cuda" if torch.cuda.is_available() else "cpu"
panel = dev_panel(); H, T = 5, 120
r = torch.from_numpy(panel.returns).to(DEV)
X_all, y_all = _windows_and_targets(r, T, H); off = T - 1
folds = walkforward_folds(panel, T, H, start_year=2018)

def finetune(m, Xtr, ytr, Xva, yva, loss_name="ic", opt_name="adam", lr=1e-3):
    loss_fn = make_loss(loss_name, 0.0)
    params = [q for q in m.parameters() if q.requires_grad]
    opt = {"adam": torch.optim.Adam(params, lr=lr),
           "sgd": torch.optim.SGD(params, lr=lr * 10),
           "rmsprop": torch.optim.RMSprop(params, lr=lr)}[opt_name]
    best, st, be = -9e9, None, 0
    for ep in range(40):
        m.train()
        order = torch.randperm(len(Xtr), device=DEV)
        for i in range(0, len(order) - 511, 512):
            idx = order[i:i+512]
            with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
                w, aux = m(Xtr[idx])
                loss = loss_fn(w.float(), ytr[idx].float(), ytr[idx].float().mean(1), logits=aux["logits"].float())
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            v = float(rank_ic(m(Xva)[1]["logits"], yva))
        if v > best: best, be, st = v, ep, {k: t.detach().clone() for k, t in m.state_dict().items()}
        elif ep - be >= 8: break
    m.load_state_dict(st); return m

def run_trained(tag, loss_name="ic", opt_name="adam"):
    W, Y, DT = [], [], []
    for year, tr, va, te in folds:
        tr = tr[::2]
        Ls = []
        for s in range(5):
            torch.manual_seed(7000 + s)
            m = REGISTRY["p7_encoder"](panel.N, T, ModelCfg(head_init_scale=1e-2)).to(DEV)
            m = finetune(m, X_all[tr-off], y_all[tr-off], X_all[va-off], y_all[va-off], loss_name, opt_name)
            with torch.no_grad():
                Ls.append(m(X_all[te-off])[1]["logits"].float().cpu().numpy())
        L = np.mean(Ls, 0); L = (L - L.mean(1, keepdims=True)) / (L.std(1, keepdims=True) + 1e-9)
        w = np.exp(L - L.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
        y_np, _ = forward_returns(panel, te[::H], H)
        W.append(w[::H][:len(y_np)]); Y.append(y_np); DT.append(panel.dates[te[::H]])
    W, Y, DT = map(np.concatenate, (W, Y, DT))
    res = net_monthly_excess(W, Y, Y.mean(1), DT, 10.0, 0.3)
    entry = {"round": 6, "arm": tag, "net_sharpe@10bps": round(res["net_sharpe_ann"], 3),
             "turnover": round(res["mean_weekly_turnover"], 3), "n_months": res["n_months"]}
    log_trial(entry); print(json.dumps(entry))

# analytic arms over the concatenated fold test anchors
te_all = np.concatenate([te[::H] for _, _, _, te in folds])
y_np, _ = forward_returns(panel, te_all, H)
dates = panel.dates[te_all]
cum = np.cumsum(panel.returns.astype(np.float64), 0)

for p_exp in (0.5, -0.5):                       # A2: SPT diversity-weighted
    m_cap = np.exp(cum[te_all])                 # pseudo-cap (equal init caps)
    w = m_cap ** p_exp; w /= w.sum(1, keepdims=True)
    for alpha in (1.0,):
        res = net_monthly_excess(w, y_np, y_np.mean(1), dates, 10.0, alpha)
        entry = {"round": 6, "arm": f"spt_p{p_exp}", "net_sharpe@10bps": round(res["net_sharpe_ann"], 3),
                 "turnover": round(res["mean_weekly_turnover"], 3), "n_months": res["n_months"]}
        log_trial(entry); print(json.dumps(entry))

for eta in (0.05, 0.5):                         # A3a: EG online portfolio (Helmbold 98)
    w = np.full(panel.N, 1.0 / panel.N); Wp = []
    for t in range(len(te_all)):
        Wp.append(w.copy())
        x = 1.0 + y_np[t]
        w = w * np.exp(eta * x / (w @ x)); w /= w.sum()
    res = net_monthly_excess(np.array(Wp), y_np, y_np.mean(1), dates, 10.0, 1.0)
    entry = {"round": 6, "arm": f"eg_eta{eta}", "net_sharpe@10bps": round(res["net_sharpe_ann"], 3),
             "turnover": round(res["mean_weekly_turnover"], 3), "n_months": res["n_months"]}
    log_trial(entry); print(json.dumps(entry))

for tag, ln, on in [("ic_lam0_adam", "ic", "adam"), ("ic_hhi_adam", "ic_hhi", "adam"),
                    ("ic_kl_adam", "ic_kl", "adam"), ("ic_lam0_sgd", "ic", "sgd"),
                    ("ic_lam0_rmsprop", "ic", "rmsprop")]:
    run_trained(tag, ln, on)
print("R6 PHASE 1-2 DONE")
