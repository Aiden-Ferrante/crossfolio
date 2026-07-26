"""Round 5 Phase 2: candidate slate, dev panel only, ledger-logged, net-graded."""
import dataclasses, json, sys
import numpy as np
import torch

sys.path.insert(0, "src")
from crossfolio.campaign import _windows_and_targets, walkforward_folds
from crossfolio.config import Cfg, DataCfg, LossCfg, ModelCfg, TrainCfg
from crossfolio.edge import (COST_BPS_HEADLINE, COST_BPS_SENSITIVITY,
                             dev_panel, log_trial, net_monthly_excess)
from crossfolio.losses import mean_ic, rank_ic
from crossfolio.models import REGISTRY
from crossfolio.synth import forward_returns

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = 5

CANDIDATES = {
    "p7_base":     dict(model="p7_encoder", T=120, mcfg=ModelCfg(head_init_scale=1e-2)),
    "p7_big":      dict(model="p7_encoder", T=120, mcfg=ModelCfg(d_model=64, enc_hidden=128, head_init_scale=1e-2)),
    "p7_reversal": dict(model="p7_encoder", T=10,  mcfg=ModelCfg(head_init_scale=1e-2)),
    "corr_bias":   dict(model="corr_bias_attention", T=120, mcfg=ModelCfg(head_init_scale=1e-2)),
    "p7_stride1":  dict(model="p7_encoder", T=120, mcfg=ModelCfg(head_init_scale=1e-2), stride=1),
}

def finetune(model, Xtr, ytr, Xva, yva, lr=1e-3, max_ep=40, patience=8):
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=lr)
    best, st, be = -9e9, None, 0
    for ep in range(max_ep):
        model.train()
        order = torch.randperm(len(Xtr), device=DEV)
        for i in range(0, len(order) - 511, 512):
            idx = order[i:i+512]
            with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
                _, aux = model(Xtr[idx])
                loss = -mean_ic(aux["logits"].float(), ytr[idx].float())
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = float(rank_ic(model(Xva)[1]["logits"], yva))
        if v > best: best, be, st = v, ep, {k: t.detach().clone() for k, t in model.state_dict().items()}
        elif ep - be >= patience: break
    model.load_state_dict(st); return model

panel = dev_panel()
cfg = Cfg()
H = cfg.data.H
r = torch.from_numpy(panel.returns).to(DEV)
for name, c in CANDIDATES.items():
    T = c["T"]
    X_all, y_all = _windows_and_targets(r, T, H)
    off = T - 1
    folds = walkforward_folds(panel, T, H, start_year=2018)
    stride = c.get("stride", 2)
    W, Y, YS, DT = [], [], [], []
    for year, tr, va, te in folds:
        tr = tr[::stride]
        logit_seeds = []
        for s in range(SEEDS):
            torch.manual_seed(7000 + s)
            m = REGISTRY[c["model"]](panel.N, T, c["mcfg"]).to(DEV)
            m = finetune(m, X_all[tr-off], y_all[tr-off], X_all[va-off], y_all[va-off])
            with torch.no_grad():
                logit_seeds.append(m(X_all[te-off])[1]["logits"].float().cpu().numpy())
        L = np.mean(logit_seeds, 0)
        # v2 amendment (ledgered): standardize logits per anchor so every
        # candidate expresses a comparable, non-microscopic tilt
        L = (L - L.mean(1, keepdims=True)) / (L.std(1, keepdims=True) + 1e-9)
        w = np.exp(L - L.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
        te5 = np.arange(len(te))[::H]
        y_np, _ = forward_returns(panel, te[::H], H)
        ys_np = y_np.mean(1)  # v2: pre-registered benchmark = universe EW, not SPY
        W.append(w[te5]); Y.append(y_np); YS.append(ys_np); DT.append(panel.dates[te[::H]])
    W, Y, YS, DT = map(np.concatenate, (W, Y, YS, DT))
    for alpha in (1.0, 0.3):
        res = net_monthly_excess(W, Y, YS, DT, COST_BPS_HEADLINE, alpha)
        sens = {f"net@{b}bps": net_monthly_excess(W, Y, YS, DT, b, alpha)["net_sharpe_ann"]
                for b in COST_BPS_SENSITIVITY}
        entry = {"eval": "v2", "candidate": name, "smooth_alpha": alpha, "seeds": SEEDS,
                 "net_sharpe@10bps": round(res["net_sharpe_ann"], 3),
                 **{k: round(v, 3) for k, v in sens.items()},
                 "weekly_turnover": round(res["mean_weekly_turnover"], 3),
                 "n_months": res["n_months"]}
        log_trial(entry)
        print(json.dumps(entry))
print("PHASE2 DONE")
