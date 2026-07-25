import pytest
import torch

from crossfolio.config import ModelCfg
from crossfolio.models import REGISTRY

B, N, T = 4, 12, 10
CFG = ModelCfg(d_model=16, heads=4, enc_hidden=8)


@pytest.mark.parametrize("name", list(REGISTRY))
def test_allocations_are_valid(name):
    model = REGISTRY[name](N, T, CFG)
    w, _ = model(torch.randn(B, N, T))
    assert w.shape == (B, N)
    assert (w >= 0).all()
    torch.testing.assert_close(w.sum(-1), torch.ones(B))


def test_attention_aux():
    model = REGISTRY["attention"](N, T, CFG)
    _, aux = model(torch.randn(B, N, T))
    attn = aux["attn"]
    assert attn.shape == (B, CFG.heads, N, N)
    torch.testing.assert_close(attn.sum(-1), torch.ones(B, CFG.heads, N))


def test_attention_params_dont_grow_with_N():
    def n_params(model):
        return sum(p.numel() for p in model.parameters())

    small = REGISTRY["attention"](12, T, CFG)
    big = REGISTRY["attention"](120, T, CFG)
    # only the ID embedding grows with N
    assert n_params(big) - n_params(small) == (120 - 12) * CFG.d_model
    # while linear explodes quadratically-ish
    assert n_params(REGISTRY["linear"](120, T, CFG)) > 50 * n_params(REGISTRY["linear"](12, T, CFG))


def test_gradients_flow():
    model = REGISTRY["attention"](N, T, CFG)
    w, _ = model(torch.randn(B, N, T))
    w.sum().backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_gated_equals_p7_at_init():
    cfg = ModelCfg(d_model=16, heads=4, enc_hidden=8, n_blocks=2)
    torch.manual_seed(0)
    p7 = REGISTRY["p7_encoder"](N, T, cfg)
    gated = REGISTRY["gated_attention"](N, T, cfg)
    gated.load_state_dict(p7.state_dict(), strict=False)  # shared backbone
    X = torch.randn(B, N, T)
    with torch.no_grad():
        lp, _ = p7(X)
        lg, aux = gated(X)
    torch.testing.assert_close(lp, lg)                     # gates=0 => exact P7
    assert torch.all(aux["gates"] == 0)


def test_gate_gradients_flow():
    cfg = ModelCfg(d_model=16, heads=4, enc_hidden=8, n_blocks=2)
    model = REGISTRY["gated_attention"](N, T, cfg)
    w, _ = model(torch.randn(B, N, T))
    w.var().backward()
    assert all(torch.isfinite(b.gate.grad) for b in model.blocks)


def test_p7_has_no_cross_stock_interaction():
    cfg = ModelCfg(d_model=16, heads=4, enc_hidden=8)
    model = REGISTRY["p7_encoder"](N, T, cfg).eval()
    X = torch.randn(1, N, T)
    X2 = X.clone(); X2[0, 3] += 5.0                       # perturb stock 3 only
    with torch.no_grad():
        z1, _ = model.logits(X)
        z2, _ = model.logits(X2)
    mask = torch.ones(N, dtype=bool); mask[3] = False
    torch.testing.assert_close(z1[0, mask], z2[0, mask])   # others unchanged
    assert not torch.isclose(z1[0, 3], z2[0, 3])


def test_round4_dials_at_init():
    cfg = ModelCfg(d_model=16, heads=4, enc_hidden=8, n_blocks=2)
    torch.manual_seed(0)
    p7 = REGISTRY["p7_encoder"](N, T, cfg)
    cb = REGISTRY["corr_bias_attention"](N, T, cfg)
    cb.load_state_dict(p7.state_dict(), strict=False)
    X = torch.randn(B, N, T)
    with torch.no_grad():
        lp, _ = p7(X)
        lc, aux = cb(X)
    # gate=0.1 (deadlock fix) => approximate, not exact, P7 equivalence
    assert torch.allclose(lp, lc, atol=0.2)
    assert torch.all(aux["lambdas"] == 0)
    # lambda gradients flow even at lambda=0 (because gate != 0)
    w, _ = cb(X)
    w.var().backward()
    assert all(torch.isfinite(l.grad).all() and l.grad.abs().sum() > 0
               for l in cb.lambdas)
