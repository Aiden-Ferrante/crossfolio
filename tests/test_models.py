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
