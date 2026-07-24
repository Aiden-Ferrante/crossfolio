import numpy as np
import torch

from crossfolio.config import ModelCfg
from crossfolio.models import REGISTRY
from crossfolio.probes import auc, p1_metrics, ridge_r2, rmt_shares, spearman

N = 30


def _block_sectors():
    sec = np.repeat(np.arange(3), N // 3)
    S = sec[:, None] == sec[None, :]
    return sec, S


def test_sector_metrics_on_block_attention():
    sec, S = _block_sectors()
    A = np.where(S, 1.0, 0.01)[None]          # perfect block attention
    C = np.where(S, 0.6, 0.05) + np.eye(N) * 0.4
    m = p1_metrics(A, S, C)[0]
    assert m["within_cross_ratio"] > 10
    assert m["sector_auc"] > 0.95
    assert m["corr_spearman"] > 0.8


def test_sector_metrics_on_uniform_attention():
    _, S = _block_sectors()
    A = np.full((1, N, N), 1.0 / N)
    C = np.where(S, 0.6, 0.05) + np.eye(N) * 0.4
    m = p1_metrics(A, S, C)[0]
    assert abs(m["sector_auc"] - 0.5) < 0.02
    assert 0.9 < m["within_cross_ratio"] < 1.1


def test_rmt_shares_recover_planted_factor():
    rng = np.random.default_rng(0)
    v = rng.normal(size=N); v /= np.linalg.norm(v)
    C = 6 * np.outer(v, v) + np.eye(N)         # one dominant mode + noise floor
    A = np.outer(v, v)                          # attention living on that mode
    shares = rmt_shares(A, C)
    assert shares["rmt_market_share"] > 0.9
    assert shares["rmt_noise_share"] < 0.1


def test_ridge_probe_recovers_linear_feature():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(2000, 16))
    w = rng.normal(size=(16, 1))
    y = Z @ w + 0.1 * rng.normal(size=(2000, 1))
    assert ridge_r2(Z[:1000], y[:1000], Z[1000:], y[1000:]) > 0.9
    y_shuf = y.copy(); rng.shuffle(y_shuf)
    assert ridge_r2(Z[:1000], y_shuf[:1000], Z[1000:], y_shuf[1000:]) < 0.05


def test_spearman_and_auc_basics():
    a = np.arange(100).astype(float)
    assert spearman(a, a) > 0.999 and spearman(a, -a) < -0.999
    labels = a > 49
    assert auc(a, labels) > 0.99
    assert abs(auc(np.zeros(100) + np.random.default_rng(2).normal(0, 1e-9, 100), labels) - 0.5) < 0.2


def test_uniform_patch_changes_outputs():
    cfg = ModelCfg(d_model=16, heads=4, enc_hidden=8, n_blocks=2, use_id_embed=False)
    model = REGISTRY["attention"](12, 10, cfg).eval()
    X = torch.randn(3, 12, 10)
    with torch.no_grad():
        base, _ = model(X)
        for b in model.blocks:
            b.mhsa.patch_uniform = True
        patched, aux = model(X)
        for b in model.blocks:
            b.mhsa.patch_uniform = False
        restored, _ = model(X)
    assert not torch.allclose(base, patched)          # patch is wired
    assert torch.allclose(base, restored)             # and fully reversible
    torch.testing.assert_close(aux["attn"][0, 0, 0].sum(), torch.tensor(1.0))


def test_head_ablation_zeroes_one_head():
    cfg = ModelCfg(d_model=16, heads=4, enc_hidden=8, n_blocks=1, use_id_embed=False)
    model = REGISTRY["attention"](12, 10, cfg).eval()
    X = torch.randn(2, 12, 10)
    with torch.no_grad():
        base, _ = model(X)
        model.blocks[0].mhsa.ablate_heads = [2]
        abl, aux = model(X)
        model.blocks[0].mhsa.ablate_heads = ()
    assert not torch.allclose(base, abl)
    assert torch.all(aux["attn"][:, 2] == 0)
