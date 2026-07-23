import torch

from crossfolio.losses import hhi, make_loss, neg_batch_sharpe, neg_mean_excess


def _batch():
    torch.manual_seed(0)
    B, N = 32, 8
    y = torch.randn(B, N) * 0.05
    y_spy = torch.zeros(B)
    return B, N, y, y_spy


def test_beating_benchmark_lowers_loss():
    B, N, y, y_spy = _batch()
    best = y.argmax(dim=1)
    w_good = torch.nn.functional.one_hot(best, N).float()      # picks each row's winner
    w_flat = torch.full((B, N), 1 / N)
    for loss in (neg_mean_excess, neg_batch_sharpe):
        assert loss(w_good, y, y_spy) < loss(w_flat, y, y_spy)


def test_positive_excess_means_negative_loss():
    B, N, y, y_spy = _batch()
    w = torch.full((B, N), 1 / N)
    y_pos = torch.abs(y)                                        # portfolio always beats spy
    assert neg_mean_excess(w, y_pos, y_spy) < 0
    assert neg_batch_sharpe(w, y_pos, y_spy) < 0


def test_hhi_direction():
    N = 8
    w_flat = torch.full((1, N), 1 / N)
    w_allin = torch.nn.functional.one_hot(torch.tensor([0]), N).float()
    assert torch.isclose(hhi(w_flat), torch.tensor(1 / N))
    assert torch.isclose(hhi(w_allin), torch.tensor(1.0))
    assert hhi(w_flat) < hhi(w_allin)


def test_make_loss_penalizes_concentration():
    B, N, y, y_spy = _batch()
    y_same = torch.zeros(B, N)                                  # no return difference at all
    w_flat = torch.full((B, N), 1 / N)
    w_allin = torch.nn.functional.one_hot(torch.zeros(B, dtype=torch.long), N).float()
    loss = make_loss("mean_excess", hhi_lambda=0.1)
    assert loss(w_flat, y_same, y_spy) < loss(w_allin, y_same, y_spy)


def test_losses_differentiable():
    B, N, y, y_spy = _batch()
    theta = torch.zeros(B, N, requires_grad=True)
    w = torch.softmax(theta, dim=-1)
    make_loss("sharpe", 0.05)(w, y, y_spy).backward()
    assert theta.grad is not None and torch.isfinite(theta.grad).all()
