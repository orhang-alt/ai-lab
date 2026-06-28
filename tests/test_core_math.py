"""Focused checks for the reusable core helpers added after Tier 1."""

import numpy as np
import pytest

from core import init, losses
from core.engine import Value
from core.nn import MLP
from core.optim import Adam, Momentum, SGD


def test_losses_match_hand_calculations():
    assert losses.mse([1.0, 2.0], [0.0, 4.0]) == pytest.approx(2.5)

    y_pred = np.array([0.8, 0.2])
    y_true = np.array([1.0, 0.0])
    expected_bce = -np.mean(np.log([0.8, 0.8]))
    assert losses.bce(y_pred, y_true) == pytest.approx(expected_bce)

    logits = np.array([[3.0, 1.0], [0.0, 2.0]])
    ce = losses.cross_entropy(logits, [0, 1])
    assert ce < 0.2


def test_initializers_return_expected_shapes_and_scales():
    rng = np.random.default_rng(0)

    assert np.array_equal(init.zeros((2, 3)), np.zeros((2, 3)))
    assert init.normal((2, 3), std=0.1, rng=rng).shape == (2, 3)

    xavier = init.xavier((500, 100), rng=np.random.default_rng(1))
    he = init.he((500, 100), rng=np.random.default_rng(2))

    assert xavier.shape == (500, 100)
    assert he.shape == (500, 100)
    assert np.std(xavier) == pytest.approx(np.sqrt(1.0 / 100), rel=0.08)
    assert np.std(he) == pytest.approx(np.sqrt(2.0 / 100), rel=0.08)


def test_optimizers_update_value_parameters():
    p = Value(1.0)
    p.grad = 0.5
    SGD([p], lr=0.1).step()
    assert p.data == pytest.approx(0.95)

    q = Value(1.0)
    opt = Momentum([q], lr=0.1, mu=0.9)
    q.grad = 0.5
    opt.step()
    q.grad = 0.5
    opt.step()
    assert q.data == pytest.approx(0.855)

    r = Value(1.0)
    r.grad = 0.5
    Adam([r], lr=0.1).step()
    assert r.data == pytest.approx(0.9)


def test_mlp_parameter_count_and_backward_flow():
    model = MLP(2, [3, 1], nonlin="tanh", out_nonlin="tanh", seed=0)
    params = model.parameters()

    assert len(params) == 13
    assert all(isinstance(p, Value) for p in params)

    out = model([1.0, -1.0])
    loss = (out - 1.0) ** 2
    loss.backward()

    assert any(abs(p.grad) > 0 for p in params)
