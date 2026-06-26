"""Numerical gradient check for the autograd engine.

Skips automatically until you implement core/engine.py (e04). Once implemented,
it verifies analytic gradients match central-difference numerical gradients.
"""

import pytest

from core.engine import Value


def _engine_ready():
    try:
        a, b = Value(2.0), Value(3.0)
        (a * b).backward()
        return True
    except NotImplementedError:
        return False


pytestmark = pytest.mark.skipif(
    not _engine_ready(), reason="core/engine.py not implemented yet (e04)"
)


def test_known_gradients():
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    d = a * b + c
    d.backward()
    assert d.data == pytest.approx(4.0)
    assert a.grad == pytest.approx(-3.0)   # = b.data
    assert b.grad == pytest.approx(2.0)    # = a.data
    assert c.grad == pytest.approx(1.0)


def test_numerical_gradcheck():
    # f(x) = tanh(3x + 1); check dy/dx against central difference.
    def f(xval):
        return (Value(xval) * 3 + 1).tanh().data

    x0, eps = 0.5, 1e-6
    numerical = (f(x0 + eps) - f(x0 - eps)) / (2 * eps)

    x = Value(x0)
    y = (x * 3 + 1).tanh()
    y.backward()
    assert x.grad == pytest.approx(numerical, abs=1e-5)
