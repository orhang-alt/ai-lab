"""e06 — Gradient checking (Tier 1).

Trust, but verify: compare the autograd engine's analytic gradients to a numerical
(central-difference) estimate. If they match to ~1e-6, backprop is correct.

Run:  python experiments/tier1_backprop/e06_manual_backprop/run.py
"""

from __future__ import annotations

from core.nn import MLP


def numerical_grad(loss_fn, p, eps=1e-5):
    """Central-difference dL/d(p.data): (L(p+eps) - L(p-eps)) / 2eps."""
    orig = p.data
    p.data = orig + eps
    plus = loss_fn().data
    p.data = orig - eps
    minus = loss_fn().data
    p.data = orig
    return (plus - minus) / (2 * eps)


def main():
    model = MLP(2, [4, 1], nonlin="tanh", out_nonlin="tanh", seed=7)
    x, y = [0.5, -0.2], 1.0

    def loss_fn():
        return (model(x) - y) ** 2

    # analytic gradients from backprop
    L = loss_fn()
    model.zero_grad()
    L.backward()

    max_err = 0.0
    for p in model.parameters():
        ng = numerical_grad(loss_fn, p)
        max_err = max(max_err, abs(ng - p.grad))

    print(f"loss = {L.data:.4f}")
    print(f"parameters checked: {len(model.parameters())}")
    print(f"max |analytic - numerical| gradient error: {max_err:.2e}")
    assert max_err < 1e-4, "analytic gradients must match numerical ones"
    print("\nBackprop verified against central differences ✓  (the engine is correct).")


if __name__ == "__main__":
    main()
