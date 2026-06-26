"""e04 — Smoke test for your autograd engine.

Implement core/engine.py first, then:
    python experiments/tier1_backprop/e04_autograd_engine/run.py
"""

from __future__ import annotations

from core.engine import Value


def main():
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    d = a * b + c          # forward
    print(f"d.data = {d.data}  (expect 4.0)")

    d.backward()           # reverse
    print(f"a.grad = {a.grad}  (expect -3.0, = b)")
    print(f"b.grad = {b.grad}  (expect  2.0, = a)")
    print(f"c.grad = {c.grad}  (expect  1.0)")

    # A slightly bigger expression with a nonlinearity.
    x = Value(0.5)
    y = (x * 3 + 1).tanh()
    y.backward()
    print(f"\ny = tanh(3x+1) at x=0.5 -> y={y.data:.4f}, dy/dx={x.grad:.4f}")


if __name__ == "__main__":
    main()
