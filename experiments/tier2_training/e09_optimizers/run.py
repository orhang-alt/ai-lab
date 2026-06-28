"""e09 — Optimizers: SGD, Momentum, Adam (Tier 2).

Run:
    python experiments/tier2_training/e09_optimizers/run.py
"""

from __future__ import annotations

from core.engine import Value
from core.optim import Adam, Momentum, SGD


def loss_for(x, y):
    # A tilted bowl with different curvature per direction.
    return (x - 3.0) ** 2 + 8.0 * (y + 2.0) ** 2


def run(name, make_opt, steps=160):
    x, y = Value(-4.0), Value(4.0)
    opt = make_opt([x, y])
    history = []
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_for(x, y)
        loss.backward()
        opt.step()
        history.append(loss.data)
    return name, x.data, y.data, history


def main():
    runs = [
        run("SGD", lambda params: SGD(params, lr=0.03)),
        run("Momentum", lambda params: Momentum(params, lr=0.03, mu=0.80)),
        run("Adam", lambda params: Adam(params, lr=0.18)),
    ]

    print("target minimum: x=3, y=-2\n")
    print(f"{'optimizer':<10} {'final x':>9} {'final y':>9} {'loss@0':>10} {'loss@end':>10}")
    print("-" * 55)
    for name, x, y, history in runs:
        print(f"{name:<10} {x:>9.3f} {y:>9.3f} {history[0]:>10.3f} {history[-1]:>10.6f}")

    print("\nAha: backprop supplies gradients; the optimizer decides how to use them.")
    print("Try changing each learning rate. Good optimizer behavior is still sensitive to scale.")

    assert all(history[-1] < history[0] for _, _, _, history in runs)


if __name__ == "__main__":
    main()
