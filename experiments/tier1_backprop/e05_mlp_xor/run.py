"""e05 — An MLP solves XOR (Tier 1).

The payoff for e03 (one neuron fails on XOR) + e04 (the autograd engine): a 2-layer
network with a hidden layer learns XOR. Training is just forward → loss →
loss.backward() → optimizer step, all on the `Value` engine.

Run:  python experiments/tier1_backprop/e05_mlp_xor/run.py
"""

from __future__ import annotations

from core.nn import MLP
from core.optim import SGD

# XOR with bipolar targets (tanh output in (-1, 1) → predict by sign).
DATA = [([0, 0], -1.0), ([0, 1], 1.0), ([1, 0], 1.0), ([1, 1], -1.0)]


def main():
    model = MLP(2, [8, 1], nonlin="tanh", out_nonlin="tanh", seed=1337)
    opt = SGD(model.parameters(), lr=0.1)

    for step in range(300):
        loss = sum((model(x) - y) ** 2 for x, y in DATA) * (1.0 / len(DATA))  # mean squared error
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == 299:
            print(f"step {step:3d}   loss {loss.data:.4f}")

    print("\nlearned XOR:")
    correct = 0
    for x, y in DATA:
        out = model(x).data
        pred = 1 if out >= 0 else -1
        correct += int(pred == y)
        print(f"  {x} -> {out:+.3f}   (pred {pred:+d}, target {int(y):+d})")

    print(f"\n{correct}/4 correct. A hidden layer bends input space so XOR becomes "
          "linearly separable — what one neuron (e03) could never do.")
    assert correct == 4, "the MLP should solve XOR"


if __name__ == "__main__":
    main()
