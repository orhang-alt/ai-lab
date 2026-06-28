"""e10 — Initialization and signal scale (Tier 2).

Run:
    python experiments/tier2_training/e10_initialization/run.py
"""

from __future__ import annotations

import numpy as np

from core import activations as A
from core import init


def forward_stats(kind, activation, depth=6, width=128, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(512, width))
    stats = []
    for _ in range(depth):
        if kind == "normal(1.0)":
            W = init.normal((width, width), std=1.0, rng=rng)
        elif kind == "xavier":
            W = init.xavier((width, width), rng=rng)
        elif kind == "he":
            W = init.he((width, width), rng=rng)
        else:
            raise ValueError(kind)
        z = x @ W.T
        x = A.tanh(z) if activation == "tanh" else A.relu(z)
        stats.append((float(x.mean()), float(x.std())))
    return stats


def print_table(title, rows):
    print(title)
    print(f"{'init':<12} {'layer1 std':>11} {'layer3 std':>11} {'layer6 std':>11}")
    print("-" * 50)
    for name, stats in rows:
        print(f"{name:<12} {stats[0][1]:>11.3f} {stats[2][1]:>11.3f} {stats[5][1]:>11.3f}")
    print()


def main():
    zero_W = init.zeros((3, 4))
    zero_out = np.ones((2, 4)) @ zero_W.T
    print("zero initialization symmetry check")
    print(f"outputs from 3 neurons with zero weights:\n{zero_out}")
    assert np.allclose(zero_out[:, 0], zero_out[:, 1])

    tanh_rows = [(name, forward_stats(name, "tanh")) for name in ("normal(1.0)", "xavier", "he")]
    relu_rows = [(name, forward_stats(name, "relu")) for name in ("normal(1.0)", "xavier", "he")]

    print()
    print_table("tanh stack: activation std by depth", tanh_rows)
    print_table("relu stack: activation std by depth", relu_rows)

    print("Aha: initialization is about keeping information alive before learning begins.")
    print("Xavier is usually a better first guess for tanh; He is usually a better first guess for ReLU.")


if __name__ == "__main__":
    main()

