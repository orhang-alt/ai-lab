"""e07 — Activation functions & vanishing gradients (STUB: implement the TODOs).

Run:  python experiments/tier1_backprop/e07_activation_zoo/run.py [--plot]
"""

from __future__ import annotations

import numpy as np

from core import activations as A


def vanishing_demo(depth=50, width=1, seed=0):
    """Product of activation derivatives across `depth` layers."""
    rng = np.random.default_rng(seed)
    # TODO: for sigmoid and relu, sample random pre-activations per layer,
    #       multiply the derivative magnitudes across depth, and return the
    #       gradient magnitude as a function of depth (1..depth).
    raise NotImplementedError("e07: implement the vanishing-gradient simulation")


def main():
    print("sigmoid_prime(0) =", A.sigmoid_prime(0.0), "(max slope of sigmoid = 0.25)")
    print("relu_prime(2)    =", A.relu_prime(2.0))
    # TODO: call vanishing_demo() and print/plot gradient magnitude vs depth
    raise NotImplementedError("e07: compare sigmoid vs relu gradient decay")


if __name__ == "__main__":
    main()
