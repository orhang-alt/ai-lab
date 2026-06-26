"""e05 — Train an MLP to solve XOR (STUB: implement the TODOs).

Depends on: core/engine.py (e04), core/losses.py + core/optim.py (e08/e09).
Run:  python experiments/tier1_backprop/e05_mlp_xor/run.py
"""

from __future__ import annotations

import numpy as np

INPUTS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
XOR = np.array([0, 1, 1, 0], dtype=float)


def main():
    # TODO: build a 2 -> 2 -> 1 MLP whose neurons use core.engine.Value params
    # TODO: pick a loss (core.losses) and optimizer (core.optim.SGD)
    # TODO: training loop:
    #         for step in range(N):
    #             opt.zero_grad()
    #             loss = sum of per-example losses (forward each input)
    #             loss.backward()
    #             opt.step()
    # TODO: print loss occasionally; assert final outputs match XOR
    # TODO: (optional) viz.plot_loss(history) and the decision boundary
    raise NotImplementedError("e05: train an MLP on XOR")


if __name__ == "__main__":
    main()
