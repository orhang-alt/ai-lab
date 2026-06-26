"""e03 — A single neuron cannot learn XOR (STUB: implement the TODOs).

Run:  python experiments/tier0_neuron/e03_xor_fails/run.py
"""

from __future__ import annotations

import numpy as np

INPUTS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
XOR = np.array([0, 1, 1, 0])


def main():
    # TODO: import train_perceptron from e02 and run it on (INPUTS, XOR)
    #       for, say, 200 epochs WITHOUT the early-stop on errors==0.
    # TODO: print the per-epoch error history and confirm it never hits 0.
    # TODO (optional): scatter the 4 points colored by label and observe that
    #       no single straight line separates the two classes.
    raise NotImplementedError("e03: run the perceptron on XOR and show it fails")


if __name__ == "__main__":
    main()
