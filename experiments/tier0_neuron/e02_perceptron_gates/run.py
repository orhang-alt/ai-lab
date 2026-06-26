"""e02 — Perceptron learning rule (STUB: implement the TODOs).

Run:  python experiments/tier0_neuron/e02_perceptron_gates/run.py
"""

from __future__ import annotations

import numpy as np

from core.neuron import Neuron

INPUTS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
AND = np.array([0, 0, 0, 1])
OR = np.array([0, 1, 1, 1])


def train_perceptron(X, y, lr=0.1, epochs=20, seed=0):
    """Train a step-activation neuron with the perceptron rule.

    Returns (neuron, history) where history[i] = misclassifications in epoch i.
    """
    rng = np.random.default_rng(seed)
    neuron = Neuron(X.shape[1], activation="step", weights=rng.normal(0, 0.1, X.shape[1]), bias=0.0)
    history = []
    for _ in range(epochs):
        errors = 0
        for xi, yi in zip(X, y):
            # TODO: forward pass -> y_pred
            # TODO: compute update = lr * (yi - y_pred)
            # TODO: neuron.w += update * xi ; neuron.b += update
            # TODO: count errors when yi != y_pred
            raise NotImplementedError("e02: implement the perceptron update")
        history.append(errors)
        if errors == 0:
            break
    return neuron, history


def main():
    for name, y in (("AND", AND), ("OR", OR)):
        neuron, history = train_perceptron(INPUTS, y)
        print(f"{name}: converged in {len(history)} epochs, errors/epoch={history}")
        print(f"     learned {neuron}")


if __name__ == "__main__":
    main()
