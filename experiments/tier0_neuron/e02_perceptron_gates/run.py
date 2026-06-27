"""e02 — The perceptron learning rule (Rosenblatt, 1958).

The first algorithm that *learns* a single neuron's weights from examples. For each
training example, predict with a step activation and nudge the weights toward the
correct answer in proportion to the error:

    error = y_true - y_pred           # in {-1, 0, +1}
    w <- w + lr * error * x
    b <- b + lr * error

For a **linearly separable** gate (AND, OR) this converges in finite time
(the Perceptron Convergence Theorem). e03 shows it fails on XOR.

Run:  python experiments/tier0_neuron/e02_perceptron_gates/run.py
"""

from __future__ import annotations

import numpy as np

from core.neuron import Neuron  # reuse the lab's building block

INPUTS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
AND = np.array([0, 0, 0, 1])
OR = np.array([0, 1, 1, 1])


def train_perceptron(X, y, lr=0.1, epochs=20, seed=0, early_stop=True):
    """Train a step-activation neuron with the perceptron rule.

    Returns (neuron, history) where history[i] = # misclassifications in epoch i.
    """
    rng = np.random.default_rng(seed)
    neuron = Neuron(X.shape[1], activation="step",
                    weights=rng.normal(0, 0.1, X.shape[1]), bias=0.0)
    history = []
    for _ in range(epochs):
        errors = 0
        for xi, yi in zip(X, y):
            pred = float(neuron.forward(xi))     # step activation -> 0 or 1
            update = lr * (yi - pred)            # <-- the perceptron rule
            neuron.w += update * xi
            neuron.b += update
            errors += int(yi != pred)
        history.append(errors)
        if early_stop and errors == 0:
            break
    return neuron, history


def main():
    for name, y in (("AND", AND), ("OR", OR)):
        neuron, history = train_perceptron(INPUTS, y)
        converged = history[-1] == 0
        preds = [int(neuron.forward(x)) for x in INPUTS]
        print(f"\n{name} gate")
        print(f"  errors/epoch : {history}")
        print(f"  {'converged' if converged else 'did NOT converge'} in {len(history)} epochs")
        print(f"  learned      : {neuron}")
        print(f"  truth table  : {preds}   (target {y.tolist()})")
        assert preds == y.tolist(), "a linearly-separable gate should be learned exactly"
    print("\nAND and OR are linearly separable -> the perceptron converges. "
          "(e03 shows XOR is not, and never converges.)")


if __name__ == "__main__":
    main()
