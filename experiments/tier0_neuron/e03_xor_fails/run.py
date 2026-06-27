"""e03 — A single neuron cannot learn XOR.

XOR is **not linearly separable**: no straight line separates
{(0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0} (the two "1" corners sit on opposite
diagonals). So the perceptron rule (e02) **never converges** — the misclassification
count keeps oscillating and never reaches 0. This is the 1969 Minsky & Papert result
that motivates hidden layers + backprop (Tier 1, e05).

Run:  python experiments/tier0_neuron/e03_xor_fails/run.py
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np

INPUTS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
XOR = np.array([0, 1, 1, 0])

# Reuse e02's perceptron rule by path (no reimplementation, no package needed).
_e02 = pathlib.Path(__file__).resolve().parents[1] / "e02_perceptron_gates" / "run.py"
_spec = importlib.util.spec_from_file_location("e02_run", _e02)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
train_perceptron = _mod.train_perceptron


def main():
    epochs = 200
    neuron, history = train_perceptron(INPUTS, XOR, epochs=epochs, early_stop=False)
    preds = [int(neuron.forward(x)) for x in INPUTS]

    print(f"Ran the perceptron rule on XOR for {epochs} epochs.")
    print(f"  best (min) errors reached : {min(history)}   (0 would mean success)")
    print(f"  errors in the last 20 epochs: {history[-20:]}")
    print(f"  final truth table          : {preds}   (target {XOR.tolist()})")

    assert min(history) > 0, "XOR is not linearly separable — errors can never reach 0"
    print("\n-> A single neuron NEVER classifies all four correctly: XOR is not linearly\n"
          "   separable. The fix is a HIDDEN LAYER (Tier 1, e05) — an MLP bends input space\n"
          "   so XOR becomes separable.")


if __name__ == "__main__":
    main()
