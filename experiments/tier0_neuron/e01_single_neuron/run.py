"""e01 — Single artificial neuron behaviour (fully worked reference).

Run:
    python experiments/tier0_neuron/e01_single_neuron/run.py
    python experiments/tier0_neuron/e01_single_neuron/run.py --plot
"""

from __future__ import annotations

import argparse

import numpy as np

from core.neuron import Neuron

# The four 2-bit inputs, used for every 2-input gate below.
INPUTS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)


def truth_table(neuron, inputs=INPUTS):
    """Return list of (input, pre_activation z, output a)."""
    rows = []
    for x in inputs:
        z = float(neuron.pre_activation(x))
        a = float(neuron.forward(x))
        rows.append((x.astype(int).tolist(), z, a))
    return rows


def print_table(title, neuron, inputs=INPUTS):
    print(f"\n{title}")
    print(f"  {neuron}")
    print("   x0 x1 |     z   |  output")
    print("  -------+---------+--------")
    for x, z, a in truth_table(neuron, inputs):
        xs = " ".join(str(v) for v in x)
        print(f"   {xs:5} | {z:7.2f} | {a:6.3f}")


def main(plot=False):
    print("=" * 60)
    print("e01 — The single artificial neuron:  a = phi(w . x + b)")
    print("=" * 60)

    # 1) AND gate.  w=[1,1], b=-1.5  ->  fires only when x0+x1 >= 2 (i.e. 1.5).
    AND = Neuron(2, activation="step", weights=[1.0, 1.0], bias=-1.5)
    print_table("AND gate (step, w=[1,1], b=-1.5)", AND)

    # 2) OR gate.  Same weights — only the BIAS changes (boundary slides).
    OR = Neuron(2, activation="step", weights=[1.0, 1.0], bias=-0.5)
    print_table("OR gate  (step, w=[1,1], b=-0.5)  <- only bias changed", OR)

    # 3) NOT gate.  Single input, negative weight flips the response.
    NOT = Neuron(1, activation="step", weights=[-1.0], bias=0.5)
    print("\nNOT gate (step, w=[-1], b=0.5)")
    print(f"  {NOT}")
    for x in (0.0, 1.0):
        print(f"   NOT {int(x)} = {int(NOT.forward([x]))}")

    # 4) Hand-trace verification for one AND input, x=(1,1).
    print("\nHand-trace check for AND on x=(1,1):")
    z_by_hand = 1.0 * 1.0 + 1.0 * 1.0 + (-1.5)  # w0*x0 + w1*x1 + b
    z_by_code = float(AND.pre_activation([1, 1]))
    print(f"   z by hand = 1*1 + 1*1 + (-1.5) = {z_by_hand}")
    print(f"   z by code = {z_by_code}")
    assert np.isclose(z_by_hand, z_by_code), "hand trace must match the code!"
    print("   match ✓  (step(0.5) = 1, so AND(1,1)=1)")

    # 5) Sigmoid neuron: same AND-ish weights, but graded output = "confidence".
    SIG = Neuron(2, activation="sigmoid", weights=[4.0, 4.0], bias=-6.0)
    print_table("Sigmoid neuron (w=[4,4], b=-6): graded, not hard 0/1", SIG)
    print("   -> note outputs approach but never reach 0 and 1.")

    if plot:
        make_plots(AND)


def make_plots(and_neuron):
    """Optional: requires matplotlib (pip install -e '.[viz]')."""
    import matplotlib.pyplot as plt

    from core import activations as A
    from core import viz

    # Activation functions side by side.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in ("step", "sigmoid", "tanh", "relu"):
        fn, _ = A.get(name)
        viz.plot_activation(fn, ax=axes[0], label=name)
    axes[0].set_title("Activation functions")

    # AND decision boundary: predict returns {0,1}.
    viz.plot_decision_boundary(
        lambda X: (and_neuron.forward(X) >= 0.5).astype(int),
        INPUTS,
        np.array([0, 0, 0, 1]),  # AND labels
        ax=axes[1],
    )
    axes[1].set_title("AND decision boundary (w·x + b = 0)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="show matplotlib plots")
    main(**vars(ap.parse_args()))
