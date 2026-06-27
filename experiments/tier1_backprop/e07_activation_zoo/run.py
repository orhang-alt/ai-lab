"""e07 — Activation zoo & vanishing gradients (Tier 1).

Why deep nets abandoned sigmoid for ReLU: backprop multiplies local activation slopes
across layers. Sigmoid's slope is at most 0.25, so the product collapses toward 0 with
depth (vanishing gradients); ReLU's active slope is 1, so it doesn't shrink.

Run:  python experiments/tier1_backprop/e07_activation_zoo/run.py
"""

from __future__ import annotations

from core import activations as A


def main():
    print(f"max sigmoid'(z) = {float(A.sigmoid_prime(0.0)):.2f} (at z=0);  "
          f"relu'(z) = 1 on the active side\n")
    print("Best-case gradient scaling through L layers (product of max local slopes):")
    print(f"{'L':>4} | {'sigmoid  (0.25^L)':>18} | {'relu  (1^L)':>12}")
    print("  " + "-" * 38)
    for L in (1, 5, 10, 20, 40):
        print(f"{L:>4} | {0.25 ** L:>18.2e} | {1.0:>12.0f}")

    print("\nEven in the best case, stacking sigmoids multiplies slopes <= 0.25, so the")
    print("gradient -> 0 with depth (vanishing). ReLU's active slope is 1, so the product")
    print("stays O(1) — a core reason deep networks use ReLU-family activations + good init.")


if __name__ == "__main__":
    main()
