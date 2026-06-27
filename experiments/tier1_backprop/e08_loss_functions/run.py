"""e08 — MSE vs cross-entropy gradients (Tier 1).

Why classification uses cross-entropy, not MSE: at a confident-but-wrong sigmoid output,
MSE's gradient carries a σ'(z) factor that nearly vanishes (slow learning), while
binary cross-entropy's gradient is the clean (p − y) — large when wrong.

Run:  python experiments/tier1_backprop/e08_loss_functions/run.py
"""

from __future__ import annotations

from core import activations as A


def main():
    z = -7.0               # pre-activation; target y=1 → the neuron is confidently WRONG
    p = float(A.sigmoid(z))
    y = 1.0
    print(f"pre-activation z = {z}, sigmoid(z) = {p:.5f}, target y = {y}\n")

    dmse = (p - y) * float(A.sigmoid_prime(z))   # MSE + sigmoid: (p-y)·σ'(z)
    dbce = (p - y)                               # BCE + sigmoid: (p-y)  — σ' cancels

    print(f"MSE + sigmoid : dL/dz = (p-y)·σ'(z) = {dmse:.6f}")
    print(f"BCE + sigmoid : dL/dz = (p-y)        = {dbce:.6f}")
    print(f"\nBCE's gradient is ~{abs(dbce / dmse):.0f}× larger here. MSE's σ'(z) factor "
          "nearly\nvanishes when the neuron is saturated, so it barely learns from a "
          "confident\nmistake — which is exactly why classification pairs sigmoid/softmax "
          "with cross-entropy.")


if __name__ == "__main__":
    main()
