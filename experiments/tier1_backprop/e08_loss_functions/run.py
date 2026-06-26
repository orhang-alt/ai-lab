"""e08 — MSE vs cross-entropy gradients (STUB: implement the TODOs).

Run:  python experiments/tier1_backprop/e08_loss_functions/run.py
"""

from __future__ import annotations

import numpy as np

from core import activations as A
from core import losses


def main():
    # A single sigmoid output that is confidently WRONG: target=1, output~0.001
    z = -7.0
    p = A.sigmoid(z)
    print(f"pre-activation z={z}, sigmoid(z)={p:.5f}, target=1")

    # TODO: implement losses.mse / losses.bce, then:
    # TODO: dL/dz for MSE+sigmoid  = (p - 1) * sigmoid_prime(z)
    # TODO: dL/dz for BCE+sigmoid  = (p - 1)          # the clean form
    # TODO: print both magnitudes and explain why BCE learns faster here.
    raise NotImplementedError("e08: compare MSE vs BCE gradients at saturation")


if __name__ == "__main__":
    main()
