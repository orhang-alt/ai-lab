"""Loss functions.  ===  YOU BUILD THIS IN e08 (and use earlier in e05)  ===

Start with NumPy versions (operate on arrays), then, once the autograd engine
exists, you can write Value-based versions used by the MLP trainer.

Implement:
    mse(y_pred, y_true)               mean squared error
    bce(y_pred, y_true)               binary cross-entropy (y_pred in (0,1))
    cross_entropy(logits, y_true)     softmax + categorical cross-entropy

For each, also derive the gradient w.r.t. y_pred on paper (e06/e08) — that
derivation is the point, not the code.
"""

from __future__ import annotations

import numpy as np


def mse(y_pred, y_true):
    # TODO: mean((y_pred - y_true)**2)
    raise NotImplementedError("e08: implement mse")


def bce(y_pred, y_true, eps=1e-12):
    # TODO: -mean(y*log(p) + (1-y)*log(1-p)); clip p to [eps, 1-eps]
    raise NotImplementedError("e08: implement bce")


def cross_entropy(logits, y_true):
    # TODO: softmax(logits) then categorical cross-entropy against integer labels
    raise NotImplementedError("e08: implement cross_entropy")
