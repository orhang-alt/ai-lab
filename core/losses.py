"""Loss functions (NumPy, Tier 1 e08).

Array-based losses for the vectorized NumPy tier and for e08's gradient comparison.
(The Value-based MLP in e05 builds its loss directly from Value arithmetic.)
"""

from __future__ import annotations

import numpy as np

from . import activations


def mse(y_pred, y_true):
    y_pred, y_true = np.asarray(y_pred, float), np.asarray(y_true, float)
    return float(np.mean((y_pred - y_true) ** 2))


def bce(y_pred, y_true, eps=1e-12):
    """Binary cross-entropy; y_pred are probabilities in (0,1)."""
    p = np.clip(np.asarray(y_pred, float), eps, 1 - eps)
    y = np.asarray(y_true, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def cross_entropy(logits, y_true):
    """Softmax + categorical cross-entropy against integer class labels.

    logits: (n_samples, n_classes); y_true: (n_samples,) integer labels.
    """
    logits = np.asarray(logits, float)
    y_true = np.asarray(y_true, int)
    p = activations.softmax(logits, axis=-1)
    n = logits.shape[0]
    return float(-np.mean(np.log(p[np.arange(n), y_true] + 1e-12)))
