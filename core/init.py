"""Weight initialization schemes (Tier 2 e10).

`shape` is (fan_out, fan_in) to match a dense weight matrix. Good init keeps the signal
and gradient variance roughly constant across layers (avoids vanishing/exploding).
"""

from __future__ import annotations

import numpy as np


def zeros(shape):
    return np.zeros(shape)


def normal(shape, std=0.5, rng=None):
    rng = rng or np.random.default_rng()
    return rng.normal(0.0, std, size=shape)


def xavier(shape, rng=None):
    """Glorot: std = sqrt(1 / fan_in). Good for tanh/sigmoid."""
    rng = rng or np.random.default_rng()
    fan_in = shape[1] if len(shape) > 1 else shape[0]
    return rng.normal(0.0, np.sqrt(1.0 / fan_in), size=shape)


def he(shape, rng=None):
    """Kaiming: std = sqrt(2 / fan_in). Good for ReLU."""
    rng = rng or np.random.default_rng()
    fan_in = shape[1] if len(shape) > 1 else shape[0]
    return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=shape)
