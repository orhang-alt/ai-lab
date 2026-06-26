"""Weight initialization schemes.  ===  YOU BUILD THIS IN e10  ===

Why it matters: bad init makes deep nets either saturate (vanishing gradients)
or explode. e10 will show this empirically.

Implement (all return a NumPy array of the requested shape):
    zeros(shape)
    normal(shape, std, rng)
    xavier(shape, rng)   # a.k.a. Glorot: std = sqrt(1 / fan_in)  (good for tanh/sigmoid)
    he(shape, rng)       # Kaiming: std = sqrt(2 / fan_in)        (good for relu)

`shape` is (fan_out, fan_in) to match DenseLayer's weight matrix.
"""

from __future__ import annotations

import numpy as np


def zeros(shape):
    return np.zeros(shape)


def normal(shape, std=0.5, rng=None):
    rng = rng or np.random.default_rng()
    return rng.normal(0.0, std, size=shape)


def xavier(shape, rng=None):
    # TODO: std = sqrt(1 / fan_in), where fan_in = shape[1]
    raise NotImplementedError("e10: implement xavier")


def he(shape, rng=None):
    # TODO: std = sqrt(2 / fan_in)
    raise NotImplementedError("e10: implement he")
