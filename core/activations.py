"""Activation functions and their derivatives (NumPy, Tier 0).

Each function works elementwise on scalars or NumPy arrays. The `*_prime`
variants return the derivative w.r.t. the pre-activation input `z`, which you'll
need for hand-rolled backprop in Tier 1 (e06) before the autograd engine exists.
"""

from __future__ import annotations

import numpy as np


def step(z):
    """Heaviside step — the original McCulloch-Pitts / perceptron activation."""
    return np.where(np.asarray(z) >= 0.0, 1.0, 0.0)


def sign(z):
    """+1 / -1 step (used by the classic perceptron with bipolar targets)."""
    return np.where(np.asarray(z) >= 0.0, 1.0, -1.0)


def linear(z):
    return np.asarray(z, dtype=float)


def linear_prime(z):
    return np.ones_like(np.asarray(z, dtype=float))


def sigmoid(z):
    # numerically stable logistic
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1.0 - s)


def tanh(z):
    return np.tanh(np.asarray(z, dtype=float))


def tanh_prime(z):
    return 1.0 - np.tanh(np.asarray(z, dtype=float)) ** 2


def relu(z):
    return np.maximum(0.0, np.asarray(z, dtype=float))


def relu_prime(z):
    return np.where(np.asarray(z) > 0.0, 1.0, 0.0)


def gelu(z):
    # tanh approximation (the one used in GPT-2)
    z = np.asarray(z, dtype=float)
    return 0.5 * z * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * z**3)))


def softmax(z, axis=-1):
    """Stable softmax over `axis`."""
    z = np.asarray(z, dtype=float)
    z = z - np.max(z, axis=axis, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=axis, keepdims=True)


# Registry so experiments can select an activation by name.
REGISTRY = {
    "step": (step, None),
    "sign": (sign, None),
    "linear": (linear, linear_prime),
    "sigmoid": (sigmoid, sigmoid_prime),
    "tanh": (tanh, tanh_prime),
    "relu": (relu, relu_prime),
}


def get(name):
    """Return (fn, fn_prime) for `name`. Raises KeyError on unknown names."""
    return REGISTRY[name]
