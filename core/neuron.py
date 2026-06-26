"""The single artificial neuron, and a NumPy forward-only layer/MLP (Tier 0).

This is the foundational building block. A neuron computes:

        z = w . x + b           (weighted sum + bias, the "pre-activation")
        a = phi(z)              (activation)

where `w` is the weight vector, `x` the input vector, `b` the bias scalar, and
`phi` an activation function from `core.activations`.

Tier 0 is forward-only (no learning yet): you set/inspect weights by hand and
watch behaviour. Learning rules arrive in e02 (perceptron) and e05 (backprop via
the autograd engine).
"""

from __future__ import annotations

import numpy as np

from . import activations


class Neuron:
    """A single artificial neuron with a fixed activation function.

    Parameters
    ----------
    n_inputs : int
        Number of inputs (length of the weight vector).
    activation : str
        Name registered in `core.activations.REGISTRY` (e.g. "step", "sigmoid").
    weights, bias : optional
        Provide explicit values to inspect specific behaviour; otherwise small
        random weights and zero bias are used.
    rng : np.random.Generator, optional
        For reproducible initialization.
    """

    def __init__(self, n_inputs, activation="sigmoid", weights=None, bias=0.0, rng=None):
        rng = rng or np.random.default_rng()
        self.n_inputs = n_inputs
        self.activation_name = activation
        self.phi, self.phi_prime = activations.get(activation)
        if weights is None:
            weights = rng.normal(0.0, 0.5, size=n_inputs)
        self.w = np.asarray(weights, dtype=float).reshape(n_inputs)
        self.b = float(bias)

    def pre_activation(self, x):
        """z = w . x + b (the linear part), supports a single vector or a batch."""
        x = np.asarray(x, dtype=float)
        return x @ self.w + self.b

    def forward(self, x):
        """a = phi(z)."""
        return self.phi(self.pre_activation(x))

    __call__ = forward

    def __repr__(self):
        return (
            f"Neuron(n_inputs={self.n_inputs}, activation={self.activation_name!r}, "
            f"w={np.round(self.w, 3).tolist()}, b={round(self.b, 3)})"
        )


class DenseLayer:
    """A layer of `n_units` neurons sharing inputs (NumPy forward only).

    Weight matrix W has shape (n_units, n_inputs); bias b has shape (n_units,).
    forward(x) accepts x of shape (n_inputs,) or a batch (n_samples, n_inputs).
    """

    def __init__(self, n_inputs, n_units, activation="sigmoid", rng=None):
        rng = rng or np.random.default_rng()
        self.phi, self.phi_prime = activations.get(activation)
        self.activation_name = activation
        self.W = rng.normal(0.0, 0.5, size=(n_units, n_inputs))
        self.b = np.zeros(n_units)

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        z = x @ self.W.T + self.b
        return self.phi(z)

    __call__ = forward


class MLP:
    """Stack of DenseLayers (NumPy forward only). Training comes in Tier 1."""

    def __init__(self, layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    __call__ = forward
