"""core — the reusable building blocks of the lab.

Import from here in every experiment; never reimplement these inside an experiment.

Tier 0 (implemented):  activations, neuron (NumPy forward), viz
Tier 1 (you build):    engine (Value autograd), losses, optim, init
"""

from . import activations, neuron, viz  # noqa: F401

__all__ = ["activations", "neuron", "viz"]
