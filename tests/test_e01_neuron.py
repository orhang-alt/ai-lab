"""Tier 0 neuron behaves as a linear classifier (these pass out of the box)."""

import numpy as np

from core.activations import sigmoid
from core.neuron import Neuron

INPUTS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)


def _outputs(neuron):
    return np.array([float(neuron.forward(x)) for x in INPUTS])


def test_and_gate():
    n = Neuron(2, activation="step", weights=[1.0, 1.0], bias=-1.5)
    assert np.array_equal(_outputs(n), [0, 0, 0, 1])


def test_or_gate_is_and_with_shifted_bias():
    n = Neuron(2, activation="step", weights=[1.0, 1.0], bias=-0.5)
    assert np.array_equal(_outputs(n), [0, 1, 1, 1])


def test_not_gate():
    n = Neuron(1, activation="step", weights=[-1.0], bias=0.5)
    assert float(n.forward([0])) == 1.0
    assert float(n.forward([1])) == 0.0


def test_pre_activation_matches_hand_trace():
    n = Neuron(2, activation="step", weights=[1.0, 1.0], bias=-1.5)
    assert np.isclose(float(n.pre_activation([1, 1])), 0.5)


def test_sigmoid_is_graded_not_binary():
    n = Neuron(2, activation="sigmoid", weights=[4.0, 4.0], bias=-6.0)
    out = _outputs(n)
    assert np.all((out > 0) & (out < 1))          # never exactly 0 or 1
    assert out[3] > out[0]                          # (1,1) more "on" than (0,0)


def test_sigmoid_function_basic():
    assert np.isclose(sigmoid(0.0), 0.5)
    assert sigmoid(50) > 0.99 and sigmoid(-50) < 0.01
