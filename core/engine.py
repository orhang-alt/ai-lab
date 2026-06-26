"""Scalar reverse-mode autograd engine.  ===  YOU BUILD THIS IN e04  ===

Goal: a tiny `Value` class that wraps a single float, records the operations
used to compute it, and can compute gradients of any output w.r.t. all inputs
via backpropagation. This is the conceptual heart of every deep learning
framework (PyTorch's autograd is the same idea, just over tensors).

Reference: Karpathy's `micrograd`. Try to implement it yourself first using the
spec below; only peek at micrograd if you get stuck.

------------------------------------------------------------------------------
SPEC
------------------------------------------------------------------------------
Each Value holds:
    data      : float                  the forward value
    grad      : float                  d(final_output)/d(self), starts at 0.0
    _backward : callable               local rule that pushes grad to parents
    _prev     : set[Value]             the parent Values that produced this one
    _op       : str                    label for debugging/graphviz

Operations to support (each returns a new Value and wires up _backward):
    __add__, __mul__, __pow__ (float exponent), and from these: __neg__,
    __sub__, __truediv__, plus reflected ops (__radd__, __rmul__) so that
    `2 * x` works. Activations: tanh() and relu() as methods.

backward():
    1. topologically sort the graph reachable from self
    2. set self.grad = 1.0
    3. walk nodes in reverse topo order, calling each node's _backward()

Example once implemented:
    a = Value(2.0); b = Value(-3.0); c = Value(10.0)
    d = a * b + c          # d.data == 4.0
    d.backward()
    a.grad == -3.0         # d(d)/d(a) == b
    b.grad ==  2.0         # d(d)/d(b) == a

Validate with tests/test_engine_gradcheck.py (compares to numerical gradients).
------------------------------------------------------------------------------
"""

from __future__ import annotations


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # --- ops to implement ----------------------------------------------------
    def __add__(self, other):
        # TODO: out = Value(self.data + other.data, (self, other), '+')
        #       define out._backward to add out.grad to self.grad and other.grad
        raise NotImplementedError("e04: implement __add__")

    def __mul__(self, other):
        # TODO: product rule — self.grad += other.data * out.grad, and symmetric
        raise NotImplementedError("e04: implement __mul__")

    def __pow__(self, other):
        # TODO: power rule for a constant exponent `other`
        raise NotImplementedError("e04: implement __pow__")

    def tanh(self):
        # TODO: t = tanh(self.data); local derivative is (1 - t**2)
        raise NotImplementedError("e04: implement tanh")

    def relu(self):
        # TODO: out = max(0, x); local derivative is 1 if x > 0 else 0
        raise NotImplementedError("e04: implement relu")

    def backward(self):
        # TODO: topo-sort, seed self.grad = 1.0, walk in reverse calling _backward
        raise NotImplementedError("e04: implement backward")

    # --- conveniences derived from the above (no extra autograd wiring) -------
    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
