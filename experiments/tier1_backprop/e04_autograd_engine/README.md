# e04 — Build a Scalar Autograd Engine

> **Stub — you implement `core/engine.py`.** This is the conceptual heart of the
> whole lab. Reference: Karpathy's `micrograd` (try yourself before peeking).

## Concept
Backprop = the chain rule applied to a computation graph. If you can build the
graph as you compute (each value remembering how it was made) and define a local
derivative rule per operation, then a single reverse pass computes the gradient
of the output w.r.t. every input. PyTorch does exactly this, over tensors.

Read [`infobase/00_foundations/backpropagation.md`](../../../infobase/00_foundations/backpropagation.md).

## Hypothesis
A ~150-line `Value` class supporting `+`, `*`, `**`, `tanh`, `relu`, and
`backward()` reproduces analytic gradients to within ~1e-6 of numerical
(finite-difference) gradients for arbitrary expressions.

## Method
1. Implement the TODOs in `core/engine.py`.
2. Run the smoke test in `run.py` (the `a*b+c` example).
3. Validate with `pytest tests/test_engine_gradcheck.py` (numerical gradient check).

## What to observe
- `d = a*b + c; d.backward()` gives `a.grad == b.data`, `b.grad == a.data`.
- Gradient check passes for randomly generated expressions.
- A value reused twice in the graph **accumulates** gradient (that `+=` matters).

## Conclusion
Once this passes, every later Tier 1+ experiment trains on top of it. Note any
gotchas (topological sort, the `+=` accumulation bug) in `notes.md`.
