# Backpropagation

The algorithm that computes `∇θ L` efficiently — i.e. how much each weight
contributed to the loss. It is just the **chain rule** applied systematically to
the network's computation graph, evaluated **backwards**.

## The idea
A network is a composition of simple operations. Build a graph where each node
remembers its inputs and how it was computed. Then:

1. **Forward pass:** compute outputs left→right, caching intermediate values.
2. **Backward pass:** seed `dL/dL = 1` at the output, then walk the graph in
   reverse. At each node, multiply the incoming gradient by the node's *local*
   derivative and route it to the node's inputs (chain rule).

Each operation needs only a **local rule**:
- `c = a + b` → `∂c/∂a = 1`, `∂c/∂b = 1` (gradient passes through to both).
- `c = a · b` → `∂c/∂a = b`, `∂c/∂b = a` (gradient scaled by the other factor).
- `c = φ(a)` → `∂c/∂a = φ'(a)`.

If a value feeds multiple consumers, its gradient is the **sum** of the
gradients from each path (multivariable chain rule) — in code, you `+=`. Forget
that accumulation and your gradients are silently wrong; it's the classic bug.

## Why it's efficient
Reverse-mode autodiff computes the gradient of one scalar output w.r.t. *all*
parameters in a single backward pass — cost ≈ one forward pass. That O(1)-passes
property (vs. perturbing each weight separately) is what makes training large
nets feasible.

## Lineage
- **Rumelhart, Hinton & Williams (1986)** popularized backprop for training
  multilayer nets (the idea has older roots in control theory / Werbos 1974).
- Modern frameworks (PyTorch, JAX) generalize this to tensors with autograd —
  exactly what you build, in miniature, in e04.

## Gradient checking (e06)
Always sanity-check analytic gradients against numerical ones:
```
∂L/∂θᵢ ≈ (L(θ + εeᵢ) − L(θ − εeᵢ)) / (2ε)     # ε ≈ 1e-5
```
Match to ~1e-6 ⇒ your backprop is correct.

**Links:** experiments [e04](../../experiments/tier1_backprop/e04_autograd_engine/),
[e06](../../experiments/tier1_backprop/e06_manual_backprop/) · code `core/engine.py`
· paper: Rumelhart et al. 1986 · related: [gradient-descent.md](gradient-descent.md)
