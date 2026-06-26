# e06 — Manual Backprop + Gradient Checking

> **Stub. Mostly pen-and-paper, then verify in code.**

## Concept
Before trusting autograd, derive the gradients of a tiny 2-layer net **by hand**,
then confirm your engine (e04) and your derivation agree with a **numerical**
gradient (finite differences). Gradient checking is the standard way to catch
backprop bugs.

## Hypothesis
Analytic gradients (hand-derived and engine-computed) match the numerical
gradient `(f(x+ε) − f(x−ε)) / 2ε` to ~1e-6 for every parameter.

## Method
1. On paper: write `z1 = W1 x + b1`, `h = φ(z1)`, `z2 = W2 h + b2`,
   `L = loss(z2, y)`. Derive `dL/dW2, dL/db2, dL/dW1, dL/db1` via the chain rule.
2. In `run.py`: implement `numerical_grad(f, param, eps=1e-5)`.
3. Compare numerical vs analytic (engine) gradients; report max abs/rel error.

## What to observe
- Errors are tiny (~1e-7) when correct; a sign error or missing term blows up to ~1.
- ε too small → floating-point noise; too large → truncation error. Find the sweet spot.

## Conclusion
Write the hand derivation into `notes.md` (this is the single most clarifying
exercise in the lab).
