# Gradient Descent

How weights actually learn: take repeated steps **downhill** on the loss surface.

```
θ ← θ − η · ∇θ L
```
- `θ` — all parameters (weights, biases).
- `∇θ L` — gradient of the loss w.r.t. θ (computed by [backprop](backpropagation.md)).
- `η` — learning rate. Too big → diverge/oscillate; too small → crawl.

## Variants (batch size)
- **Batch GD** — gradient over the whole dataset per step. Stable, slow, memory-heavy.
- **SGD** — one example (or mini-batch) per step. Noisy but fast; the noise can
  help escape shallow minima. Mini-batch (e.g. 32–256) is the practical default.

## Better update rules (e09)
- **Momentum** — accumulate a velocity `v = μv − η∇L`; smooths and accelerates
  along consistent directions, damping oscillation across ravines.
- **Adam** (Kingma & Ba, 2014) — per-parameter adaptive step using running
  estimates of the 1st and 2nd moment of the gradient (with bias correction).
  Robust default for deep nets.

## What can go wrong
- **Learning rate** is the single most important knob. Use a schedule (warmup +
  decay) for deep nets.
- **Initialization** (e10) sets the starting point and gradient scale.
- **Local minima / saddles** — less of a problem in high dimensions than feared;
  saddles + poor conditioning matter more, which is what momentum/Adam address.

**Links:** experiment [e09](../../experiments/tier2_training/) · code `core/optim.py`
· paper: Kingma & Ba 2014 (Adam) · related: [backpropagation.md](backpropagation.md)
