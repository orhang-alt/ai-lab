# e07 — Activation Zoo & Vanishing Gradients

> **Implemented in `run.py`.** `core/activations.py` supplies the functions; here
> you study their shapes and gradient behavior.

## Concept
The choice of activation shapes both expressiveness and trainability. Saturating
activations (sigmoid, tanh) have near-zero derivative in their tails, so stacking
many of them makes gradients **vanish** during backprop. ReLU/GELU largely fix this.

## Hypothesis
In a deep stack, the product of sigmoid derivatives across layers shrinks toward
0 (gradients vanish); with ReLU the gradient magnitude stays O(1) for active units.

## Method
1. Plot each activation and its derivative (`core.activations.*_prime`).
2. Simulate an L-layer chain: multiply random pre-activations' derivatives across
   depth for sigmoid vs relu; plot gradient magnitude vs depth L = 1..50.
3. (Optional) Train the e05 XOR MLP with each activation; compare convergence.

## What to observe
- sigmoid_prime maxes at 0.25 → gradient ≤ 0.25^L through L layers → vanishes fast.
- ReLU keeps gradient at 1 on the active side (but can "die" at 0).

## Conclusion
This is *why* modern nets use ReLU/GELU + good init (e10) + normalization. Note in `notes.md`.
