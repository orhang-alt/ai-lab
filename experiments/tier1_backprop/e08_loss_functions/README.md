# e08 — Loss Functions: MSE vs Cross-Entropy

> **Stub — you implement `core/losses.py`.**

## Concept
The loss defines *what* the network optimizes. For regression, MSE; for
classification, cross-entropy. CE pairs with sigmoid/softmax so that the
gradient simplifies to the clean `(prediction − target)` form — which also avoids
the flat-gradient problem MSE+sigmoid suffers from.

Read [`infobase/00_foundations/loss-functions.md`](../../../infobase/00_foundations/loss-functions.md).

## Hypothesis
For a saturated sigmoid output that is confidently *wrong*, MSE gives a tiny
gradient (slow learning) while BCE gives a large one (fast correction).

## Method (implement in `run.py`)
1. Implement `mse`, `bce`, `cross_entropy` in `core/losses.py`.
2. For a single sigmoid neuron with target 1 and output ≈ 0.001 (confident, wrong),
   compute dL/dz under MSE vs BCE; compare magnitudes.
3. (Optional) verify softmax+CE gradient equals `softmax(logits) − onehot(y)`.

## What to observe
- MSE+sigmoid gradient ∝ σ'(z) → ~0 when saturated → "stuck" neuron.
- BCE+sigmoid gradient ∝ (σ(z) − y) → large when wrong → escapes saturation.

## Conclusion
This is *why* classification uses cross-entropy. Record the gradient comparison in `notes.md`.
