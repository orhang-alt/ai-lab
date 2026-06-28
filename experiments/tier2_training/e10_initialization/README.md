# e10 — Initialization and Signal Scale

## Concept
Initialization decides what signals and gradients look like before learning starts.
Bad initialization can make neurons identical, saturate activations, or shrink
signals to zero.

## Hypothesis
Zero initialization destroys symmetry, naive large normal initialization makes
tanh saturate, Xavier keeps tanh signal scale steadier, and He keeps ReLU signal
scale steadier.

Read [`../../../infobase/04_derivations/sigmoid-derivative.md`](../../../infobase/04_derivations/sigmoid-derivative.md).

## Method
1. Feed random data through several random layers.
2. Compare activation mean/std layer by layer.
3. Show why all-zero rows create identical neurons.

## What to observe
- Zero weights make neurons produce the same output.
- Large normal weights push tanh toward saturation.
- Xavier and He are not magic constants; they preserve scale for different activations.

## Conclusion
Initialization is the first defense against vanishing/exploding signals.

