# e11 — Regularization and Overfitting

## Concept
Overfitting happens when a model learns the quirks of the training set instead of
the pattern that generalizes. Regularization adds pressure toward simpler
solutions.

## Hypothesis
A high-degree polynomial can drive train error low while validation error stays
high. L2 regularization increases train error slightly but improves validation
error and shrinks the weights.

## Method
1. Generate noisy samples from a smooth curve.
2. Fit a high-degree polynomial with and without L2 regularization.
3. Compare train MSE, validation MSE, and weight norm.

## What to observe
- The unregularized model chases noise.
- L2 regularization prefers smaller weights.
- Generalization is judged on validation loss, not training loss.

## Conclusion
The goal is not "lowest training loss"; the goal is useful behavior on new data.

