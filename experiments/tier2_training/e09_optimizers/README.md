# e09 — Optimizers: SGD, Momentum, Adam

## Concept
An optimizer decides how parameters move after backprop gives gradients. The loss
function is the landscape; the optimizer is the walking rule.

## Hypothesis
On the same curved loss, plain SGD moves directly along the current gradient,
Momentum smooths the path by remembering velocity, and Adam adapts step sizes
using first and second moment estimates.

Read [`../../../infobase/04_derivations/adam-update.md`](../../../infobase/04_derivations/adam-update.md).

## Method
1. Start all optimizers from the same two parameters.
2. Minimize the same bowl-shaped loss.
3. Print the final parameters and loss after the same number of steps.

## What to observe
- SGD works, but step size is fragile.
- Momentum often moves faster once velocity points downhill.
- Adam handles different gradient scales well.

## Conclusion
Optimizer choice changes the path through parameter space, not the objective.

