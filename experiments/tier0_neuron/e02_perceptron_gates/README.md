# e02 — The Perceptron Learning Rule

> **Implemented in `run.py`.** Read the training loop and then try changing the
> learning rate, initialization seed, or target gate.

## Concept
In e01 you *hand-set* the weights. Here the neuron **learns** them from examples
using Rosenblatt's perceptron rule (1958) — the first learning algorithm for a
neuron. For each misclassified example:

```
w ← w + η (y_true − y_pred) x
b ← b + η (y_true − y_pred)
```

Read [`infobase/00_foundations/perceptron.md`](../../../infobase/00_foundations/perceptron.md).

## Hypothesis
For a **linearly separable** gate (AND, OR), the perceptron rule converges to
weights that classify all examples correctly, in a finite number of passes
(the Perceptron Convergence Theorem).

## Method
1. Start from `core.neuron.Neuron(2, activation="step")` with small random weights.
2. Loop over epochs; for each training example apply the update rule above.
3. Track the number of misclassifications per epoch.
4. Run for AND, then OR.

## What to observe
- Misclassifications → 0; weights stabilize.
- The learned boundary is one of many valid separators (depends on init/order).
- Try `η` (learning rate) values — does it affect *whether* it converges?

## Conclusion
Record in `notes.md`: epochs to converge, final weights, and how they compare to
the hand-set values from e01.
