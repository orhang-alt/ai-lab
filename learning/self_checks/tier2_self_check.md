# Tier 2 Self-Check

Do this without looking at the source.

## Build
1. Compare SGD, Momentum, and Adam on the same one-dimensional loss.
2. Show all-zero initialization creates identical neurons.
3. Compare normal, Xavier, and He initialization through many layers.
4. Create a small overfitting example.
5. Add L2 regularization and show validation behavior improves.

## Explain
- Why optimizer choice changes the path, not the objective.
- Why initialization is about preserving signal and gradient scale.
- Why lower training loss is not always better.

## Pass Condition
You can diagnose whether a failed run is caused by learning rate, initialization,
or overfitting.

