# e05 — An MLP Solves XOR

> **Implemented in `run.py` using `core.nn.MLP`.** The payoff for e03 + e04: a
> 2-layer network learns what one neuron couldn't.

## Concept
A hidden layer lets the network bend input space into a representation where XOR
*is* linearly separable. With the autograd engine from e04, training is just:
forward → loss → `loss.backward()` → optimizer step.

## Hypothesis
A `2 → 2 → 1` MLP with tanh/sigmoid activations, trained with SGD on BCE/MSE
loss, reaches ~0 error on XOR within a few hundred steps.

## Method
1. Build an MLP of `Value`-based neurons with `core.nn.MLP`.
2. Use `core.optim.SGD`.
3. Train loop: zero_grad → forward all 4 examples → loss → backward → step.
4. Print the loss curve checkpoints and final truth table.

## What to observe
- Loss drops to near 0; all four XOR outputs become correct.
- The decision boundary is now **nonlinear** (two regions, not one line).
- Compare training time/behaviour to the failed perceptron in e03.

## Conclusion
You now understand, end to end, what a framework's `.fit()` does. Record in `notes.md`.
