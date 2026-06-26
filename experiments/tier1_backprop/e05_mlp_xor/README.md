# e05 — An MLP Solves XOR

> **Stub.** The payoff for e03 + e04: a 2-layer network learns what one neuron couldn't.

## Concept
A hidden layer lets the network bend input space into a representation where XOR
*is* linearly separable. With the autograd engine from e04, training is just:
forward → loss → `loss.backward()` → optimizer step.

## Hypothesis
A `2 → 2 → 1` MLP with tanh/sigmoid activations, trained with SGD on BCE/MSE
loss, reaches ~0 error on XOR within a few hundred steps.

## Method (implement in `run.py`)
1. Build an MLP of `Value`-based neurons (you may extend `core/neuron.py` with a
   `Value`-backed `MLP`, or build a small one here and promote it to core later).
2. Loss from `core/losses.py`; optimizer from `core/optim.py` (`SGD`).
3. Train loop: zero_grad → forward all 4 examples → loss → backward → step.
4. Plot the loss curve (`core.viz.plot_loss`) and the learned decision boundary.

## What to observe
- Loss drops to near 0; all four XOR outputs become correct.
- The decision boundary is now **nonlinear** (two regions, not one line).
- Compare training time/behaviour to the failed perceptron in e03.

## Conclusion
You now understand, end to end, what a framework's `.fit()` does. Record in `notes.md`.
