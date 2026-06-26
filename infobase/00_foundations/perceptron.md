# The Perceptron Learning Rule

The first algorithm that *learns* a neuron's weights from labeled examples
(Rosenblatt, 1958). Activation is the step/sign function; targets are 0/1.

## The rule
For each training example `(x, y)`, predict `ŷ = step(w·x + b)`, then nudge the
weights toward the correct answer in proportion to the error:

```
error = y − ŷ                 # ∈ {−1, 0, +1}
w ← w + η · error · x         # η = learning rate
b ← b + η · error
```

- If correct (`error = 0`): no change.
- If it should have fired but didn't (`error = +1`): push `w` toward `x`.
- If it fired but shouldn't (`error = −1`): push `w` away from `x`.

Geometrically: each mistake rotates/shifts the decision hyperplane to fix that point.

## Convergence theorem
If the data is **linearly separable**, the perceptron rule is guaranteed to find
a separating hyperplane in a **finite** number of updates. If it's *not*
separable (e.g. XOR), it never converges — it cycles forever. This is the wall
in e03.

## Limits → what comes next
- Only linear boundaries. Multi-layer networks + backprop remove this limit.
- The update is not gradient descent on a smooth loss (the step function has zero
  gradient a.e.). Replacing step with a smooth activation + a differentiable loss
  is exactly what makes [backpropagation](backpropagation.md) possible.

**Links:** experiment [e02](../../experiments/tier0_neuron/e02_perceptron_gates/),
[e03](../../experiments/tier0_neuron/e03_xor_fails/) · paper: Rosenblatt 1958 ·
prev: [single-neuron.md](single-neuron.md) · next: [gradient-descent.md](gradient-descent.md)
