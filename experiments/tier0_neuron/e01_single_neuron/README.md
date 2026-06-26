# e01 — The Single Artificial Neuron

> **This experiment is fully worked — it's the reference template.** Read the
> code, run it, change the numbers, and watch what happens. Every later
> experiment follows this same `README` shape: Hypothesis → Method → Observe →
> Conclusion.

## Concept
A single neuron computes a **weighted sum of its inputs plus a bias**, then
passes that through an **activation function**:

```
z = w·x + b          (pre-activation: a line/plane in input space)
a = φ(z)             (activation: squashes / thresholds z)
```

Everything in deep learning is built from this. Read
[`infobase/00_foundations/single-neuron.md`](../../../infobase/00_foundations/single-neuron.md) first.

## Hypothesis
With **hand-chosen weights and bias** (no learning yet), a single neuron with a
step activation can implement the linearly-separable logic gates **AND, OR, NOT**,
and the weights have a clear geometric meaning: `w·x + b = 0` is the decision
boundary. A sigmoid activation turns the hard threshold into a graded
"confidence".

## Method
`run.py`:
1. Builds a 2-input neuron with explicit weights and a **step** activation.
2. Sets weights/bias to realize **AND**, then **OR**, then a 1-input **NOT**.
3. Hand-traces one input and checks it against `Neuron.pre_activation` / `forward`.
4. Swaps in a **sigmoid** activation to show graded output.
5. (If matplotlib is installed) plots the activation functions and the AND
   decision boundary.

```bash
python experiments/tier0_neuron/e01_single_neuron/run.py
# optional plots:
python experiments/tier0_neuron/e01_single_neuron/run.py --plot
```

## What to observe
- The full truth tables match the target gates exactly.
- Moving **only the bias** `b` shifts AND→OR (it slides the boundary).
- Negative weights invert the response (that's how NOT works).
- The sigmoid neuron outputs values *near* 0/1 but never exactly — magnitude of
  `w` controls how sharp the transition is.

## Things to try (turn observation into intuition)
- Make a **NAND** gate. (Hint: it's AND with negated weights and bias.)
- Find weights for XOR. *You can't* — that's the cliff e03 walks you off.
- Scale `w` by 10×. What happens to the sigmoid output sharpness?

## Conclusion
Record your findings in [`notes.md`](notes.md).
