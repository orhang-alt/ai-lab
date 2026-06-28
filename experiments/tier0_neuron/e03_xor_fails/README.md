# e03 — Why a Single Neuron Cannot Learn XOR

> **Implemented in `run.py`.** This is the "wall" that motivates multi-layer
> networks.

## Concept
XOR is **not linearly separable**: no single straight line separates the
outputs `{(0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0}`. A single neuron can only carve
input space with *one* hyperplane, so it cannot represent XOR — this is the
1969 Minsky & Papert result that helped trigger the first "AI winter".

## Hypothesis
Running the e02 perceptron rule on XOR will **never converge**: the
misclassification count oscillates and never reaches 0, no matter how many epochs.

## Method
1. Reuse `train_perceptron` from e02 (import it) with the XOR targets.
2. Run for many epochs; plot/print misclassifications per epoch.
3. (Optional) plot the four points and try to draw one line splitting them.

## What to observe
- Errors never settle to 0 (typically stuck at ≥1).
- Visually: the two classes sit on opposite diagonals — unsplittable by a line.

## Conclusion
This is the cliff. The fix in **Tier 1**: stack neurons into layers and train
them with **backpropagation** (e04–e05), where an MLP *does* solve XOR. Note in
`notes.md` exactly how the failure manifests, so the e05 success feels earned.
