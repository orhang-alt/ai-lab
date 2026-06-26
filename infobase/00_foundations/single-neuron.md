# The Single Artificial Neuron

The atom of every neural network. A neuron maps an input vector **x** to a single
scalar output **a**:

```
z = w · x + b          # pre-activation  (a weighted sum + bias)
a = φ(z)               # activation       (a nonlinearity)
```

- **w** — weight vector. One weight per input; sign and magnitude say how much,
  and in which direction, each input pushes the output.
- **b** — bias. Shifts the threshold; lets the neuron fire even when all inputs
  are 0. Equivalently, a weight on a constant `+1` input.
- **φ** — activation function. Without it, stacking neurons would collapse into a
  single linear map (composition of linear functions is linear). The
  nonlinearity is what makes depth meaningful.

## Geometric picture
`z = w · x + b = 0` defines a **hyperplane** (a line in 2D, a plane in 3D). The
neuron asks: *which side of this hyperplane is x on, and how far?* `w` is the
normal vector (direction the plane faces); `b` slides it toward/away from the
origin. A step/sign activation turns "which side" into a hard 0/1; a sigmoid
turns "how far" into a graded confidence.

This is why a single neuron is a **linear classifier**: it can only split input
space with one straight cut. (That limitation is the whole point of e03/XOR.)

## Lineage
- **McCulloch & Pitts (1943)** — first mathematical model of a neuron: binary
  inputs, a threshold, binary output. Showed neurons can compute logic.
- **Rosenblatt (1958)** — the *Perceptron*: added a learning rule so the weights
  could be found from data instead of set by hand → [perceptron.md](perceptron.md).
- **Minsky & Papert (1969)** — proved a single-layer perceptron can't represent
  XOR, cooling the field until backprop revived multilayer nets.

## Worked example (from e01)
An AND gate with a step activation: `w = [1, 1]`, `b = -1.5`.

| x0 | x1 | z = x0+x1−1.5 | step(z) |
|----|----|----------------|---------|
| 0  | 0  | −1.5           | 0       |
| 0  | 1  | −0.5           | 0       |
| 1  | 0  | −0.5           | 0       |
| 1  | 1  | +0.5           | 1       |

Change **only the bias** to −0.5 and the same neuron becomes OR — the boundary
just slid. That single knob, `b`, is the threshold.

## Mental model for a SW dev
A neuron is `dot(w, x) + b` piped through one nonlinear function. A *network* is
just many of these composed — a differentiable program whose "constants" (the
weights) are found by gradient descent instead of written by you.

**Links:** experiment [e01](../../experiments/tier0_neuron/e01_single_neuron/) ·
code `core/neuron.py`, `core/activations.py` · next: [perceptron.md](perceptron.md)
