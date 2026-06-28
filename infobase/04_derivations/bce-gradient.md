# Binary Cross-Entropy Gradient

Binary cross-entropy:

```
L = -[y log(p) + (1-y) log(1-p)]
```

If `p = sigmoid(z)`, the derivative with respect to the logit simplifies to:

```
dL/dz = p - y
```

Why it matters: for a confident wrong prediction, `p - y` is large. The sigmoid
slope cancels out, so the model receives a strong correction.

