# Sigmoid Derivative

Sigmoid:

```
sigma(z) = 1 / (1 + exp(-z))
```

Derivative:

```
d sigma / dz = sigma(z) * (1 - sigma(z))
```

Why it matters: the derivative is largest at `z = 0` and becomes tiny in the
tails. A saturated sigmoid learns slowly because backprop multiplies by this
small local slope.

Memory check: derive it once using the chain rule on `(1 + exp(-z))^-1`.

