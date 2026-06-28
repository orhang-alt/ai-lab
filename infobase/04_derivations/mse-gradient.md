# MSE Gradient

For one prediction:

```
L = (y_hat - y)^2
dL/dy_hat = 2 * (y_hat - y)
```

For mean squared error over `n` examples:

```
L = mean((y_hat - y)^2)
dL/dy_hat = 2 * (y_hat - y) / n
```

Why it matters: MSE is natural for regression, but when paired with a saturated
sigmoid for classification it also multiplies through `sigma'(z)`, which can make
the correction very small.

