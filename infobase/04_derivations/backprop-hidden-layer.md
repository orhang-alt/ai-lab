# Backprop Through One Hidden Layer

Forward pass:

```
z1 = W1 x + b1
h  = phi(z1)
z2 = W2 h + b2
L  = loss(z2, y)
```

Backward pass idea:

```
dL/dW2 = dL/dz2 * h
dL/db2 = dL/dz2
dL/dh  = W2^T * dL/dz2
dL/dz1 = dL/dh * phi'(z1)
dL/dW1 = dL/dz1 * x
dL/db1 = dL/dz1
```

Why it matters: every layer receives a signal from the layer after it, multiplies
by its local derivative, and passes the result backward.

