# Tek Gizli Katmanda Backprop

Forward pass:

```
z1 = W1 x + b1
h  = phi(z1)
z2 = W2 h + b2
L  = loss(z2, y)
```

Backward pass fikri:

```
dL/dW2 = dL/dz2 * h
dL/db2 = dL/dz2
dL/dh  = W2^T * dL/dz2
dL/dz1 = dL/dh * phi'(z1)
dL/dW1 = dL/dz1 * x
dL/db1 = dL/dz1
```

Neden önemli: Her katman kendinden sonraki katmandan sinyal alır, kendi yerel
türeviyle çarpar ve sonucu geriye iletir.

