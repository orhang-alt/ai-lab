# Softmax + Cross-Entropy

Softmax:

```
p_i = exp(z_i) / sum_j exp(z_j)
```

Cross-entropy for class `k`:

```
L = -log(p_k)
```

Combined gradient:

```
dL/dz_i = p_i - 1(i == k)
```

Why it matters: classification training becomes "predicted probability minus
target one-hot". This is why logits plus cross-entropy are the standard pairing.

