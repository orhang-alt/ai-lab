# Softmax + Cross-Entropy

Softmax:

```
p_i = exp(z_i) / sum_j exp(z_j)
```

Sınıf `k` için cross-entropy:

```
L = -log(p_k)
```

Birleşik gradyan:

```
dL/dz_i = p_i - 1(i == k)
```

Neden önemli: Sınıflandırma eğitimi "tahmin edilen olasılık eksi hedef one-hot"
haline gelir. Bu yüzden logits + cross-entropy standart eşleşmedir.

