# Binary Cross-Entropy Gradyanı

Binary cross-entropy:

```
L = -[y log(p) + (1-y) log(1-p)]
```

Eğer `p = sigmoid(z)` ise logit'e göre türev sadeleşir:

```
dL/dz = p - y
```

Neden önemli: Kendinden emin ama yanlış bir tahminde `p - y` büyüktür. Sigmoid'in
eğimi sadeleştiği için model güçlü bir düzeltme sinyali alır.

