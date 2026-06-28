# Adam Güncellemesi

Adam iki hareketli ortalama tutar:

```
m_t = beta1 * m_(t-1) + (1-beta1) * g_t
v_t = beta2 * v_(t-1) + (1-beta2) * g_t^2
```

Bias düzeltmesi:

```
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)
```

Güncelleme:

```
w = w - lr * m_hat / (sqrt(v_hat) + eps)
```

Neden önemli: Adam, momentum benzeri yön hafızasını her parametre için ayrı adım
ölçeğiyle birleştirir.

