# MSE Gradyanı

Tek tahmin için:

```
L = (y_hat - y)^2
dL/dy_hat = 2 * (y_hat - y)
```

`n` örnek üzerinden mean squared error:

```
L = mean((y_hat - y)^2)
dL/dy_hat = 2 * (y_hat - y) / n
```

Neden önemli: MSE regresyon için doğaldır; ama sınıflandırmada doymuş sigmoid ile
eşleşirse `sigma'(z)` ile çarpılır ve düzeltme çok küçük kalabilir.

