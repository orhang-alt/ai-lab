# Sigmoid Türevi

Sigmoid:

```
sigma(z) = 1 / (1 + exp(-z))
```

Türev:

```
d sigma / dz = sigma(z) * (1 - sigma(z))
```

Neden önemli: Türev en büyük değerini `z = 0` noktasında alır ve uçlarda çok
küçülür. Doymuş sigmoid yavaş öğrenir çünkü backprop bu küçük yerel eğimle çarpar.

Hafıza kontrolü: Bunu bir kez `(1 + exp(-z))^-1` ifadesine chain rule uygulayarak
türet.

