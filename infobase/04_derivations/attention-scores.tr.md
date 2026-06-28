# Attention Skorları

Attention query ve key vektörlerini karşılaştırır:

```
scores = Q K^T / sqrt(d_k)
weights = softmax(scores)
output = weights V
```

Neden `sqrt(d_k)` ile bölüyoruz: vektör boyutu büyüdükçe dot product değerleri
büyür. Ölçekleme, softmax'in çok erken aşırı keskinleşmesini engeller.

Causal self-attention'da gelecekteki token skorları softmax'ten önce maskelenir;
böylece her konum yalnızca önceki konumlara bakabilir.

