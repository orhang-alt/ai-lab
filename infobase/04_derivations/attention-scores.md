# Attention Scores

Attention compares queries and keys:

```
scores = Q K^T / sqrt(d_k)
weights = softmax(scores)
output = weights V
```

Why divide by `sqrt(d_k)`: dot products get larger as vector dimension grows.
Scaling keeps softmax from becoming too sharp too early.

For causal self-attention, future-token scores are masked before softmax, so each
position can only attend to earlier positions.

