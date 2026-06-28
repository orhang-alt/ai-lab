# Adam Update

Adam keeps two moving averages:

```
m_t = beta1 * m_(t-1) + (1-beta1) * g_t
v_t = beta2 * v_(t-1) + (1-beta2) * g_t^2
```

Bias correction:

```
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)
```

Update:

```
w = w - lr * m_hat / (sqrt(v_hat) + eps)
```

Why it matters: Adam combines momentum-like direction memory with per-parameter
step-size scaling.

