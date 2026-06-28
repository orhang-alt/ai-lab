# Tier 1 Questions — Backprop and Autograd

Answer these aloud before looking at code.

1. Why does `backward()` start by setting the output gradient to `1.0`?
2. Why does a computation graph need a topological order?
3. Why do gradients accumulate with `+=` instead of assignment?
4. For `d = a*b + c`, why are `dd/da = b`, `dd/db = a`, and `dd/dc = 1`?
5. What does a local derivative rule know, and what does it not know?
6. Why does ReLU pass gradient only on the active side?
7. Why can an MLP solve XOR?
8. What exactly does `zero_grad()` prevent?
9. Why does cross-entropy learn faster than MSE for confident wrong classification?
10. What does gradient checking prove, and what does it not prove?

