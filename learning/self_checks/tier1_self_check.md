# Tier 1 Self-Check

Do this without looking at the source.

## Build
1. Implement a tiny `Value` class with `+`, `*`, `tanh`, and `backward()`.
2. Verify `d = a*b + c` by hand and with code.
3. Build a 2-layer MLP from `Value` objects.
4. Train XOR to 4/4 correct.
5. Numerically gradient-check one parameter.

## Explain
- Why topological order matters.
- Why gradients accumulate.
- Why hidden layers fix XOR.
- Why `zero_grad()` is part of every training loop.

## Pass Condition
You can rebuild scalar backprop and train XOR from memory.

