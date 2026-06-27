"""Optimizers (Tier 1 e09; SGD usable from e05).

Each takes a list of `core.engine.Value` parameters and updates them from their `.grad`,
mirroring the PyTorch API:

    opt = SGD(params, lr=0.1)
    opt.zero_grad(); loss.backward(); opt.step()
"""

from __future__ import annotations


class _Base:
    def __init__(self, params):
        self.params = list(params)

    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0


class SGD(_Base):
    def __init__(self, params, lr=0.01, momentum=0.0):
        super().__init__(params)
        self.lr = lr
        self.mu = momentum
        self.v = [0.0 for _ in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            self.v[i] = self.mu * self.v[i] - self.lr * p.grad   # velocity (mu=0 → plain SGD)
            p.data += self.v[i]


# Kept as a distinct name for the e09 lesson; same idea as SGD(momentum=...).
class Momentum(SGD):
    def __init__(self, params, lr=0.01, mu=0.9):
        super().__init__(params, lr=lr, momentum=mu)


class Adam(_Base):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = [0.0 for _ in self.params]
        self.v = [0.0 for _ in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g          # 1st moment
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g      # 2nd moment
            m_hat = self.m[i] / (1 - self.b1 ** self.t)                   # bias correction
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * m_hat / (v_hat ** 0.5 + self.eps)
