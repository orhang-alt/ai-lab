"""Optimizers.  ===  YOU BUILD THIS IN e09 (SGD usable from e05)  ===

An optimizer updates a list of parameters from their `.grad`. Mirror the PyTorch
API so the muscle memory transfers:

    opt = SGD(params, lr=0.1)
    ...
    opt.zero_grad()      # reset grads to 0
    loss.backward()      # autograd fills .grad
    opt.step()           # update each param in-place

`params` is a list of `core.engine.Value` (Tier 1) — later, arrays/tensors.

Implement progressively:
    SGD        : p.data -= lr * p.grad
    Momentum   : v = mu*v - lr*g ; p += v
    Adam       : bias-corrected 1st/2nd moment estimates (Kingma & Ba 2014)
"""

from __future__ import annotations


class SGD:
    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0

    def step(self):
        # TODO: p.data -= self.lr * p.grad  for each param
        raise NotImplementedError("e09: implement SGD.step")


class Momentum:
    def __init__(self, params, lr=0.01, mu=0.9):
        self.params = list(params)
        self.lr = lr
        self.mu = mu
        self.v = [0.0 for _ in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0

    def step(self):
        raise NotImplementedError("e09: implement Momentum.step")


class Adam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = [0.0 for _ in self.params]
        self.v = [0.0 for _ in self.params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0

    def step(self):
        raise NotImplementedError("e09: implement Adam.step")
