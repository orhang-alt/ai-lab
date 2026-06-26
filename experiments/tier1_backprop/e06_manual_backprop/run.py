"""e06 — Gradient checking (STUB: implement the TODOs).

Run:  python experiments/tier1_backprop/e06_manual_backprop/run.py
"""

from __future__ import annotations


def numerical_grad(f, get_set_param, eps=1e-5):
    """Central-difference gradient of scalar f w.r.t. one parameter.

    get_set_param: (getter, setter) for the single float parameter under test.
    Returns the approximate dL/dparam.
    """
    # TODO: save p; set p+eps -> f_plus; set p-eps -> f_minus; restore p
    # TODO: return (f_plus - f_minus) / (2*eps)
    raise NotImplementedError("e06: implement central-difference numerical_grad")


def main():
    # TODO: build a tiny 2-layer net with core.engine.Value params
    # TODO: for each param, compare engine grad (after backward) to numerical_grad
    # TODO: print max absolute and relative error; expect ~1e-6
    raise NotImplementedError("e06: run the gradient check")


if __name__ == "__main__":
    main()
