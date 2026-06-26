"""Shared visualization helpers (matplotlib).

Kept in core/ so every experiment plots results the same way. matplotlib is an
optional dependency — install with `pip install -e ".[viz]"`. If it's missing,
these functions raise a clear message instead of an ImportError deep in a plot.
"""

from __future__ import annotations

import numpy as np

try:
    import matplotlib.pyplot as plt

    _HAVE_MPL = True
except ImportError:  # pragma: no cover
    _HAVE_MPL = False


def _require_mpl():
    if not _HAVE_MPL:
        raise RuntimeError(
            "matplotlib is not installed. Run: pip install -e \".[viz]\"  "
            "(experiments still run headless; plotting is optional)."
        )


def plot_activation(fn, lo=-6.0, hi=6.0, n=400, ax=None, label=None):
    """Plot an activation function over [lo, hi]."""
    _require_mpl()
    z = np.linspace(lo, hi, n)
    ax = ax or plt.gca()
    ax.plot(z, fn(z), label=label or getattr(fn, "__name__", "phi"))
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.legend()
    return ax


def plot_decision_boundary(predict, X, y, ax=None, resolution=200):
    """Plot a 2D decision boundary for a binary `predict(X) -> {0,1}` function.

    X: (n, 2) points, y: (n,) labels in {0,1}.
    """
    _require_mpl()
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    ax = ax or plt.gca()
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = np.asarray(predict(grid)).reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=0.3, levels=1)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
    return ax


def plot_loss(history, ax=None, label="loss"):
    """Plot a list/array of loss values over training steps."""
    _require_mpl()
    ax = ax or plt.gca()
    ax.plot(np.asarray(history), label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend()
    return ax


def show():
    _require_mpl()
    plt.show()
