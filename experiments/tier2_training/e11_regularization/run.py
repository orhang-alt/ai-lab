"""e11 — Regularization and overfitting (Tier 2).

Run:
    python experiments/tier2_training/e11_regularization/run.py
"""

from __future__ import annotations

import numpy as np


def make_data(seed=2):
    rng = np.random.default_rng(seed)
    x_train = np.linspace(-1.0, 1.0, 18)
    y_train = np.sin(3.0 * x_train) + rng.normal(0.0, 0.18, size=x_train.shape)
    x_val = np.linspace(-0.95, 0.95, 80)
    y_val = np.sin(3.0 * x_val)
    return x_train, y_train, x_val, y_val


def poly_features(x, degree):
    return np.stack([x ** i for i in range(degree + 1)], axis=1)


def fit_ridge(X, y, l2):
    penalty = l2 * np.eye(X.shape[1])
    penalty[0, 0] = 0.0  # do not regularize the bias term
    return np.linalg.solve(X.T @ X + penalty, X.T @ y)


def mse(pred, y):
    return float(np.mean((pred - y) ** 2))


def main():
    x_train, y_train, x_val, y_val = make_data()
    degree = 14
    X_train = poly_features(x_train, degree)
    X_val = poly_features(x_val, degree)

    configs = [("no L2", 0.0), ("L2=1e-3", 1e-3), ("L2=1e-1", 1e-1)]
    print(f"degree-{degree} polynomial on {len(x_train)} noisy samples\n")
    print(f"{'model':<10} {'train MSE':>11} {'val MSE':>11} {'weight norm':>13}")
    print("-" * 52)
    results = []
    for name, l2 in configs:
        w = fit_ridge(X_train, y_train, l2)
        train_mse = mse(X_train @ w, y_train)
        val_mse = mse(X_val @ w, y_val)
        norm = float(np.linalg.norm(w[1:]))
        results.append((name, train_mse, val_mse, norm))
        print(f"{name:<10} {train_mse:>11.5f} {val_mse:>11.5f} {norm:>13.2f}")

    no_l2 = results[0]
    best = min(results, key=lambda row: row[2])
    print("\nAha: the unregularized model can win on train loss and lose on validation loss.")
    print(f"Best validation here: {best[0]} (val MSE {best[2]:.5f}).")

    assert best[2] < no_l2[2]


if __name__ == "__main__":
    main()

