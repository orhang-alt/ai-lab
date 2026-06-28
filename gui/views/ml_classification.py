import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import lessons
import ml_lessons

LESSON = ml_lessons.CLASSIFICATION


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _fit_logreg(X, y, lr=0.3, steps=800):
    """Logistic regression via gradient descent; returns a proba(X) function."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        p = _sigmoid(Xs @ w + b)
        w -= lr * (Xs.T @ (p - y)) / len(y)
        b -= lr * (p - y).mean()
    return lambda Xn: _sigmoid(((np.asarray(Xn) - mu) / sd) @ w + b)


st.title("M2 · Classification — playground & lesson")
st.caption("Logistic regression = a sigmoid neuron. Move the threshold and watch the "
           "confusion matrix and precision/recall trade off.")

tab_play, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🎛 Playground", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

with tab_play:
    left, right = st.columns([0.42, 0.58])
    with left:
        n = st.slider("points per class", 20, 200, 60, key="c_n")
        sep = st.slider("class separation", 0.2, 3.0, 1.4, 0.1, key="c_sep")
        spread = st.slider("spread / overlap (σ)", 0.3, 2.5, 1.0, 0.1, key="c_spread")
        seed = st.number_input("seed", 0, 9999, 0, key="c_seed")
        t = st.slider("decision threshold", 0.05, 0.95, 0.50, 0.05, key="c_thr")

    rng = np.random.default_rng(int(seed))
    X0 = rng.normal([-sep, -sep], spread, (int(n), 2))
    X1 = rng.normal([sep, sep], spread, (int(n), 2))
    X = np.vstack([X0, X1])
    y = np.r_[np.zeros(int(n)), np.ones(int(n))]

    proba = _fit_logreg(X, y)
    p = proba(X)
    pred = (p >= t).astype(int)
    TP = int(((pred == 1) & (y == 1)).sum())
    FP = int(((pred == 1) & (y == 0)).sum())
    FN = int(((pred == 0) & (y == 1)).sum())
    TN = int(((pred == 0) & (y == 0)).sum())
    acc = (TP + TN) / len(y)
    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    with right:
        lim = float(np.abs(X).max()) + 0.5
        gx = np.linspace(-lim, lim, 200)
        xx, yy = np.meshgrid(gx, gx)
        zz = proba(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        fig, ax = plt.subplots(figsize=(4.7, 4.3))
        cf = ax.contourf(xx, yy, zz, levels=20, cmap="RdBu", alpha=0.8, vmin=0, vmax=1)
        ax.contour(xx, yy, zz, levels=[t], colors="k", linewidths=1.8)  # boundary at p=t
        ax.scatter(X0[:, 0], X0[:, 1], s=20, color="#A32D2D", edgecolors="white",
                   linewidths=0.4, label="class 0", zorder=3)
        ax.scatter(X1[:, 0], X1[:, 1], s=20, color="#185FA5", edgecolors="white",
                   linewidths=0.4, label="class 1", zorder=3)
        ax.set_title(f"P(class 1) + boundary at p={t:.2f}")
        ax.legend(fontsize=8, loc="upper left")
        fig.colorbar(cf, ax=ax, shrink=0.8, label="p")
        st.pyplot(fig, width="stretch")

    m = st.columns(4)
    m[0].metric("Accuracy", f"{acc:.2f}")
    m[1].metric("Precision", f"{prec:.2f}")
    m[2].metric("Recall", f"{rec:.2f}")
    m[3].metric("F1", f"{f1:.2f}")

    cm = pd.DataFrame([[TN, FP], [FN, TP]],
                      index=["actual 0", "actual 1"], columns=["pred 0", "pred 1"])
    st.markdown("**Confusion matrix**")
    st.dataframe(cm, width="content")
    st.info("Lower the threshold → more points called **class 1** → recall ↑, precision ↓ "
            "(a spam filter wants high precision; a cancer screen wants high recall).",
            icon=":material/lightbulb:")

with tab_theory:
    st.markdown(LESSON.theory, unsafe_allow_html=True)

with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(LESSON.quiz, prefix="mlclf")

with tab_tasks:
    st.subheader("Tasks")
    st.markdown(LESSON.tasks)

with tab_ref:
    st.subheader("Reading & references")
    st.markdown(LESSON.references)
