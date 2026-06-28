import pathlib
import sys
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import lessons
import ml_lessons

LESSON = ml_lessons.FOUNDATIONS
SCALE = 3.0  # x range is [-3, 3]; fit on x/SCALE for numerical stability


def _true(t):
    return np.sin(1.3 * t) + 0.3 * t


st.title("M0 · Foundations — playground & lesson")
st.caption("The core idea of ML: generalize, don't memorize. Crank model complexity and watch "
           "underfitting turn into overfitting.")

tab_play, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🎛 Playground", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

with tab_play:
    left, right = st.columns([0.42, 0.58])
    with left:
        n = st.slider("data points", 20, 200, 60, key="f_n")
        noise = st.slider("noise σ", 0.0, 1.0, 0.25, 0.05, key="f_noise")
        degree = st.slider("model complexity (polynomial degree)", 1, 12, 3, key="f_deg")
        train_frac = st.slider("train fraction", 0.3, 0.9, 0.6, 0.05, key="f_tf")
        seed = st.number_input("seed", 0, 9999, 0, key="f_seed")

    rng = np.random.default_rng(int(seed))
    x = np.sort(rng.uniform(-3.0, 3.0, int(n)))
    y = _true(x) + rng.normal(0.0, noise, int(n))

    idx = rng.permutation(len(x))
    n_tr = max(2, int(len(x) * train_frac))
    tr, te = idx[:n_tr], idx[n_tr:]
    xtr, ytr, xte, yte = x[tr], y[tr], x[te], y[te]

    eff_deg = min(int(degree), len(xtr) - 1)  # can't fit degree >= #train points
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coef = np.polyfit(xtr / SCALE, ytr, eff_deg)
    pred = lambda t: np.polyval(coef, np.asarray(t) / SCALE)

    tr_mse = float(np.mean((pred(xtr) - ytr) ** 2))
    te_mse = float(np.mean((pred(xte) - yte) ** 2)) if len(xte) else float("nan")

    with right:
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        gx = np.linspace(-3.0, 3.0, 400)
        ax.plot(gx, _true(gx), color="#1D9E75", ls="--", lw=2, label="true function")
        ax.plot(gx, pred(gx), color="#C0507A", lw=2, label=f"degree {eff_deg} fit")
        ax.scatter(xtr, ytr, s=22, color="#5B8FC2", zorder=3, label="train")
        if len(xte):
            ax.scatter(xte, yte, s=34, color="#EF9F27", edgecolors="k",
                       linewidths=0.4, zorder=3, label="test")
        pad = 2.0
        ax.set_ylim(_true(gx).min() - pad, _true(gx).max() + pad)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_title("under- vs over-fitting")
        ax.legend(fontsize=8)
        st.pyplot(fig, width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("Degree", eff_deg)
    c2.metric("Train MSE", f"{tr_mse:.3f}")
    c3.metric("Test MSE", f"{te_mse:.3f}")
    if eff_deg < int(degree):
        st.caption(f"(Degree capped at {eff_deg}: can't fit a degree-{degree} polynomial to {len(xtr)} train points.)")
    st.info("**Low degree → underfit** (high train *and* test error). "
            "**High degree → overfit** (tiny train error, large test error — it memorized noise). "
            "The best model **minimizes test error**.", icon=":material/lightbulb:")

with tab_theory:
    st.markdown(LESSON.theory, unsafe_allow_html=True)

with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(LESSON.quiz, prefix="mlfound")

with tab_tasks:
    st.subheader("Tasks")
    st.markdown(LESSON.tasks)

with tab_ref:
    st.subheader("Reading & references")
    st.markdown(LESSON.references)
