import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import lessons
import ml_lessons

LESSON = ml_lessons.LINEAR_REGRESSION


def _gen(n, tw, tb, noise, seed):
    rng = np.random.default_rng(int(seed))
    x = rng.uniform(-3.0, 3.0, int(n))
    y = tw * x + tb + rng.normal(0.0, noise, int(n))
    return x, y


def _ols(x, y):
    X = np.c_[np.ones_like(x), x]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[1]), float(beta[0])  # (w_hat, b_hat)


def _fit_callback():
    s = st.session_state
    x, y = _gen(s.lr_n, s.lr_tw, s.lr_tb, s.lr_noise, s.lr_seed)
    w_hat, b_hat = _ols(x, y)
    s.lr_w = float(np.clip(round(w_hat, 1), -5.0, 5.0))
    s.lr_b = float(np.clip(round(b_hat, 1), -5.0, 5.0))


st.title("Linear regression — playground & lesson")
st.caption("M1 · ŷ = w·x + b.  Fit a line to data by hand, then let least squares do it perfectly.")

tab_play, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🎛 Playground", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

with tab_play:
    # Model sliders are also driven by the Fit button via session_state, so seed
    # defaults here and omit the value= arg (avoids a Streamlit double-set warning).
    st.session_state.setdefault("lr_w", 1.0)
    st.session_state.setdefault("lr_b", 0.0)

    left, right = st.columns([0.42, 0.58])
    with left:
        st.markdown("**Data**")
        n = st.slider("points", 10, 200, 40, key="lr_n")
        tw = st.slider("true slope", -3.0, 3.0, 1.5, 0.1, key="lr_tw")
        tb = st.slider("true intercept", -3.0, 3.0, 0.5, 0.1, key="lr_tb")
        noise = st.slider("noise σ", 0.0, 3.0, 1.0, 0.1, key="lr_noise")
        seed = st.number_input("seed", 0, 9999, 0, key="lr_seed")
        st.markdown("**Your model**")
        w = st.slider("w (slope)", -5.0, 5.0, step=0.1, key="lr_w")
        b = st.slider("b (intercept)", -5.0, 5.0, step=0.1, key="lr_b")
        show_resid = st.checkbox("show residuals", value=True)
        st.button("Fit (least squares)", icon=":material/auto_fix_high:",
                  type="primary", on_click=_fit_callback)

    x, y = _gen(n, tw, tb, noise, seed)
    w_hat, b_hat = _ols(x, y)

    yhat_user = w * x + b
    mse_user = float(np.mean((y - yhat_user) ** 2))
    mse_ols = float(np.mean((y - (w_hat * x + b_hat)) ** 2))
    ss_res = float(np.sum((y - yhat_user) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2_user = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    with right:
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        xs = np.array([x.min(), x.max()])
        if show_resid:
            ax.vlines(x, np.minimum(y, yhat_user), np.maximum(y, yhat_user),
                      color="#C0507A", alpha=0.25, lw=1, zorder=1)
        ax.scatter(x, y, s=28, color="#5B8FC2", edgecolors="white", linewidths=0.5,
                   zorder=2, label="data")
        ax.plot(xs, w_hat * xs + b_hat, color="#1D9E75", lw=2, ls="--",
                zorder=3, label="least-squares fit")
        ax.plot(xs, w * xs + b, color="#C0507A", lw=2.5, zorder=4, label="your model")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_title("fit a line to the data")
        ax.legend(fontsize=8)
        st.pyplot(fig, width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("Your MSE", f"{mse_user:.3f}")
    c2.metric("Best MSE (OLS)", f"{mse_ols:.3f}")
    c3.metric("Your R²", f"{r2_user:.3f}")
    st.latex(rf"\hat y = {w:.1f}\,x + ({b:.1f}) \qquad \text{{(least squares: }} "
             rf"\hat y = {w_hat:.2f}\,x + ({b_hat:.2f}))")
    st.info("Minimize **Your MSE** by hand, then press **Fit** to jump to the optimum. "
            "Open **Theory** for why least squares is that optimum.", icon=":material/lightbulb:")

with tab_theory:
    st.markdown(LESSON.theory)

with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(LESSON.quiz, prefix="linreg")

with tab_tasks:
    st.subheader("Tasks")
    st.markdown(LESSON.tasks)

with tab_ref:
    st.subheader("Reading & references")
    st.markdown(LESSON.references)
