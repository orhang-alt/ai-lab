import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import lessons
import math_lessons2

LESSON = math_lessons2.OPTIMIZATION
START = np.array([-4.0, 4.0])

st.title("X4 · Optimization — playground & lesson")
st.caption("Watch gradient descent roll downhill on f(x,y) = x² + b·y². Push the learning rate "
           "until it diverges; raise the anisotropy to see it zig-zag.")

tab_play, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🎛 Playground", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

with tab_play:
    left, right = st.columns([0.42, 0.58])
    with left:
        lr = st.slider("learning rate η", 0.01, 1.10, 0.10, 0.01, key="o_lr")
        b = st.slider("anisotropy b (ravine)", 1.0, 20.0, 1.0, 1.0, key="o_b",
                      help="f = x² + b·y². Larger b → a steep, narrow ravine.")
        steps = st.slider("steps", 5, 100, 30, key="o_steps")

    def f(x, y):
        return x ** 2 + b * y ** 2

    p = START.copy().astype(float)
    path = [p.copy()]
    losses = [float(f(*p))]
    diverged = False
    for _ in range(int(steps)):
        g = np.array([2 * p[0], 2 * b * p[1]])     # gradient of f
        p = p - lr * g                              # GD update
        if not np.all(np.isfinite(p)) or np.abs(p).max() > 1e6:
            diverged = True
            break
        path.append(p.copy())
        losses.append(float(f(*p)))
    path = np.array(path)

    with right:
        g = np.linspace(-5, 5, 200)
        xx, yy = np.meshgrid(g, g)
        fig, ax = plt.subplots(figsize=(4.7, 4.3))
        ax.contour(xx, yy, f(xx, yy), levels=20, cmap="Blues", linewidths=0.8)
        vis = np.clip(path, -5, 5)
        ax.plot(vis[:, 0], vis[:, 1], "-o", color="#C0507A", ms=3, lw=1.3, label="GD path")
        ax.scatter([START[0]], [START[1]], color="#EF9F27", s=70, zorder=5, label="start")
        ax.scatter([0], [0], marker="*", color="#1D9E75", s=200, zorder=5, label="minimum")
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
        ax.set_title("loss surface + descent path")
        ax.legend(fontsize=8, loc="upper right")
        st.pyplot(fig, width="stretch")

    grew = losses[-1] > losses[0]
    if diverged or grew:
        status = "💥 diverging"
    elif losses[-1] < 0.05:
        status = "✅ converged"
    else:
        status = "→ descending"
    c1, c2, c3 = st.columns(3)
    c1.metric("Final loss", "overflow" if diverged else f"{losses[-1]:.3f}")
    c2.metric("Steps taken", len(path) - 1)
    c3.metric("Status", status)
    st.line_chart({"loss": [min(l, 1e6) for l in losses]})
    st.info("Too-large η **overshoots and diverges**; tiny η **crawls**. With b ≫ 1 the bowl "
            "becomes a ravine and GD **zig-zags** — which is why feature scaling (M7) and "
            "momentum/Adam (§6) matter.", icon=":material/lightbulb:")

with tab_theory:
    st.markdown(LESSON.theory)
with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(LESSON.quiz, prefix="mathopt")
with tab_tasks:
    st.subheader("Tasks")
    st.markdown(LESSON.tasks)
with tab_ref:
    st.subheader("Reading & references")
    st.markdown(LESSON.references)
