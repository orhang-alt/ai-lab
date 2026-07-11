import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import lessons
import math_lessons

LESSON = math_lessons.VECTORS

st.title("Vectors & the dot product — playground & lesson")
st.caption("X1 · w·x is a neuron's pre-activation. Drag two vectors and watch the dot product, "
           "angle, and projection.")

lessons.predict(
    'Drag **x** to point the *same* way as **w**, then perpendicular, then opposite. What does the dot product w·x do at each — and where is it largest?',
    "Largest (positive) when **aligned** (θ=0), **zero** when perpendicular (θ=90°), negative when opposed: w·x = ‖w‖‖x‖cos θ. That's exactly a neuron's pre-activation z = w·x + b — the neuron scores how much the input aligns with its weight direction.",
)

tab_play, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🎛 Playground", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

with tab_play:
    left, right = st.columns([0.42, 0.58])
    with left:
        st.markdown("**Vector w**")
        w1 = st.slider("w₁", -5.0, 5.0, 3.0, 0.1)
        w2 = st.slider("w₂", -5.0, 5.0, 1.0, 0.1)
        st.markdown("**Vector x**")
        x1 = st.slider("x₁", -5.0, 5.0, 1.0, 0.1)
        x2 = st.slider("x₂", -5.0, 5.0, 2.0, 0.1)

    w = np.array([w1, w2])
    x = np.array([x1, x2])
    dot = float(w @ x)
    nw, nx = float(np.linalg.norm(w)), float(np.linalg.norm(x))
    cos = max(-1.0, min(1.0, dot / (nw * nx))) if nw > 0 and nx > 0 else 0.0
    theta = float(np.degrees(np.arccos(cos))) if nw > 0 and nx > 0 else 0.0
    scalar_proj = dot / nw if nw > 0 else 0.0

    with right:
        fig, ax = plt.subplots(figsize=(4.6, 4.4))
        lim = 5.6
        ax.axhline(0, color="0.85", lw=0.8)
        ax.axvline(0, color="0.85", lw=0.8)
        ax.annotate("", xy=(w1, w2), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#C0507A", lw=2.5))
        ax.annotate("", xy=(x1, x2), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#1D6FB8", lw=2.5))
        ax.text(w1, w2, "  w", color="#C0507A", fontweight="bold")
        ax.text(x1, x2, "  x", color="#1D6FB8", fontweight="bold")
        if nw > 0:
            p = (dot / (nw * nw)) * w  # projection of x onto w
            ax.plot([p[0]], [p[1]], "o", color="#1D9E75")
            ax.plot([x1, p[0]], [x2, p[1]], "--", color="#1D9E75", lw=1)
            ax.text(p[0], p[1], "  proj", color="#1D9E75", fontsize=9)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title("w·x = |w||x|cos θ")
        st.pyplot(fig, width="stretch")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("w · x", f"{dot:.2f}")
    c2.metric("angle θ", f"{theta:.0f}°")
    c3.metric("cos θ", f"{cos:.2f}")
    c4.metric("proj of x on w", f"{scalar_proj:.2f}")
    st.latex(rf"w\cdot x = ({w1:.1f})({x1:.1f}) + ({w2:.1f})({x2:.1f}) = {dot:.2f}")
    st.info("This is exactly a neuron's pre-activation z = w·x (the bias just shifts it). "
            "Maximum when x points the same way as w (θ=0°); zero when perpendicular (θ=90°); "
            "negative when opposed.", icon=":material/lightbulb:")

with tab_theory:
    st.markdown(LESSON.theory, unsafe_allow_html=True)

with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(LESSON.quiz, prefix="vectors")

with tab_tasks:
    st.subheader("Tasks")
    st.markdown(LESSON.tasks)

with tab_ref:
    st.subheader("Reading & references")
    st.markdown(LESSON.references)
