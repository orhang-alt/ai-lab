import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import lessons
import math_lessons2

LESSON = math_lessons2.INFORMATION
EPS = 1e-3


def _H(a):
    return float(-a * np.log2(a) - (1 - a) * np.log2(1 - a))


st.title("X5 · Information theory — playground & lesson")
st.caption("Cross-entropy is the classification loss. Set the true p and the model's prediction "
           "q, and watch cross-entropy bottom out exactly when q = p (KL = 0).")

tab_play, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🎛 Playground", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

with tab_play:
    left, right = st.columns([0.42, 0.58])
    with left:
        p = st.slider("true P(class 1)  ·  p", 0.0, 1.0, 0.70, 0.01, key="i_p")
        q = st.slider("model prediction  ·  q", 0.0, 1.0, 0.50, 0.01, key="i_q")
    pc, qc = np.clip(p, EPS, 1 - EPS), np.clip(q, EPS, 1 - EPS)
    Hp = _H(pc)
    CE = float(-pc * np.log2(qc) - (1 - pc) * np.log2(1 - qc))
    KL = CE - Hp

    with right:
        qs = np.linspace(0.01, 0.99, 200)
        ce = -pc * np.log2(qs) - (1 - pc) * np.log2(1 - qs)
        fig, ax = plt.subplots(figsize=(4.7, 4.3))
        ax.plot(qs, ce, color="#185FA5", lw=2, label="cross-entropy H(p,q)")
        ax.axvline(pc, ls="--", color="#1D9E75", lw=1.5, label="q = p (minimum)")
        ax.scatter([qc], [CE], color="#C0507A", s=70, zorder=5, label="your q")
        ax.set_xlabel("model prediction q")
        ax.set_ylabel("bits")
        ax.set_title("cross-entropy vs. prediction")
        ax.legend(fontsize=8)
        st.pyplot(fig, width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("entropy H(p)", f"{Hp:.3f}")
    c2.metric("cross-entropy H(p,q)", f"{CE:.3f}")
    c3.metric("KL(p‖q)", f"{KL:.3f}")
    st.info("Cross-entropy is minimized (= the entropy H(p), so **KL = 0**) exactly when **q = p**. "
            "Being confident and wrong (q → 0 while p is high) makes it **explode** — the same "
            "behavior that trains classifiers in M2 / ANN.", icon=":material/lightbulb:")

with tab_theory:
    st.markdown(LESSON.theory, unsafe_allow_html=True)
with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(LESSON.quiz, prefix="mathinfo")
with tab_tasks:
    st.subheader("Tasks")
    st.markdown(LESSON.tasks)
with tab_ref:
    st.subheader("Reading & references")
    st.markdown(LESSON.references)
