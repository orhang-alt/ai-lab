import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import lessons
import math_lessons2

LESSON = math_lessons2.NUMERICAL


def _softmax(z):
    with np.errstate(over="ignore", invalid="ignore"):  # the overflow IS the demo
        e = np.exp(z)                  # naive — exp() can overflow to inf
        return e / e.sum()


def _softmax_stable(z):
    e = np.exp(z - z.max())            # log-sum-exp trick
    return e / e.sum()


st.title("X6 · Numerical computing — playground & lesson")
st.caption("Math on a computer isn't exact. Push the logits up and watch the naive softmax "
           "overflow to NaN while the stable (log-sum-exp) version stays correct.")

lessons.predict(
    'Push the logits way up. What does the *naive* softmax do — and how does subtracting the max (log-sum-exp) fix it?',
    "Naive `exp()` of large logits **overflows to inf**, giving `inf/inf = NaN`. Subtracting the max first (**log-sum-exp**) shifts everything ≤ 0 so `exp` stays finite — same result mathematically, but numerically stable. That's what `core/activations.py`'s softmax does.",
)

tab_play, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🎛 Playground", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

with tab_play:
    left, right = st.columns([0.42, 0.58])
    with left:
        M = st.slider("max logit  M", 1, 1000, 50, key="n_m",
                      help="softmax of [0, M/2, M]. exp() overflows around M ≈ 710 (float64).")
    logits = np.array([0.0, M / 2, M])
    naive = _softmax(logits)
    stable = _softmax_stable(logits)
    naive_ok = bool(np.all(np.isfinite(naive)))

    with right:
        fig, ax = plt.subplots(figsize=(4.7, 4.0))
        ax.bar([0, 1, 2], stable, color="#185FA5")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["logit 0", "M/2", "M"])
        ax.set_ylim(0, 1)
        ax.set_title("stable softmax (always valid)")
        st.pyplot(fig, width="stretch")

    c1, c2 = st.columns(2)
    c1.metric("naive softmax", "OK" if naive_ok else "💥 NaN / inf")
    c2.metric("stable softmax sums to", f"{stable.sum():.3f}")
    st.markdown(f"**naive** softmax([0, {M/2:.0f}, {M}]) = `{np.array2string(naive, precision=3)}`")
    st.markdown(f"**stable** = `{np.array2string(stable, precision=3)}`")
    st.caption(f"float curiosities:  0.1 + 0.2 = {0.1 + 0.2!r}   ·   "
               f"float32 machine-ε ≈ {np.finfo(np.float32).eps:.2e}")
    st.info("Large logits make naive `exp()` overflow → `inf/inf = NaN`. Subtracting the max "
            "first (**log-sum-exp**) keeps it finite — exactly what `core/activations.py`'s "
            "stable sigmoid/softmax does.", icon=":material/lightbulb:")

with tab_theory:
    st.markdown(LESSON.theory, unsafe_allow_html=True)
with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(LESSON.quiz, prefix="mathnum")
with tab_tasks:
    st.subheader("Tasks")
    st.markdown(LESSON.tasks)
with tab_ref:
    st.subheader("Reading & references")
    st.markdown(LESSON.references)
