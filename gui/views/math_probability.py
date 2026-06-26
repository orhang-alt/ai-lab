import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import lessons
import math_lessons2

LESSON = math_lessons2.PROBABILITY
DISTS = {
    "Uniform [0,1]": lambda rng, shape: rng.uniform(0, 1, shape),
    "Exponential": lambda rng, shape: rng.exponential(1.0, shape),
    "Bernoulli(0.3)": lambda rng, shape: (rng.random(shape) < 0.3).astype(float),
}

st.title("X3 · Probability — playground & lesson")
st.caption("The Central Limit Theorem: average n samples from ANY distribution and the average "
           "becomes Gaussian as n grows. Slide n up and watch the bell appear.")

tab_play, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🎛 Playground", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

with tab_play:
    left, right = st.columns([0.42, 0.58])
    with left:
        name = st.selectbox("base distribution (not Gaussian!)", list(DISTS))
        n = st.slider("sample size n (averaged)", 1, 50, 5, key="p_n")
        seed = st.number_input("seed", 0, 9999, 0, key="p_seed")
        trials = 4000

    rng = np.random.default_rng(int(seed))
    samples = DISTS[name](rng, (trials, int(n)))
    means = samples.mean(axis=1)
    m, s = float(means.mean()), float(means.std())

    with right:
        fig, ax = plt.subplots(figsize=(4.7, 4.3))
        ax.hist(means, bins=40, density=True, color="#185FA5", alpha=0.75,
                label=f"mean of n={n}")
        if s > 1e-9:
            xs = np.linspace(means.min(), means.max(), 200)
            pdf = np.exp(-(xs - m) ** 2 / (2 * s ** 2)) / (s * np.sqrt(2 * np.pi))
            ax.plot(xs, pdf, color="#C0507A", lw=2, label="Gaussian fit")
        ax.set_title("distribution of the sample mean")
        ax.legend(fontsize=8)
        st.pyplot(fig, width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("n (averaged)", int(n))
    c2.metric("mean of means", f"{m:.3f}")
    c3.metric("std of means", f"{s:.3f}")
    st.info("At **n = 1** you see the base distribution's true (often skewed) shape. As **n "
            "grows**, the average becomes **Gaussian** (CLT) and its spread shrinks ~1/√n — "
            "this is why the normal distribution is everywhere.", icon=":material/lightbulb:")

with tab_theory:
    st.markdown(LESSON.theory)
with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(LESSON.quiz, prefix="mathprob")
with tab_tasks:
    st.subheader("Tasks")
    st.markdown(LESSON.tasks)
with tab_ref:
    st.subheader("Reading & references")
    st.markdown(LESSON.references)
