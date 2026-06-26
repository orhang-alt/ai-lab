import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import lessons
import ml_lessons2

LESSON = ml_lessons2.UNSUPERVISED


def _kmeans(X, k, seed, iters=100):
    rng = np.random.default_rng(seed)
    centroids = X[rng.choice(len(X), k, replace=False)].copy()
    assign = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(2)
        assign = d.argmin(1)
        new = np.array([X[assign == j].mean(0) if np.any(assign == j) else centroids[j]
                        for j in range(k)])
        if np.allclose(new, centroids):
            break
        centroids = new
    inertia = float(((X - centroids[assign]) ** 2).sum())
    return assign, centroids, inertia


st.title("M5 · Unsupervised — playground & lesson")
st.caption("No labels — find structure. Watch k-means group points and place centroids; "
           "try the wrong k and see the inertia change.")

tab_play, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🎛 Playground", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

with tab_play:
    left, right = st.columns([0.42, 0.58])
    with left:
        pts = st.slider("points per true cluster", 20, 150, 50, 10, key="u_pts")
        true_k = st.slider("true clusters in the data", 2, 5, 3, key="u_truek")
        spread = st.slider("spread (overlap)", 0.3, 2.0, 0.8, 0.1, key="u_spread")
        k = st.slider("k (clusters to find)", 1, 6, 3, key="u_k")
        seed = st.number_input("seed", 0, 9999, 0, key="u_seed")

    rng = np.random.default_rng(int(seed))
    centers = rng.uniform(-4, 4, (int(true_k), 2))
    X = np.vstack([rng.normal(c, spread, (int(pts), 2)) for c in centers])

    assign, centroids, inertia = _kmeans(X, int(k), int(seed))

    with right:
        fig, ax = plt.subplots(figsize=(4.7, 4.3))
        ax.scatter(X[:, 0], X[:, 1], c=assign, cmap="tab10", s=18, alpha=0.8,
                   vmin=0, vmax=9)
        ax.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="X", s=180,
                   edgecolors="white", linewidths=1.2, zorder=5, label="centroids")
        ax.set_title(f"k-means: found {k} clusters")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_aspect("equal", "datalim")
        st.pyplot(fig, width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("k (chosen)", int(k))
    c2.metric("true clusters", int(true_k))
    c3.metric("inertia", f"{inertia:.0f}")
    st.info("Inertia always **drops as k rises** (more centroids), so you can't just minimize "
            "it — use the **elbow** or **silhouette** to choose k. Try k below/above the true "
            "number and change the seed to see init sensitivity.", icon=":material/lightbulb:")

with tab_theory:
    st.markdown(LESSON.theory)
with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(LESSON.quiz, prefix="mlunsup")
with tab_tasks:
    st.subheader("Tasks")
    st.markdown(LESSON.tasks)
with tab_ref:
    st.subheader("Reading & references")
    st.markdown(LESSON.references)
