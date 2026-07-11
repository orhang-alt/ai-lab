import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import lessons
import ml_lessons2

LESSON = ml_lessons2.TREES


def _gini(y):
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=2) / len(y)
    return 1.0 - float((p ** 2).sum())


class _Node:
    __slots__ = ("feat", "thr", "left", "right", "pred")

    def __init__(self):
        self.feat = None
        self.thr = 0.0
        self.left = self.right = None
        self.pred = 0


def _build(X, y, depth, max_depth, min_samples=5):
    node = _Node()
    node.pred = int(2 * int(y.sum()) >= len(y)) if len(y) else 0  # majority (ties → 1)
    if depth >= max_depth or len(y) < min_samples or len(np.unique(y)) < 2:
        return node
    parent, best_gain, bf, bt = _gini(y), 1e-9, None, 0.0
    for f in range(X.shape[1]):
        vals = np.unique(X[:, f])
        if len(vals) < 2:
            continue
        for t in (vals[:-1] + vals[1:]) / 2:
            lm = X[:, f] <= t
            nl = int(lm.sum())
            nr = len(y) - nl
            if nl == 0 or nr == 0:
                continue
            imp = (nl * _gini(y[lm]) + nr * _gini(y[~lm])) / len(y)
            if parent - imp > best_gain:
                best_gain, bf, bt = parent - imp, f, t
    if bf is None:
        return node
    node.feat, node.thr = bf, bt
    lm = X[:, bf] <= bt
    node.left = _build(X[lm], y[lm], depth + 1, max_depth, min_samples)
    node.right = _build(X[~lm], y[~lm], depth + 1, max_depth, min_samples)
    return node


def _predict(node, X):
    out = np.empty(len(X), dtype=int)
    for i, x in enumerate(X):
        nd = node
        while nd.feat is not None:
            nd = nd.left if x[nd.feat] <= nd.thr else nd.right
        out[i] = nd.pred
    return out


st.title("M3 · Trees & ensembles — playground & lesson")
st.caption("A decision tree carves space with axis-aligned splits. Raise the depth and watch "
           "it go from underfit → good → overfit (train ↑, test ↓).")

lessons.predict(
    'Raise the tree **depth** on the checkerboard. What happens to train vs test accuracy — and what would a random forest change?',
    'Shallow **underfits**; a few levels capture it; very deep **overfits** — train accuracy climbs toward 100% while test accuracy falls (it fences off single noisy points). A **random forest** averages many decorrelated trees, cutting that variance and stabilizing test accuracy.',
)

tab_play, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🎛 Playground", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

with tab_play:
    left, right = st.columns([0.42, 0.58])
    with left:
        n = st.slider("data points", 40, 400, 160, 20, key="t_n")
        noise = st.slider("label noise", 0.0, 0.4, 0.1, 0.02, key="t_noise")
        max_depth = st.slider("max tree depth", 1, 10, 4, key="t_depth")
        seed = st.number_input("seed", 0, 9999, 0, key="t_seed")

    rng = np.random.default_rng(int(seed))
    X = rng.uniform(-3, 3, (int(n), 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)          # XOR / checkerboard
    flip = rng.random(len(y)) < noise
    y = np.where(flip, 1 - y, y)

    idx = rng.permutation(len(y))
    n_tr = int(0.7 * len(y))
    tr, te = idx[:n_tr], idx[n_tr:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    tree = _build(Xtr, ytr, 0, int(max_depth))
    acc_tr = float((_predict(tree, Xtr) == ytr).mean())
    acc_te = float((_predict(tree, Xte) == yte).mean()) if len(yte) else float("nan")

    with right:
        g = np.linspace(-3.2, 3.2, 140)
        xx, yy = np.meshgrid(g, g)
        zz = _predict(tree, np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        fig, ax = plt.subplots(figsize=(4.7, 4.3))
        ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], cmap="RdBu", alpha=0.45)
        ax.scatter(Xtr[ytr == 0, 0], Xtr[ytr == 0, 1], s=16, color="#A32D2D", label="class 0")
        ax.scatter(Xtr[ytr == 1, 0], Xtr[ytr == 1, 1], s=16, color="#185FA5", label="class 1")
        ax.set_title(f"decision regions (depth {max_depth})")
        ax.legend(fontsize=8, loc="upper right")
        st.pyplot(fig, width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("Depth", int(max_depth))
    c2.metric("Train accuracy", f"{acc_tr:.2f}")
    c3.metric("Test accuracy", f"{acc_te:.2f}")
    st.info("Depth 1–2 **underfits** the checkerboard; a few levels capture it; very deep "
            "**overfits** the noise — train accuracy keeps climbing while test accuracy falls. "
            "An ensemble (random forest) would stabilize this.", icon=":material/lightbulb:")

with tab_theory:
    st.markdown(LESSON.theory, unsafe_allow_html=True)
with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(LESSON.quiz, prefix="mltrees")
with tab_tasks:
    st.subheader("Tasks")
    st.markdown(LESSON.tasks)
    st.divider()
    st.markdown("#### ✅ Worked solutions")
    st.caption("Attempt each first, then check.")
    lessons.solution(
        r"""**1.** Depth 1 underfits (a single axis split); test accuracy peaks after a few levels; very deep → train accuracy climbs toward 100% while **test** accuracy falls (it fences off single noisy points).

**2.** More label noise → the best depth gets **shallower**: less capacity to memorize the noise.""",
        label="Playground 1–2",
    )
    lessons.solution(
        r"""**3.** Gini of $[8,2]$: $1-(0.8^2+0.2^2)=1-0.68=\mathbf{0.32}$.

**4.** Parent Gini $=0.32$. Children: $[6,0]$ → Gini $0$ (weight $0.6$); $[2,2]$ → Gini $1-(0.5^2+0.5^2)=0.5$ (weight $0.4$). Weighted child $=0.6\cdot0+0.4\cdot0.5=0.20$. **Information gain $=0.32-0.20=0.12$.**

**5.** A deep tree can isolate individual training points in their own leaves, so tiny data changes reshuffle the splits — **high variance**.""",
        label="Pencil & paper 3–5",
    )
    lessons.solution(
        r"""**6–8.** A Gini + best-split **stump**, then **bagging** 20 stumps on bootstrap samples cuts variance vs. a single tree; `sklearn`'s decision-tree / random-forest / gradient-boosting compare directly.

**9.** Logistic regression draws one **straight** line, so it fails on XOR (not linearly separable); the tree's **axis-aligned staircase** can carve XOR out.""",
        label="Code & bridge 6–9",
    )
with tab_ref:
    st.subheader("Reading & references")
    st.markdown(LESSON.references)
