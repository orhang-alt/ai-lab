"""Deep nets on 2-D data (ANN module).

The MLP page trains the lab's own engine on XOR; here the *same idea* — a multilayer
perceptron with a nonlinearity — is scaled up (via scikit-learn's fast MLP) to real 2-D
shapes: two moons, concentric circles, a spiral. Watch the decision boundary curve to fit
as you change the architecture and activation.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # gui/

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import lessons

ARCHS = {"1 × 8": (8,), "1 × 32": (32,), "2 × 16": (16, 16), "3 × 24": (24, 24, 24)}


def _spiral(n, noise, seed):
    rng = np.random.default_rng(seed)
    k = n // 2
    t = np.sqrt(rng.random(k)) * 2.6 * np.pi
    a = np.c_[t * np.cos(t), t * np.sin(t)]
    b = np.c_[-t * np.cos(t), -t * np.sin(t)]
    X = np.vstack([a, b]) + noise * 2.2 * rng.standard_normal((2 * k, 2))
    X = X / np.abs(X).max() * 3.0
    return X, np.r_[np.zeros(k), np.ones(k)].astype(int)


def _dataset(name, noise, seed):
    if name == "Two moons":
        return make_moons(n_samples=260, noise=noise, random_state=seed)
    if name == "Concentric circles":
        return make_circles(n_samples=260, noise=noise, factor=0.45, random_state=seed)
    return _spiral(260, noise, seed)


@st.cache_data(show_spinner=False)
def fit(name, arch_key, activation, noise, seed):
    X, y = _dataset(name, noise, seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    clf = make_pipeline(StandardScaler(),
                        MLPClassifier(hidden_layer_sizes=ARCHS[arch_key], activation=activation,
                                      alpha=1e-4, max_iter=2500, random_state=seed))
    clf.fit(Xtr, ytr)
    pad = 0.6
    gx = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 70)
    gy = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 70)
    xx, yy = np.meshgrid(gx, gy)
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
    return {
        "Xtr": Xtr.tolist(), "ytr": ytr.tolist(), "Xte": Xte.tolist(), "yte": yte.tolist(),
        "gx": gx.tolist(), "gy": gy.tolist(), "Z": Z.tolist(),
        "train": clf.score(Xtr, ytr), "test": clf.score(Xte, yte),
    }


_THEORY = r"""
## 1. Beyond XOR — curved boundaries

XOR (MLP page) is the smallest non-linear problem. The same machine — a hidden layer plus
a nonlinearity — scales straight up to **moons, rings, and spirals**: with enough hidden
units it can bend its boundary into *any* shape (the universal-approximation idea, MLP §2).
This page uses scikit-learn's fast, vectorized MLP so you can train on hundreds of points
in real time.

## 2. Width & depth = capacity

More hidden **units** (width) and more **layers** (depth) give the network more "folds" to
work with, so it can fit more intricate boundaries — a circle, then a double spiral.
But capacity cuts both ways: too much and the model starts tracing the **noise** instead of
the signal (overfitting, M0). The next page (**Regularization**) is about taming that.

## 3. Activation shapes the boundary

- **ReLU** — piecewise-linear, so boundaries look **polygonal** (made of straight
  segments); cheap and the deep-learning default.
- **tanh / logistic** — smooth, so boundaries are **curvy**; can saturate in deep stacks
  (ANN §6).

## 4. Same recipe underneath

It's still forward → loss → backward → optimizer step (Backprop + Optimizers pages), just
on more data and with a tuned solver. The lesson: **depth + nonlinearity learns features**
that make even tangled classes separable. *(Lab tie-ins: MLP, M0, M7.)*
"""

_QUIZ = [
    lessons.Question(
        "Why can an MLP separate two interleaved spirals when a line cannot?",
        ["it uses more data", "hidden units + a nonlinearity bend the boundary into an arbitrary curved shape",
         "it removes outliers", "spirals are actually linear"], 1,
        "With enough hidden capacity an MLP approximates any boundary — straight lines need not apply."),
    lessons.Question(
        "Increasing hidden width/depth mainly increases…",
        ["the learning rate", "model capacity — the complexity of boundary it can represent (and the overfitting risk)",
         "the number of classes", "the input dimension"], 1,
        "More units/layers = more capacity; great until it starts fitting noise (→ regularization)."),
    lessons.Question(
        "ReLU vs tanh boundaries tend to look…",
        ["identical", "ReLU polygonal (piecewise-linear), tanh smooth/curvy",
         "ReLU circular, tanh square", "both perfectly straight"], 1,
        "ReLU is piecewise-linear so regions are polygonal; tanh is smooth so boundaries curve."),
]

_TASKS = r"""
### In the Train tab
1. On **Two moons**, compare `1 × 8` vs `2 × 16` — does the extra capacity give a cleaner
   boundary, or just more wiggle?
2. Switch to **Spiral** — what's the smallest architecture that separates it? Watch train
   vs **test** accuracy as you add capacity.
3. Toggle **ReLU ↔ tanh** on the same data — describe how the boundary's *texture* changes.
4. Crank the **noise** up — when does even a big network's test accuracy fall? (overfitting.)

### Bridge
5. Connect to **MLP** (the same net on XOR, on the lab's engine), **Optimizers** (how it's
   trained), and **M0/M7** (capacity vs. overfitting).
"""

_REFS = r"""
- **TensorFlow Playground** (playground.tensorflow.org) — the classic interactive version of this.
- scikit-learn — [neural_network.MLPClassifier](https://scikit-learn.org/stable/modules/neural_networks_supervised.html).
- Nielsen — *Neural Networks and Deep Learning*, ch. 4 (a visual proof MLPs can fit any function).
- In this lab: **MLP**, **Backprop**, **Optimizers**, and ML **M0** (over/underfitting).
"""


st.title("Deep nets on 2-D data")
st.caption("The MLP idea scaled past XOR — watch a multilayer perceptron bend its boundary "
           "to fit moons, rings and spirals (scikit-learn's fast MLP).")

tab_train, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🌀 Train it", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

with tab_train:
    c = st.columns(4)
    name = c[0].selectbox("dataset", ["Two moons", "Concentric circles", "Spiral"], key="dp_ds")
    arch_key = c[1].selectbox("hidden layers", list(ARCHS), index=2, key="dp_arch")
    activation = c[2].selectbox("activation", ["relu", "tanh", "logistic"], key="dp_act")
    noise = c[3].select_slider("noise", [0.05, 0.1, 0.2, 0.3], value=0.2, key="dp_noise")
    seed = st.slider("seed", 0, 30, 0, key="dp_seed")

    r = fit(name, arch_key, activation, noise, int(seed))

    m = st.columns(2)
    m[0].metric("train accuracy", f"{r['train']:.0%}")
    m[1].metric("test accuracy", f"{r['test']:.0%}", help="held-out 30% — the honest score")

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    xx, yy = np.meshgrid(np.array(r["gx"]), np.array(r["gy"]))
    ax.contourf(xx, yy, np.array(r["Z"]), levels=20, cmap="RdBu", alpha=0.7, vmin=0, vmax=1)
    ax.contour(xx, yy, np.array(r["Z"]), levels=[0.5], colors="k", linewidths=1.3)
    cmap = ListedColormap(["#A32D2D", "#185FA5"])
    Xtr, ytr = np.array(r["Xtr"]), np.array(r["ytr"])
    Xte, yte = np.array(r["Xte"]), np.array(r["yte"])
    ax.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr, cmap=cmap, s=18, edgecolors="k", linewidths=0.3)
    ax.scatter(Xte[:, 0], Xte[:, 1], c=yte, cmap=cmap, s=42, marker="^",
               edgecolors="k", linewidths=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("● train  ▲ test  ·  boundary = where p = 0.5", fontsize=10)
    st.pyplot(fig, width="stretch")

    gap = r["train"] - r["test"]
    if gap > 0.12:
        st.warning(f"Train beats test by {gap:.0%} — the network is **overfitting** (tracing "
                   "noise). Fewer units or **regularization** (next page) would help.",
                   icon=":material/warning:")
    else:
        st.success("Train and test agree — the boundary captures the real shape, not the noise.",
                   icon=":material/check_circle:")
    st.info("Same MLP idea as the XOR page, scaled to real 2-D data with scikit-learn's "
            "vectorized solver. Capacity (width/depth) lets it bend the boundary; too much "
            "and it overfits.", icon=":material/blur_on:")

with tab_theory:
    st.markdown(_THEORY, unsafe_allow_html=True)

with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(_QUIZ, prefix="deepplay")

with tab_tasks:
    st.subheader("Tasks")
    st.markdown(_TASKS)

with tab_ref:
    st.subheader("Reading & references")
    st.markdown(_REFS)
